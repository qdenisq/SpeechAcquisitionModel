import json
import yaml
from pprint import pprint
import datetime
import copy
import os
from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from src.reinforcement_v2.envs.env import EnvironmentManager

from src.reinforcement_v2.common.replay_buffer import ReplayBuffer
from src.reinforcement_v2.common.nn import SoftQNetwork, PolicyNetwork
from src.reinforcement_v2.common.tensorboard import DoubleSummaryWriter
from src.reinforcement_v2.common.noise import OUNoise
from src.reinforcement_v2.utils.timer import Timer

from src.siamese_net_sound_similarity.train_v2 import SiameseDeepLSTMNet


class AsyncrhonousSoftActorCritic(nn.Module):
    def __init__(self, *args, **kwargs):
        super(AsyncrhonousSoftActorCritic, self).__init__()

        self.__dt = str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p"))
        self.__env_name = kwargs['env']['env_id']

        self.params = kwargs
        self.use_cuda = kwargs.get('use_cuda', True)
        self.device = torch.device("cuda" if self.use_cuda and torch.cuda.is_available() else "cpu")
        self.soft_q_net1 = SoftQNetwork(**kwargs['soft_q_network']).to(self.device)
        self.soft_q_net2 = SoftQNetwork(**kwargs['soft_q_network']).to(self.device)

        self.target_soft_q_net1 = SoftQNetwork(**kwargs['soft_q_network']).to(self.device)
        self.target_soft_q_net2 = SoftQNetwork(**kwargs['soft_q_network']).to(self.device)

        if kwargs["pretrained_policy"] is not None:
            self.policy_net = torch.load(kwargs['pretrained_policy']).policy_net
            self.policy_net.to(self.device)
        else:
            self.policy_net = PolicyNetwork(**kwargs['policy_network']).to(self.device)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.use_alpha = kwargs.get('use_alpha', True)
        if not self.use_alpha:
            self.noise = OUNoise(kwargs['soft_q_network']['action_dim'] * kwargs['env']['num_workers'],
                                 kwargs['env']['seed'])
            self.noise_decay = kwargs['noise_decay']  # decay after each 1000 steps
            self.noise_level = 1.0
            self.noise_min = kwargs['noise_min']

        t_entropy = -kwargs['soft_q_network']['action_dim']
        self.target_entropy = kwargs.get('target_entropy', t_entropy)
        self.log_alpha = 0 * torch.ones(1).to(self.device)
        self.log_alpha.requires_grad_()

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()
        self.lr = kwargs['lr']

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=self.lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=self.lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)

        self.alpha_l1 = kwargs.get('alpha_l1', 1e-3)
        self.alpha_l2 = kwargs.get('alpha_l2', 1e-3)

        # init replay buffer
        self.replay_buffer_size = kwargs['replay_buffer']['size']

        init_data_fname = kwargs['init_data_fname']

        self.replay_buffer_csv_filename = None
        if kwargs['collect_data']:
            self.replay_buffer_csv_filename = f'../../../data/{self.__env_name}_sac_data_{self.__dt}.csv'

        self.replay_buffer = ReplayBuffer(self.replay_buffer_size, self.replay_buffer_csv_filename, init_data_fname)

        self.gamma = kwargs.get('gamma', 0.99)
        self.soft_tau = kwargs.get('soft_tau', 5e-3)
        self.soft_update_period = kwargs.get('soft_update_period', 5)
        self.num_updates_per_step = kwargs.get("num_updates_per_step", 5)
        self.train_step = 0
        self.frame_idx = 0
        # ???
        self.init_pose = None

    def forward(self, *input):
        return self.policy_net(*input)

    def update(self, batch_size):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        """
        Policy loss
        """
        new_action, log_prob, epsilon, mean, log_std = self.policy_net.evaluate(state)

        if self.use_alpha:
            alpha_loss = -(self.log_alpha * (log_prob.sum(dim=1, keepdim=True) + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss = alpha_loss.detach().cpu().numpy()
            alpha = self.log_alpha.exp()
        else:
            alpha = 0.0
            alpha_loss = 0

        predicted_q_value1 = self.soft_q_net1(state, new_action)
        predicted_q_value2 = self.soft_q_net2(state, new_action)
        predicted_q_value = torch.min(predicted_q_value1, predicted_q_value2)

        policy_loss = (alpha * log_prob.sum(dim=1, keepdim=True) - predicted_q_value).mean()

        """
        Q function loss
        """
        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        new_next_action, new_log_prob, epsilon, mean, log_std = self.policy_net.evaluate(next_state)

        target_q_value = torch.min(
            self.target_soft_q_net1(next_state, new_next_action),
            self.target_soft_q_net2(next_state, new_next_action),
        ) - alpha * new_log_prob.sum(dim=1, keepdim=True)
        target_q_value = reward + (1. - done) * self.gamma * target_q_value
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())


        """
        Update them all
        """
        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()

        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        total_loss = policy_loss
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        self.policy_optimizer.step()

        """
        Soft update target bad boys
        """
        if self.train_step % self.soft_update_period == 0:
            self.soft_update(self.soft_q_net1, self.target_soft_q_net1, self.soft_tau)
            self.soft_update(self.soft_q_net2, self.target_soft_q_net2, self.soft_tau)

        self.train_step += 1
        return self.train_step, policy_loss.detach().cpu(), q_value_loss1.detach().cpu(), q_value_loss2.detach().cpu(),\
               alpha_loss, alpha

    def soft_update(self, from_net, target_net, soft_tau):
        for target_param, param in zip(target_net.parameters(), from_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

    def add_weights_histogram(self, writer, global_step):
        for name, param in self.policy_net.named_parameters():
            writer.add_histogram("Policynet_" + name, param, global_step)

        for name, param in self.soft_q_net1.named_parameters():
            writer.add_histogram("Q1net_" + name, param, global_step)

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries or unnecessary objects.
        del state['replay_buffer']
        return state

    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)
        # Do not restore replay buffer
        # self.replay_buffer = ReplayBuffer(self.replay_buffer_size,
        #                                   self.replay_buffer_csv_filename,
        #                                   self.replay_buffer_csv_filename)
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size, None, None)
        # Get device
        self.device = torch.device("cuda" if self.use_cuda and torch.cuda.is_available() else "cpu")
        # Move all the networks to device
        self.soft_q_net1.to(self.device)
        self.soft_q_net2.to(self.device)
        self.target_soft_q_net1.to(self.device)
        self.target_soft_q_net2.to(self.device)
        self.policy_net.to(self.device)
        self.log_alpha.to(self.device)

    def train(self, env, **kwargs):
        """
        Initialize utility objects (e.g. summary writers, timers)
        """

        timer = defaultdict(Timer)
        writer = DoubleSummaryWriter(log_dir=f'../../../runs/{self.__env_name}_asac_{self.__dt}/',
                                     light_log_dir=f'../../../runs_light/light_{self.__env_name}_asac_{self.__dt}/',
                                     mode=kwargs['log_mode'])
        # TODO: only last called graph is added
        dummy_input = torch.rand(1, self.soft_q_net1.state_dim).to(self.device)
        writer.add_graph(self.policy_net, dummy_input)

        max_frames = kwargs.get('max_frames', 40000)
        max_steps = kwargs.get('max_steps', 100)
        batch_size = kwargs.get('batch_size', 256)

        self.params['soft_tau'] = self.soft_tau
        self.params['gamma'] = self.gamma
        self.params['use_alpha'] = self.use_alpha
        self.params['target_entropy'] = self.target_entropy
        self.params['soft_update_period'] = self.soft_update_period
        self.params['train'] = {
            'max_frames': max_frames,
            'max_steps': max_steps,
            'batch_size': batch_size
        }

        writer.add_text('Hyper parameters', json.dumps(self.params, indent=4, sort_keys=True))

        """
        Train
        """

        rewards = []
        best_total_reward = -10000.
        reward_running = 0.
        while self.frame_idx < max_frames:
            ep_reward = np.zeros(kwargs['num_workers'])
            state = env.reset()
            if not self.use_alpha:
                self.noise.reset()
                self.noise_level = max(self.noise_min, self.noise_level * self.noise_decay)

            for step in range(max_steps):
                timer['algo'].start()
                action, pi_mean, pi_log_std, log_prob = self.policy_net.get_action(state)
                log_prob = log_prob.sum(dim=1).detach().cpu().numpy()
                action = action.detach()
                # ??? not sure if pi_entropy is calculated correctly
                pi_entropy = pi_log_std.sum(dim=1, keepdim=True)
                pi_entropy = pi_entropy.detach().cpu().numpy()
                pi_mean = pi_mean.detach().cpu().mean().numpy()
                pi_log_std = pi_log_std.detach().cpu().mean().numpy()
                # log_prob = Normal(pi_mean, pi_log_std.exp()).log_prob(action.to(self.device)).sum(dim=1).detach().cpu().numpy()
                action = action.numpy()
                if not self.use_alpha:
                    action = action + self.noise.sample().reshape(*action.shape) * self.noise_level
                    # action = np.clip(action, 0, 1)

                timer['algo'].stop()
                timer['env'].start()

                next_state, reward, done, _ = env.step(action)
                if kwargs['visualize']:
                    env.render()

                timer['env'].stop()
                timer['replay_buffer'].start()

                for worker_idx in range(kwargs['num_workers']):
                    # skip steps with undefined reward
                    if np.isnan(reward[worker_idx]):
                        continue
                    self.replay_buffer.push((list(state[worker_idx]),
                                             list(action[worker_idx]),
                                             reward[worker_idx],
                                             list(next_state[worker_idx]),
                                             done[worker_idx]))

                timer['replay_buffer'].stop()

                state = next_state
                ep_reward += reward
                self.frame_idx += 1

                timer['utils'].start()
                writer.add_histogram("Action", action, self.frame_idx)
                timer['utils'].stop()

                if len(self.replay_buffer) > 1 * batch_size:
                    timer['algo'].start()
                    for _ in range(self.num_updates_per_step):
                        _, policy_loss, q_value_loss1, q_value_loss2, alpha_loss, alpha = self.update(batch_size)
                    timer['algo'].stop()

                    timer['utils'].start()
                    if not self.use_alpha:
                        writer.add_scalar('Noise_level', self.noise_level, self.frame_idx)
                    writer.add_scalar('Alpha', alpha, self.frame_idx)
                    writer.add_scalar('Alpha_loss', alpha_loss, self.frame_idx)
                    writer.add_scalar('Policy_loss', policy_loss, self.frame_idx)
                    writer.add_scalar('Q_loss_1', q_value_loss1, self.frame_idx)
                    writer.add_scalar('Q_loss_2', q_value_loss2, self.frame_idx)
                    writer.add_scalar('Policy entropy', pi_entropy.mean(), self.frame_idx)
                    writer.add_scalar('Policy mean action', pi_mean, self.frame_idx)
                    writer.add_scalar('Policy log std action', pi_log_std, self.frame_idx)
                    writer.add_scalar('Action probability', log_prob.mean(), self.frame_idx)

                    # explicitly remove all tensor-related variables just to ensure well-behaved memory management
                    del alpha, alpha_loss, policy_loss, q_value_loss1, q_value_loss2, pi_entropy,\
                        pi_mean, pi_log_std, log_prob, action


                    timer['utils'].stop()


                if any(done):
                    timer['utils'].start()
                    for i in range(kwargs['num_workers']):
                        if done[i]:
                            reward_running = 0.90 * reward_running + 0.10 * ep_reward[i]
                            rewards.append(ep_reward[i])
                            ep_reward[i] = 0.
                            state[i] = env.reset([i])

                    writer.add_scalar("Reward", reward_running, self.frame_idx)
                    for k, v in timer.items():
                        writer.add_scalar(f"Elapsed time: {k}", v.elapsed_time, self.frame_idx)
                    self.add_weights_histogram(writer, self.frame_idx)
                    timer['utils'].stop()
                    # break

            # save best performing policy
            if reward_running > best_total_reward or self.frame_idx % 10000 == 0:
                timer['utils'].start()
                # self.save_networks(episode_reward)
                name = f'{self.__env_name}_AsyncSoftActorCritic_' + f'{reward_running:.2f}'
                path_agent = f'../../../models/{name}.asac'
                torch.save(self, path_agent)
                timer['utils'].stop()
                print(f'step={self.frame_idx} | reward_avg={reward_running:.2f} | saving agent: {name}')
                if reward_running > best_total_reward:
                    best_total_reward = reward_running

            if self.frame_idx % 100 == 0:
                print(f'step={self.frame_idx} | reward_avg={reward_running:.2f} |')

        writer.close()


if __name__ == '__main__':
    with open('../configs/SoftActorCritic_e0.yaml', 'r') as data_file:
        kwargs = yaml.safe_load(data_file)
    pprint(kwargs)

    # create env
    env_mgr = EnvironmentManager()
    env_kwargs = copy.deepcopy(kwargs['env'])
    env_args = [kwargs['env']['lib_path'], kwargs['env']['speaker_fname']]
    env_id = env_kwargs.pop('env_id')
    env = env_mgr.make(env_id, *env_args, **env_kwargs)

    kwargs['train'].update(kwargs['env'])
    kwargs['train']['collect_data'] = kwargs['collect_data']
    kwargs['train']['log_mode'] = kwargs['log_mode']

    if kwargs["init_data_fname"] is not None:
        kwargs['train']['init_data_fname'] = kwargs['init_data_fname']

    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]

    kwargs['soft_q_network']['state_dim'] = state_dim
    kwargs['soft_q_network']['action_dim'] = action_dim

    kwargs['policy_network']['state_dim'] = state_dim
    kwargs['policy_network']['action_dim'] = action_dim

    if 'agent_fname' in kwargs:
        # load agent
        agent_fname = kwargs['agent_fname']
        print(f'Loading agent from "{agent_fname}"')
        agent = torch.load(kwargs['agent_fname'])
        if not kwargs['use_alpha']:
            agent.noise_level = kwargs['noise_init_level']
            agent.noise = OUNoise(kwargs['soft_q_network']['action_dim'] * kwargs['env']['num_workers'],
                                     kwargs['env']['seed'])
        # to enable agent starting with custom (full) replay buffer
        if agent.replay_buffer_csv_filename is not None:
            agent.replay_buffer_csv_filename = os.path.splitext(agent.replay_buffer_csv_filename)[0] + "_new.csv"
            agent.replay_buffer = ReplayBuffer(agent.replay_buffer_size, agent.replay_buffer_csv_filename, None)
    else:
        # create agent
        agent = AsyncrhonousSoftActorCritic(**kwargs)

    # train
    agent.train(env, **kwargs['train'])
