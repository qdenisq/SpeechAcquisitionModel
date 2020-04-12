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
from src.reinforcement_v2.common.nn import SoftQNetwork, PolicyNetwork, ModelDynamics, DeterministicPolicyNetwork

from src.reinforcement_v2.common.tensorboard import DoubleSummaryWriter
from src.reinforcement_v2.common.noise import OUNoise
from src.reinforcement_v2.utils.timer import Timer

from src.siamese_net_sound_similarity.train_v2 import SiameseDeepLSTMNet


def soft_update(from_net, target_net, soft_tau=0.1):
    for target_param, param in zip(target_net.parameters(), from_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )

class BackpropIntoPolicy(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BackpropIntoPolicy, self).__init__()

        self._dt = str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p"))
        self._env_name = kwargs['env']['env_id']

        self.params = kwargs
        self.use_cuda = kwargs.get('use_cuda', True)
        self.device = torch.device("cuda" if self.use_cuda and torch.cuda.is_available() else "cpu")

        self.agent_state_dim = kwargs['agent_state_dim']
        self.state_dim = kwargs['state_dim']
        self.action_dim = kwargs['action_dim']
        self.reference_mask = kwargs['reference_mask']
        self.acoustic_dim = kwargs['model_dynamics']['acoustic_dim']

        self.model_dynamics = ModelDynamics(**kwargs['model_dynamics']).to(self.device)
        self.model_dynamics_target = copy.deepcopy(self.model_dynamics)

        if kwargs["pretrained_policy"] is not None:
            self.policy_net = torch.load(kwargs['pretrained_policy']).policy_net
            self.policy_net.to(self.device)
        else:
            self.policy_net = DeterministicPolicyNetwork(**kwargs['policy_network']).to(self.device)

        self.use_alpha = kwargs.get('use_alpha', True)
        if not self.use_alpha:
            self.noise = OUNoise(kwargs['policy_network']['action_dim'] * kwargs['env']['num_workers'],
                                 kwargs['env']['seed'])
            self.noise_decay = kwargs['noise_decay']  # decay after each 1000 steps
            self.noise_level = 1.0
            self.noise_min = kwargs['noise_min']

        self.lr = kwargs['lr']

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.md_optimizer = optim.Adam(self.model_dynamics.parameters(), lr=self.lr)

        # init replay buffer
        self.replay_buffer_size = kwargs['replay_buffer']['size']

        init_data_fname = kwargs['init_data_fname']

        self.replay_buffer_csv_filename = None
        if kwargs['collect_data']:
            self.replay_buffer_csv_filename = f'../../../data/{self._env_name}_sac_data_{self._dt}.csv'

        self.replay_buffer = ReplayBuffer(self.replay_buffer_size, self.replay_buffer_csv_filename, init_data_fname)

        self.gamma = kwargs.get('gamma', 0.99)
        self.num_updates_per_step = kwargs.get("num_updates_per_step", 30)
        self.train_step = 0
        self.frame_idx = 0

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
        Model Dynamics Update
        """

        agent_state = state[:, :self.agent_state_dim]
        agent_next_state = next_state[:, :self.agent_state_dim]
        predicted_next_agent_state = self.model_dynamics(agent_state, action)

        self.md_optimizer.zero_grad()
        md_loss = torch.nn.SmoothL1Loss(reduction="sum")(predicted_next_agent_state, agent_next_state) / batch_size
        md_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model_dynamics.parameters(), 1)
        self.md_optimizer.step()
        if self.train_step % 5 == 0:
            self.soft_update(self.model_dynamics, self.model_dynamics_target, 0.1)

        """
        Policy loss
        """

        new_action = self.policy_net(state)
        predicted_next_agent_state = self.model_dynamics(agent_state, new_action)
        predicted_next_agent_state_masked = predicted_next_agent_state[:, self.reference_mask]
        state_masked = state[:, self.agent_state_dim:]

        predicted_next_agent_state_masked = predicted_next_agent_state[:, self.reference_mask][:, : -self.acoustic_dim]
        state_masked = state[:, self.agent_state_dim:][:, : -self.acoustic_dim]

        policy_loss = torch.nn.SmoothL1Loss(reduction="sum")(predicted_next_agent_state_masked, state_masked) / batch_size

        self.md_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.policy_optimizer.step()

        self.train_step += 1
        return self.train_step, policy_loss.detach().cpu(), md_loss.detach().cpu()

    def soft_update(self, from_net, target_net, soft_tau):
        for target_param, param in zip(target_net.parameters(), from_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

    def add_weights_histogram(self, writer, global_step):
        for name, param in self.policy_net.named_parameters():
            writer.add_histogram("Policynet_" + name, param, global_step)

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
        self.policy_net.to(self.device)
        self.model_dynamics.to(self.device)

    def train(self, env, **kwargs):
        """
        Initialize utility objects (e.g. summary writers, timers)
        """

        timer = defaultdict(Timer)
        writer = DoubleSummaryWriter(log_dir=f'../../../runs/{self._env_name}_backprop_{self._dt}/',
                                     light_log_dir=f'../../../runs_light/light_{self._env_name}_backprop_{self._dt}/',
                                     mode=kwargs['log_mode'])



        # TODO: only last called graph is added
        dummy_input = torch.rand(1, self.state_dim).to(self.device)
        writer.add_graph(self.policy_net, dummy_input)

        max_frames = kwargs.get('max_frames', 40000)
        max_steps = kwargs.get('max_steps', 100)
        batch_size = kwargs.get('batch_size', 256)

        self.params['gamma'] = self.gamma
        self.params['use_alpha'] = self.use_alpha
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
                action = self.policy_net.get_action(state)
                action = action.detach().cpu().numpy()
                if not self.use_alpha:
                    action = action + self.noise.sample().reshape(*action.shape) * self.noise_level
                    # just for testing
                    # action *= 0
                    action = np.clip(action, -1, 1)

                timer['algo'].stop()
                timer['env'].start()

                next_state, reward, done, info = env.step(action)
                # dtw_dist = info['dtw_dist']
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

                # if len(self.replay_buffer) > 3 * batch_size and self.frame_idx % 25 == 0:
                if len(self.replay_buffer) > 3 * batch_size:

                    timer['algo'].start()
                    for _ in range(self.num_updates_per_step):
                        _, policy_loss, md_loss = self.update(batch_size)
                    timer['algo'].stop()

                    timer['utils'].start()
                    if not self.use_alpha:
                        writer.add_scalar('Noise_level', self.noise_level, self.frame_idx)
                    writer.add_scalar('Policy_loss', policy_loss, self.frame_idx)
                    writer.add_scalar('Model Dynamics Loss', md_loss, self.frame_idx)
                    print(f"policy_loss: {policy_loss:.2f} | md_loss: {md_loss:.2f}")
                    # explicitly remove all tensor-related variables just to ensure well-behaved memory management
                    del policy_loss, md_loss, action
                    timer['utils'].stop()


                # ADD termination condition:
                current_steps = env.get_attr('current_step')
                print(current_steps)
                for w in range(kwargs['num_workers']):
                    # if reward[w] < 0.001 and current_steps[w] > 3: # dtw > 7
                    #     done[w] = True

                    if np.mean(abs(info[w]['dtw_dist'])) > 5.0 and current_steps[w] > 3: # dtw > 7
                        # continue
                        done[w] = True

                if any(done):

                    timer['utils'].start()
                    for i in range(kwargs['num_workers']):
                        if done[i]:
                            reward_running = 0.90 * reward_running + 0.10 * ep_reward[i]
                            rewards.append(ep_reward[i])
                            ep_reward[i] = 0.
                            state[i] = env.reset([i])
                    env.render()

                    writer.add_scalar("Reward", reward_running, self.frame_idx)
                    for k, v in timer.items():
                        writer.add_scalar(f"Elapsed time: {k}", v.elapsed_time, self.frame_idx)
                    self.add_weights_histogram(writer, self.frame_idx)
                    timer['utils'].stop()
                    # break

            # save best performing policy
            if reward_running > best_total_reward or self.frame_idx % 200 == 0:
                timer['utils'].start()
                # self.save_networks(episode_reward)
                name = f'{self._env_name}_BackpropIntoPolicy_' + f'{reward_running:.2f}'
                save_dir = f'../../../models/{self._env_name}_backprop_{self._dt}/'
                try:
                    os.makedirs(save_dir)
                except:
                    pass
                path_agent = f'{save_dir}{name}.bp'
                torch.save(self, path_agent)
                timer['utils'].stop()
                print(f'step={self.frame_idx} | reward_avg={reward_running:.2f} | saving agent: {name}')
                if reward_running > best_total_reward:
                    best_total_reward = reward_running

            if self.frame_idx % 100 == 0:
                print(f'step={self.frame_idx} | reward_avg={reward_running:.2f} |')

        writer.close()


if __name__ == '__main__':
    with open('../configs/BackpropIntoPolicy_e0.yaml', 'r') as data_file:
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
    agent_state_dim = env.get_attr('agent_state_dim')[0]
    reference_mask = env.get_attr('reference_mask')[0]
    acoustics_dim = env.get_attr('audio_dim')[0]

    kwargs['action_dim'] = action_dim
    kwargs['agent_state_dim'] = agent_state_dim
    kwargs['state_dim'] = state_dim
    kwargs['reference_mask'] = reference_mask

    kwargs['model_dynamics']['agent_state_dim'] = agent_state_dim
    kwargs['model_dynamics']['action_dim'] = action_dim
    kwargs['model_dynamics']['acoustic_dim'] = acoustics_dim

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
        agent = BackpropIntoPolicy(**kwargs)

    # train

    run_dir = f'../../../runs/{agent._env_name}_backprop_{agent._dt}/'
    try:
        os.makedirs(run_dir)
    except:
        pass
    with open(os.path.join(run_dir, 'md_backprop.yaml'), 'w') as f:
        yaml.dump(kwargs, f)


    agent.train(env, **kwargs['train'])
