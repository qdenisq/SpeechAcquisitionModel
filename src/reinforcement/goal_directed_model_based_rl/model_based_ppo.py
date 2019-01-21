import json
from pprint import pprint
import os
import numpy as np
import pandas as pd
import datetime

import torch
from torch.nn import Linear, LSTM, Tanh, ReLU, Module, MSELoss
from torchvision import transforms

from src.speech_classification.audio_processing import AudioPreprocessor
from src.speech_classification.pytorch_conv_lstm import LstmNet
from src.reinforcement.goal_directed_model_based_rl.model import StochasticLstmModelDynamics, SimpleStochasticPolicy
from src.reinforcement.goal_directed_model_based_rl.replay_buffer import ReplayBuffer
from src.VTL.vtl_environment import VTLEnv


def train(*args, **kwargs):
    print(kwargs)

    torch.random.manual_seed(0)


    device = kwargs['train']['device']

    # 1. Init audio preprocessing
    preproc = AudioPreprocessor(**kwargs['preprocessing_params'])
    sr = kwargs['preprocessing_params']['sample_rate']

    # 2. Load preprocessing net
    preproc_net = torch.load(kwargs['preproc_net_fname']).to(device)

    # 3. Init model dynamics net
    md_net = StochasticLstmModelDynamics(**kwargs['model_dynamics_params']).to(device)
    optim = torch.optim.Adam(md_net.parameters(), lr=kwargs['train']['learning_rate'], eps=kwargs['train']['learning_rate_eps'])

    # 4. Init Policy
    policy = SimpleStochasticPolicy(**kwargs['policy_params']).to(device)

    # 5. Init environment
    speaker_fname = os.path.join(kwargs['vtl_dir'], 'JD2.speaker')
    lib_path = os.path.join(kwargs['vtl_dir'], 'VocalTractLab2.dll')
    num_episodes = 10
    ep_duration = 1000
    timestep = 20
    num_steps_per_ep = ep_duration // timestep

    env = VTLEnv(lib_path, speaker_fname, timestep, max_episode_duration=ep_duration)

    # 6. Load reference for policy
    reference_wav_fname = kwargs['reference_fname']
    reference_preproc = torch.from_numpy(preproc(reference_wav_fname)[np.newaxis]).float().to(device)
    _, _, reference = preproc_net(reference_preproc, seq_lens=np.array([reference_preproc.shape[1]]))
    reference = reference.detach().cpu().numpy().squeeze()

    # 7. Init replay buffer
    replay_buffer = ReplayBuffer(kwargs['buffer_size'])

    # 8. Train loop
    params = kwargs['train']
    policy.eval()
    md_net.train()
    for i in range(params['num_steps']):

        state = env.reset()
        goal_state = np.zeros(kwargs['model_dynamics_params']['goal_dim'])
        states = [state]
        actions = []
        goal_states = [goal_state]
        hidden = None
        for step in range(num_steps_per_ep):
            policy_input = np.concatenate((state, reference[step, :]))[np.newaxis]
            policy_input = torch.from_numpy(policy_input).float().to(device)
            action, _, _ = policy(policy_input)
            # action = (np.random.rand(action_space)) * 100.
            action = action.detach().cpu().numpy().squeeze()
            action[env.number_vocal_tract_parameters:] = 0.
            action = action * 0.1 # reduce amplitude for now
            new_state, audio = env.step(action, True)

            preproc_audio = preproc(audio, sr)[np.newaxis]
            preproc_audio = torch.from_numpy(preproc_audio).float().to(device)
            _, hidden, new_goal_state = preproc_net(preproc_audio, seq_lens=np.array([preproc_audio.shape[1]]), hidden=hidden)
            new_goal_state = new_goal_state.detach().cpu().numpy().squeeze()

            states.append(new_state)
            goal_states.append(new_goal_state)
            actions.append(action)

            state = new_state
            goal_state = new_goal_state

            env.render()

        replay_buffer.add(states, goal_states, actions, None, None)

        minibatch_size = kwargs['train']['minibatch_size']
        if replay_buffer.size() > minibatch_size:
            num_updates_per_epoch = kwargs['train']['updates_per_episode']
            for k in range(num_updates_per_epoch):
                # sample minibatch
                s0, g0, a, _, _ = replay_buffer.sample_batch(minibatch_size)

                # train
                seq_len = a.shape[1]
                goal_dim = kwargs['model_dynamics_params']["goal_dim"]

                s_bound = env.state_bound
                a_bound = env.action_bound

                s = torch.from_numpy(normalize(s0, s_bound)).float().to(device)
                g = torch.from_numpy(g0).float().to(device)
                a = torch.from_numpy(normalize(a, a_bound)).float().to(device)


                # forward prop
                s_pred, g_pred, s_prob, g_prob, state_dists, goal_dists = md_net(s[:, :-1, :], g[:, :-1, :], a)

                # compute error
                mse_loss = MSELoss(reduction='sum')(g_pred, g[:, 1:, :]) / (seq_len * kwargs['train']['minibatch_size'])

                loss = -goal_dists.log_prob(g[:, 1:, :]).sum(dim=-1, keepdim=True).mean()

                state_mse_loss = MSELoss(reduction='sum')(s_pred, s[:, 1:, :]) / (seq_len * kwargs['train']['minibatch_size'])
                state_loss = -state_dists.log_prob(s[:, 1:, :]).sum(dim=-1, keepdim=True).mean()
                total_loss = loss + state_loss
                # backprop
                optim.zero_grad()
                total_loss.backward()
                optim.step()

                dynamics = MSELoss(reduction='sum')(g[:, 1:, :], g[:, :-1, :]) / (seq_len * kwargs['train']['minibatch_size'])

            print("\rstep: {} | stochastic_loss: {:.4f} | loss: {:.4f}| actual_dynamics: {:.4f} |  state stochastic loss: {:.4f} | state_loss: {:.4f}".format(i, loss.detach().cpu().item(),
                                                                                                        mse_loss.detach().cpu().item(),
                                                                                                        dynamics.detach().cpu().item(),
                                                                                                        state_loss.detach().cpu().item(),
                  state_mse_loss.detach().cpu().item()),
                  end="")
            if step % 100 == 0:
                print()

    # 9. Save model
    dt = str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p"))
    md_fname = os.path.join(kwargs['save_dir'], '{}_{}.pt'.format("rnn_md", dt))
    torch.save(md_net, md_fname)


def normalize(data, bound):
    largest = np.array([max(abs(y[0]), abs(y[1])) for y in bound])
    normed_data = data / largest
    return normed_data


def denormalize(normed_data, bound):
    largest = np.array([max(abs(y[0]), abs(y[1])) for y in bound])
    data = normed_data * largest
    return data


import numpy as np
import torch


class ModelBasedPPO:
    def __init__(self, policy, model_dynamics, preproc, preproc_net, **kwargs):
        self.policy = policy
        self.model_dynamics = model_dynamics
        self.preproc = preproc
        self.preproc_net = preproc_net

        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=kwargs['policy_lr'], eps=kwargs['learning_rate_eps'])
        self.model_dynamics_optim = torch.optim.Adam(self.model_dynamics.parameters(), lr=kwargs['model_dynamics_lr'], eps=kwargs['learning_rate_eps'])

        self.num_epochs_actor = kwargs['num_epochs_actor']
        self.num_epochs_critic = kwargs['num_epochs_critic']
        self.discount = kwargs['discount']
        self.lmbda = kwargs['lambda']
        self.minibatch_size = kwargs['minibatch_size']
        self.epsilon = kwargs['epsilon']
        self.beta = kwargs['beta']
        self.clip_grad = kwargs['clip_grad']
        self.device = kwargs['device']
        self.episode_length = kwargs['episode_length'] // kwargs['timestep']
        self.sr = kwargs['preprocessing_params']['sample_rate']

    def rollout(self, env, reference):
        """
           Runs an agent in the environment and collects trajectory
           :param env: Environment to run the agent in (ReacherEnvironment)
           :return states: (torch.Tensor)
           :return actions: (torch.Tensor)
           :return rewards: (torch.Tensor)
           :return dones: (torch.Tensor)
           :return values: (torch.Tensor)
           :return old_log_probs: (torch.Tensor)
           """
        state = env.reset()
        # Experiences
        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        old_log_probs = []

        self.policy.eval()

        state = env.reset()
        goal_state = np.zeros(kwargs['model_dynamics_params']['goal_dim'])
        states = [state]
        actions = []
        goal_states = [goal_state]
        hidden = None
        for step in range(self.episode_length):
            policy_input = np.concatenate((state, reference[step, :]))[np.newaxis]
            policy_input = torch.from_numpy(policy_input).float().to(self.device)
            action, _, _ = self.policy(policy_input)
            # action = (np.random.rand(action_space)) * 100.
            action = action.detach().cpu().numpy().squeeze()
            action[env.number_vocal_tract_parameters:] = 0.
            action = action  # reduce amplitude for now
            new_state, audio = env.step(action, True)

            preproc_audio = self.preproc(audio, self.sr)[np.newaxis]
            preproc_audio = torch.from_numpy(preproc_audio).float().to(self.device)
            _, hidden, new_goal_state = self.preproc_net(preproc_audio, seq_lens=np.array([preproc_audio.shape[1]]),
                                                    hidden=hidden)
            new_goal_state = new_goal_state.detach().cpu().numpy().squeeze()

            states.append(new_state)
            goal_states.append(new_goal_state)
            actions.append(action)

            state = new_state
            goal_state = new_goal_state

            env.render()



        # Rollout
        while True:
            action, old_log_prob = self.policy(torch.from_numpy(state).float().to(self.device))
            action = np.clip(action.detach().cpu().numpy(), -1., 1.)
            _, old_log_prob, _, _ = self.agent(torch.from_numpy(state).float().to(self.device), torch.from_numpy(action).float().to(self.device))

            next_state, reward, done = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            values.append(value.detach().cpu().numpy())
            old_log_probs.append(old_log_prob.detach().cpu().numpy())

            state = next_state

            if np.any(done):
                break

        states = torch.from_numpy(np.asarray(states)).float().to(self.device)
        actions = torch.from_numpy(np.asarray(actions)).float().to(self.device)
        rewards = torch.from_numpy(np.asarray(rewards)).float().to(self.device)
        dones = torch.from_numpy(np.asarray(dones).astype(int)).long().to(self.device)
        values = torch.from_numpy(np.asarray(values)).float().to(self.device)
        values = torch.cat([values, torch.zeros(1, values.shape[1], 1).to(self.device)], dim=0)
        old_log_probs = torch.from_numpy(np.asarray(old_log_probs)).float().to(self.device)

        return states, actions, rewards, dones, values, old_log_probs

    def train(self, env, num_episodes):
        """
        Train the agent to solve environment
        :param env: environment object (ReacherEnvironment)
        :param num_episodes: number of episodes (int)
        :return scores: list of scores for each episode (list)
        """

        scores = []
        for episode in range(num_episodes):
            states, actions, rewards, dones, values, old_log_probs = self.rollout(env)

            score = rewards.sum(dim=0).mean()

            T = rewards.shape[0]
            last_advantage = torch.zeros((rewards.shape[1], 1))
            last_return = torch.zeros(rewards.shape[1])
            returns = torch.zeros(rewards.shape)
            advantages = torch.zeros(rewards.shape)

            # calculate return and advantage
            for t in reversed(range(T)):
                # calc return
                last_return = rewards[t] + last_return * self.discount * (1 - dones[t]).float()
                returns[t] = last_return

            # Update
            returns = returns.view(-1, 1)
            states = states.view(-1, env.get_state_dim())
            actions = actions.view(-1, env.get_action_dim())
            old_log_probs = old_log_probs.view(-1, 1)

            # update critic
            num_updates = actions.shape[0] // self.minibatch_size
            self.agent.train()
            for k in range(self.num_epochs_critic):
                for _ in range(num_updates):
                    idx = np.random.randint(0, actions.shape[0], self.minibatch_size)
                    returns_batch = returns[idx]
                    states_batch = states[idx]

                    _, _, _, values_pred = self.agent(states_batch)

                    critic_loss = torch.nn.MSELoss()(values_pred, returns_batch)

                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.agent.get_critic_parameters(), self.clip_grad)
                    self.critic_optim.step()

            # calc advantages
            self.agent.eval()
            for t in reversed(range(T)):
                # advantage
                next_val = self.discount * values[t + 1] * (1 - dones[t]).float()[:, np.newaxis]
                delta = rewards[t][:, np.newaxis] + next_val - values[t]
                last_advantage = delta + self.discount * self.lmbda * last_advantage
                advantages[t] = last_advantage.squeeze()

            advantages = advantages.view(-1, 1)
            advantages = (advantages - advantages.mean()) / advantages.std()

            # update actor
            self.agent.train()
            for k in range(self.num_epochs_actor):
                for _ in range(num_updates):
                    idx = np.random.randint(0, actions.shape[0], self.minibatch_size)
                    advantages_batch = advantages[idx]
                    old_log_probs_batch = old_log_probs[idx]
                    states_batch = states[idx]
                    actions_batch = actions[idx]

                    _, new_log_probs, entropy, _ = self.agent(states_batch, actions_batch)

                    ratio = (new_log_probs.view(-1, 1) - old_log_probs_batch).exp()
                    obj = ratio * advantages_batch
                    obj_clipped = ratio.clamp(1.0 - self.epsilon,
                                              1.0 + self.epsilon) * advantages_batch
                    entropy_loss = entropy.mean()

                    policy_loss = -torch.min(obj, obj_clipped).mean(0) - self.beta * entropy_loss

                    self.actor_optim.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.agent.get_actor_parameters(), self.clip_grad)
                    self.actor_optim.step()

            scores.append(score)
            print("episode: {} | score:{:.4f} | action_mean: {:.2f}, action_std: {:.2f}".format(
                episode, score, actions.mean().cpu(), actions.std().cpu()))

        print("Training finished. Result score: ", score)
        return scores



if __name__ == '__main__':
    with open('rnn_md_config.json') as data_file:
        data = json.load(data_file)

    pprint(data)
    kwargs = data

    train(**kwargs)