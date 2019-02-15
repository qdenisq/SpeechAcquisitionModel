import json
from pprint import pprint
import os
import numpy as np
import pandas as pd
import datetime

import torch
from src.reinforcement.goal_directed_model_based_rl.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt


# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.003, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class RNNModelBased1StepBackProp:
    def __init__(self, agent=None, model_dynamics=None, **kwargs):
        self.agent = agent

        self.actor_optim = torch.optim.Adam(agent.parameters(), lr=kwargs['actor_lr'],
                                            eps=kwargs['learning_rate_eps'])

        self.num_epochs_actor = kwargs['num_epochs_actor']
        self.num_epochs_critic = kwargs['num_epochs_critic']
        self.minibatch_size = kwargs['minibatch_size']
        self.clip_grad = kwargs['clip_grad']
        self.device = kwargs['device']
        self.num_rollouts_per_update = kwargs['rollouts_per_update']

        self.model_dynamics = model_dynamics
        self.model_dynamics_optim = torch.optim.Adam(self.model_dynamics.parameters(), lr=kwargs['model_dynamics_lr'], eps=kwargs['learning_rate_eps'])
        self.num_epochs_model_dynamics = kwargs['num_epochs_model_dynamics']
        self.num_virtual_rollouts_per_update = kwargs['virtual_rollouts_per_update']

        self.replay_buffer = ReplayBuffer(kwargs['buffer_size'])
        self.buffer_trajectories = kwargs['buffer_trajectories']

    def train(self, env, num_episodes):
        """
        Train the agent to solve environment
        :param env: environment object (ReacherEnvironment)
        :param num_episodes: number of episodes (int)
        :return scores: list of scores for each episode (list)
        """

        action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(env.action_dim), sigma=0.001)
        scores = []
        train_step_i = 0
        for episode in range(num_episodes):
            # rollout
            T = len(env.get_current_reference())
            reference = env.get_current_reference()[:, 4:]
            plt.imshow(reference.T)
            plt.show()
            state = env.reset()
            state = env.normalize(state, env.state_bound)
            score = 0.

            self.agent.eval()
            while True:
                state_tensor = torch.from_numpy(state).float().to(self.device).view(1, -1)
                action = self.agent(state_tensor).detach().cpu().numpy().squeeze()
                action = action + action_noise()
                action_denorm = env.denormalize(action, env.action_bound)
                next_state, reward, done, _ = env.step(action_denorm)
                next_state = env.normalize(next_state, env.state_bound)
                env.render()

                # if i % save_step != 0:
                self.replay_buffer.add((state, action, next_state))

                score += reward
                miss = torch.abs(torch.from_numpy(next_state).float().to(self.device)[:-env.goal_dim][torch.from_numpy(np.array(env.state_goal_mask, dtype=np.uint8)).byte()] -
                                 torch.from_numpy(state).float().to(self.device)[-env.goal_dim:])
                if miss.max() > 0.1 and env.current_step > 3:
                    break

                if np.any(done):
                    break
                state = next_state
            scores.append(score)

            if self.replay_buffer.size() > self.minibatch_size:
                # train nets couple of times relative to the increase of replay buffer
                n_train_steps = round(
                    self.replay_buffer.size() / self.replay_buffer.buffer_size * self.num_epochs_model_dynamics + 1)
                ##############################################################
                # train model dynamics
                ##############################################################
                self.model_dynamics.train()
                for _ in range(n_train_steps):
                    train_step_i += 1
                    s0_batch, a_batch, s1_batch = self.replay_buffer.sample_batch(self.minibatch_size)

                    self.model_dynamics_optim.zero_grad()
                    s1_pred, _ = self.model_dynamics(s0_batch.float().to(self.device), a_batch.float().to(self.device))
                    md_loss = torch.nn.MSELoss()(s1_pred, s1_batch[:, :-env.goal_dim].float().to(self.device))
                    md_loss.backward()
                    self.model_dynamics_optim.step()

                ##############################################################
                # train policy
                ##############################################################
                self.agent.train()
                self.model_dynamics.eval()
                for _ in range(n_train_steps):
                    # train_step_i += 1
                    s0_batch, a_batch, s1_batch = self.replay_buffer.sample_batch(self.minibatch_size)

                    self.actor_optim.zero_grad()
                    actions_predicted = self.agent(s0_batch.float().to(self.device))
                    # predict state if predicted actions will be applied
                    s1_pred, _ = self.model_dynamics(s0_batch.float().to(self.device), actions_predicted)
                    actor_loss = torch.nn.MSELoss()(
                        s1_pred[:, torch.from_numpy(np.array(env.state_goal_mask, dtype=np.uint8)).byte()],
                        s0_batch[:, -env.goal_dim:].float().to(self.device))
                    actor_loss.backward()
                    self.actor_optim.step()

                    # expected_actions_normed = normalize(denormalize(target_batch_normed - g0_batch_normed, g_bound),
                    #                                     a_bound)
                    # policy_loss_out = np.mean(
                    #     np.sum(np.square(expected_actions_normed - actions_normed.detach().numpy()), axis=1), axis=0)

                print("|episode: {}| train step: {}| model_dynamics loss: {:.8f}| policy loss: {:.5f}| score:{:.2f} |".format(episode,
                                                                                                                              train_step_i,
                                                                                                                              md_loss.detach().numpy(),
                                                                                                                              actor_loss.detach().numpy(),
                                                                                                                              score))

        print("Training finished. Result score: ", score)
        return scores