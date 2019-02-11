import json
from pprint import pprint
import os
import numpy as np
import pandas as pd
import datetime

import torch
from src.reinforcement.goal_directed_model_based_rl.replay_buffer import ReplayBuffer


class ModelBased1StepBackProp:
    def __init__(self, agent=None, model_dynamics=None, **kwargs):
        self.agent = agent

        self.actor_optim = torch.optim.Adam(agent.get_actor_parameters(), lr=kwargs['actor_lr'],
                                            eps=kwargs['learning_rate_eps'])
        self.critic_optim = torch.optim.Adam(agent.get_critic_parameters(), lr=kwargs['critic_lr'],
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

    def train(self, env, num_episodes):
        """
        Train the agent to solve environment
        :param env: environment object (ReacherEnvironment)
        :param num_episodes: number of episodes (int)
        :return scores: list of scores for each episode (list)
        """

        scores = []
        train_step_i = 0
        for episode in range(num_episodes):
            # rollout
            T = len(env.get_current_reference())
            state = env.reset()
            state = env.normalize(state, env.state_bound)
            score = 0.

            self.agent.eval()
            while True:
                state_tensor = torch.from_numpy(state).float().to(self.device).view(1, -1)
                action, _, _, _ = self.agent(state_tensor)
                action_denorm = env.denormalize(action.detach().cpu().numpy().squeeze(), env.action_bound)
                next_state, reward, done, _ = env.step(action_denorm)
                next_state = env.normalize(next_state, env.state_bound)
                env.render()

                # if i % save_step != 0:
                self.replay_buffer.add((state, action.detach().cpu().numpy().squeeze(), next_state))

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
                    s1_pred, _, _, _, _, _ = self.model_dynamics(s0_batch.float().to(self.device), a_batch.float().to(self.device))
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
                    actions_predicted, _, _, _ = self.agent(s0_batch.float().to(self.device))
                    # predict state if predicted actions will be applied
                    s1_pred, _, _, _, _, _ = self.model_dynamics(s0_batch.float().to(self.device), actions_predicted)
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