import json
from pprint import pprint
import os
import numpy as np
import pandas as pd
import datetime

import torch
from src.reinforcement.goal_directed_model_based_rl.algs.ppo import PPO


class ModelBasedPPO(PPO):
    def __init__(self, agent=None, model_dynamics=None, **kwargs):
        super(ModelBasedPPO, self).__init__(agent=agent, **kwargs)
        self.model_dynamics = model_dynamics
        self.model_dynamics_optim = torch.optim.Adam(self.model_dynamics.parameters(), lr=kwargs['model_dynamics_lr'], eps=kwargs['learning_rate_eps'])
        self.num_epochs_model_dynamics = kwargs['num_epochs_model_dynamics']
        self.num_virtual_rollouts_per_update = kwargs['virtual_rollouts_per_update']

    def virtual_rollout(self, env, num_rollouts):
        """
          Runs an agent in the "virtual environment" using trained model dynamics and collects trajectory
          :return states: (torch.Tensor)
          :return actions: (torch.Tensor)
          :return rewards: (torch.Tensor)
          :return dones: (torch.Tensor)
          :return values: (torch.Tensor)
          :return old_log_probs: (torch.Tensor)
          """
        states_out = []
        actions_out = []
        rewards_out = []
        dones_out = []
        values_out = []
        old_log_probs_out = []
        next_states_out = []
        for i in range(num_rollouts):
            state = env.reset()
            state = env.normalize(state, env.state_bound)
            reference = env.get_current_reference()

            cur_step = 0
            episode_length = len(reference)
            # Experiences
            states = []
            actions = []
            rewards = []
            dones = []
            values = []
            old_log_probs = []
            next_states = []

            self.agent.eval()
            self.model_dynamics.eval()
            # Rollout
            while True:
                action, old_log_prob, _, value = self.agent(torch.from_numpy(state).float().to(self.device).view(1, -1))
                action = np.clip(action.detach().cpu().numpy(), -1., 1.)
                _, old_log_prob, _, _ = self.agent(torch.from_numpy(state).float().to(self.device).view(1, -1),
                                                   torch.from_numpy(action).float().to(self.device))

                action_denorm = env.denormalize(action.squeeze(), env.action_bound)
                next_state, goals, state_log_prob, goal_log_prob, state_dists, goal_dists = self.model_dynamics(torch.from_numpy(state).float().to(self.device).view(1, -1),
                                                 torch.from_numpy(action).float().to(self.device))

                next_state = next_state.detach().cpu().numpy().squeeze()
                goal = env.normalize(reference[cur_step], env.goal_bound)
                reward = env._reward(next_state, goal, action)
                cur_step += 1
                done = cur_step >= episode_length
                next_state = np.concatenate((next_state, goal))

                states.append(state)
                actions.append(action.squeeze())
                rewards.append(reward)
                dones.append(done)
                values.append(value.detach().cpu().numpy().squeeze())
                old_log_probs.append(old_log_prob.detach().cpu().numpy())
                next_states.append(next_state)

                state = next_state
                if np.any(done):
                    break

            states_out.append(states)
            actions_out.append(actions)
            rewards_out.append(rewards)
            dones_out.append(dones)
            values_out.append(values)
            old_log_probs_out.append(old_log_probs)
            next_states_out.append(next_states)

        states_out = torch.from_numpy(np.asarray(states_out)).float().to(self.device)
        actions_out = torch.from_numpy(np.asarray(actions_out)).float().to(self.device)
        rewards_out = torch.from_numpy(np.asarray(rewards_out)).float().to(self.device)
        dones_out = torch.from_numpy(np.asarray(dones_out).astype(int)).long().to(self.device)
        values_out = torch.from_numpy(np.asarray(values_out)).float().to(self.device).unsqueeze(2)
        values_out = torch.cat([values_out, torch.zeros(1, values_out.shape[1], 1).to(self.device)], dim=0)
        old_log_probs_out = torch.from_numpy(np.asarray(old_log_probs_out)).float().to(self.device)
        next_states_out = torch.from_numpy(np.asarray(next_states_out)).float().to(self.device)

        return states_out, actions_out, rewards_out, dones_out, values_out, old_log_probs_out, next_states_out

    def train(self, env, num_episodes):
        """
        Train the agent to solve environment
        :param env: environment object (ReacherEnvironment)
        :param num_episodes: number of episodes (int)
        :return scores: list of scores for each episode (list)
        """

        scores = []
        for episode in range(num_episodes):
            # rollout
            states, actions, rewards, dones, values, old_log_probs, next_states = self.rollout(env, self.num_rollouts_per_update)
            score = rewards.sum(dim=1).mean()
            states = states.view(-1, states.shape[-1])
            actions = actions.view(-1, actions.shape[-1])
            next_states = next_states.view(-1, next_states.shape[-1])

            # update model dynamics
            num_updates = actions.shape[0] // self.minibatch_size
            self.model_dynamics.train()
            for k in range(self.num_epochs_model_dynamics):
                for _ in range(num_updates):
                    idx = np.random.randint(0, actions.shape[0], self.minibatch_size)

                    states_batch = states[idx]
                    actions_batch = actions[idx]
                    next_states_batch = next_states[idx]

                    next_state_pred, _, _, _, _, _ = self.model_dynamics(states_batch, actions_batch)

                    next_states_batch = next_states_batch[:, :next_state_pred.shape[-1]]
                    model_dynamics_loss = torch.nn.MSELoss()(next_state_pred, next_states_batch)

                    self.model_dynamics_optim.zero_grad()
                    model_dynamics_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model_dynamics.parameters(), self.clip_grad)
                    self.model_dynamics_optim.step()


            # do virtual rollouts
            # states, actions, rewards, dones, values, old_log_probs, _ = self.virtual_rollout(env, self.num_virtual_rollouts_per_update)
            score_pred = rewards.sum(dim=1).mean()

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
            states = states.view(-1, env.state_dim)
            actions = actions.view(-1, env.action_dim)
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
            print("episode: {} | md_loss:{:.4f}  | score:{:.4f} | score_pred:{:.4f}| action_mean: {:.2f}, action_std: {:.2f}".format(
                episode, model_dynamics_loss.detach().cpu(), score, score_pred, actions.mean().cpu(), actions.std().cpu()))

        print("Training finished. Result score: ", score)
        return scores



if __name__ == '__main__':
    with open('rnn_md_config.json') as data_file:
        data = json.load(data_file)

    pprint(data)
    kwargs = data

    train(**kwargs)