import json
from pprint import pprint
import os
import numpy as np
import pandas as pd
import datetime

import torch
from src.reinforcement.goal_directed_model_based_rl.replay_buffer import ReplayBuffer


class ModelBasedBackProp:
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

    def rollout(self, env, num_rollouts):
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
        states_out = []
        actions_out = []
        dones_out = []
        next_states_out = []
        for i in range(num_rollouts):
            state = env.reset()
            # Experiences
            states = []
            actions = []
            dones = []
            next_states = []

            self.agent.eval()
            # Rollout
            while True:
                state = env.normalize(state, env.state_bound)
                action, old_log_prob, _, value = self.agent(torch.from_numpy(state).float().to(self.device).view(1, -1))
                action = np.clip(action.detach().cpu().numpy(), -1., 1.)
                _, old_log_prob, _, _ = self.agent(torch.from_numpy(state).float().to(self.device).view(1, -1), torch.from_numpy(action).float().to(self.device))

                action_denorm = env.denormalize(action.squeeze(), env.action_bound)
                next_state, reward, done, _ = env.step(action_denorm)

                states.append(state)
                actions.append(action.squeeze())
                dones.append(done)
                next_states.append(env.normalize(next_state, env.state_bound))

                state = next_state
                env.render()
                if np.any(done):
                    break

            states_out.append(states)
            actions_out.append(actions)
            dones_out.append(dones)
            next_states_out.append(next_states)

        states_out = torch.from_numpy(np.asarray(states_out)).float().to(self.device)
        actions_out = torch.from_numpy(np.asarray(actions_out)).float().to(self.device)
        dones_out = torch.from_numpy(np.asarray(dones_out).astype(int)).long().to(self.device)
        next_states_out = torch.from_numpy(np.asarray(next_states_out)).float().to(self.device)

        return states_out, actions_out, dones_out, next_states_out

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
        states = []
        actions = []
        dones = []
        next_states = []
        goals = []

        state = env.reset()
        state = env.normalize(state, env.state_bound)
        state = torch.from_numpy(state).float().to(self.device).view(1, -1).repeat(num_rollouts, 1)
        reference = env.get_current_reference()

        cur_step = 0
        episode_length = len(reference)
        # Rollout
        while True:
            action, _, _, _ = self.agent(state)
            next_state, _, _, _, _, _ = self.model_dynamics(state, action)

            goal = env.normalize(reference[cur_step], env.goal_bound)
            goal = torch.from_numpy(goal).float().to(self.device).view(1, -1).repeat(num_rollouts, 1)
            cur_step += 1
            done = cur_step >= episode_length
            next_state = torch.cat((next_state, goal), dim=-1)

            states.append(state)
            actions.append(action)
            dones.append(done)
            next_states.append(next_state)
            goals.append(goal)

            state = next_state
            if np.any(done):
                break

        states = torch.cat(states)
        actions = torch.cat(actions)
        next_states = torch.cat(next_states)
        goals = torch.cat(goals)

        return states, actions, dones, next_states

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
            states, actions, dones, next_states = self.rollout(env, self.num_rollouts_per_update)

            score = np.mean(np.sum(np.exp(-10.*np.sum((next_states.detach().numpy()[:, :, :-env.goal_dim][:, :, env.state_goal_mask]
                                               - states.detach().numpy()[:, :, -env.goal_dim:])**2, axis=-1)), axis=-1))


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
            self.agent.train()

            states, actions, dones, next_states = self.virtual_rollout(env, self.num_virtual_rollouts_per_update)

            T = states.shape[0]
            # update actor
            for k in range(self.num_epochs_actor):
                for _ in range(num_updates):
                    idx = np.random.randint(0, actions.shape[0], self.minibatch_size)
                    states_batch = states[idx]
                    actions_batch = actions[idx]
                    next_states_batch = next_states[idx]

                    policy_loss = torch.nn.MSELoss()(
                        next_states_batch[:, :-env.goal_dim][:, torch.from_numpy(np.array(env.state_goal_mask, dtype=np.uint8)).byte()],
                        states_batch[:, -env.goal_dim:])

                    self.actor_optim.zero_grad()
                    policy_loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.agent.get_actor_parameters(), self.clip_grad)
                    self.actor_optim.step()

            scores.append(score)
            print("episode: {} | md_loss:{:.4f}  | score:{:.4f} | md_loss:{:.4f} | policy_loss:{:.4f} |".format(
                episode, model_dynamics_loss.detach().cpu(), score,  model_dynamics_loss.detach().cpu(), policy_loss.detach().cpu()))

        print("Training finished. Result score: ", score)
        return scores



if __name__ == '__main__':
    with open('rnn_md_config.json') as data_file:
        data = json.load(data_file)

    pprint(data)
    kwargs = data

    train(**kwargs)