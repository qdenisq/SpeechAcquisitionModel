import numpy as np
import torch
import sys
import datetime
import os

class Logger(object):
    def __init__(self, fname):
        self.terminal = sys.stdout
        self.log_name = fname
        self.log = open(fname, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        self.log.close()
        self.log = open(self.log_name, "a")
        pass


class PPO:
    def __init__(self, agent=None, **kwargs):
        self.agent = agent

        self.actor_optim = torch.optim.Adam(agent.get_actor_parameters(), lr=kwargs['actor_lr'], eps=kwargs['learning_rate_eps'])
        self.critic_optim = torch.optim.Adam(agent.get_critic_parameters(), lr=kwargs['critic_lr'], eps=kwargs['learning_rate_eps'])

        self.num_epochs_actor = kwargs['num_epochs_actor']
        self.num_epochs_critic = kwargs['num_epochs_critic']
        self.discount = kwargs['discount']
        self.lmbda = kwargs['lambda']
        self.minibatch_size = kwargs['minibatch_size']
        self.epsilon = kwargs['epsilon']
        self.beta = kwargs['beta']
        self.clip_grad = kwargs['clip_grad']
        self.device = kwargs['device']
        self.num_rollouts_per_update = kwargs['rollouts_per_update']
        self.videos_dir = kwargs['videos_dir']
        self.train_step = 0


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
        rewards_out = []
        dones_out = []
        values_out = []
        old_log_probs_out = []
        next_states_out = []
        for i in range(num_rollouts):
            state = env.reset()
            # Experiences
            states = []
            actions = []
            rewards = []
            dones = []
            values = []
            old_log_probs = []
            next_states = []

            self.agent.eval()

            step = 0
            res_step = 0
            miss_max_idx = -1
            terminated = False
            misses = []
            # Rollout
            while True:
                state = env.normalize(state, env.state_bound)
                action, old_log_prob, _, value = self.agent(torch.from_numpy(state).float().to(self.device).view(1, -1))
                action = np.clip(action.detach().cpu().numpy(), -1., 1.)
                _, old_log_prob, _, _ = self.agent(torch.from_numpy(state).float().to(self.device).view(1, -1), torch.from_numpy(action).float().to(self.device))

                action_denorm = env.denormalize(action.squeeze(), env.action_bound)
                next_state, reward, done, _ = env.step(action_denorm)

                miss = torch.abs(torch.from_numpy(next_state).float().to(self.device)[:-env.goal_dim][
                                     torch.from_numpy(np.array(env.state_goal_mask, dtype=np.uint8)).byte()] -
                                 torch.from_numpy(state).float().to(self.device)[-env.goal_dim:])

                misses.append(miss[:].max().detach().cpu().numpy())
                if len(misses) > 3 and np.all(np.array(misses[-3:]) > 0.1) and not terminated:
                    terminated = True
                    res_step = step
                    miss_max_idx = np.argmax(miss[:].detach().cpu().numpy())


                states.append(state)
                actions.append(action.squeeze())
                if not terminated:
                    rewards.append(reward)
                    dones.append(done)
                else:
                    rewards.append(0.)
                    dones.append(True)
                # dones.append(done)
                values.append(value.detach().cpu().numpy().squeeze())
                old_log_probs.append(old_log_prob.detach().cpu().numpy())
                next_states.append(env.normalize(next_state, env.state_bound))

                state = next_state
                env.render()
                if np.any(done):
                    break

                step += 1
            episode = self.train_step * num_rollouts + i
            print(
                "|episode: {}| score:{:.2f} | steps {}| miss_max_idx {}".format(
                    episode,
                    sum(rewards),
                    res_step,
                    miss_max_idx))

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
        values_out = torch.cat([values_out, torch.zeros(values_out.shape[0], 1, 1).to(self.device)], dim=1)
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

        dt = str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p"))
        video_dir = self.videos_dir + '/video_ppo_' + dt

        try:
            os.makedirs(video_dir)
        except:
            print("directory '{}' already exists")
        sys.stdout = Logger(video_dir + "/log.txt")

        scores = []
        for episode in range(num_episodes):
            states, actions, rewards, dones, values, old_log_probs, _ = self.rollout(env, self.num_rollouts_per_update)

            score = rewards.sum(dim=-1).mean()

            T = rewards.shape[1]
            last_advantage = torch.zeros((rewards.shape[0], 1))
            last_return = torch.zeros(rewards.shape[0])
            returns = torch.zeros(rewards.shape)
            advantages = torch.zeros(rewards.shape)

            # calculate return and advantage
            for t in reversed(range(T)):
                # calc return
                last_return = rewards[:, t] + last_return * self.discount * (1 - dones[:, t]).float()
                returns[:, t] = last_return

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
                next_val = self.discount * values[:, t + 1] * (1 - dones[:, t]).float()[:, np.newaxis]
                delta = rewards[:, t][:, np.newaxis] + next_val - values[:, t]
                last_advantage = delta + self.discount * self.lmbda * last_advantage
                advantages[:, t] = last_advantage.squeeze()

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
            # print("episode: {} | score:{:.4f} | action_mean: {:.2f}, action_std: {:.2f}".format(
            #     episode, score, actions.mean().cpu(), actions.std().cpu()))
            self.train_step += 1
        print("Training finished. Result score: ", score)
        return scores
