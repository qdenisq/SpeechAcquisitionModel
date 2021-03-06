import json
from pprint import pprint
import os
import numpy as np
import pandas as pd
import datetime
import sys

import torch
from collections import deque
from src.reinforcement.goal_directed_model_based_rl.replay_buffer import ReplayBuffer

import matplotlib.pyplot as plt


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


class ModelBasedMultiStepBackProp:
    def __init__(self, agent=None, model_dynamics=None, **kwargs):
        self.agent = agent

        self.actor_optim = torch.optim.Adam(agent.parameters(), lr=kwargs['actor_lr'], eps=kwargs['learning_rate_eps'])
        # self.actor_optim = torch.optim.SGD(agent.parameters(), lr=kwargs['actor_lr'])

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
        self.videos_dir = kwargs['videos_dir']
        self.noise_gamma = 1.0
        self.noise_decay = kwargs['noise_decay']

    def train(self, env, num_episodes, dir=None):
        """
        Train the agent to solve environment
        :param env: environment object (ReacherEnvironment)
        :param num_episodes: number of episodes (int)
        :return scores: list of scores for each episode (list)
        """

        action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(env.action_dim), sigma=0.05)
        scores = []
        train_step_i = 0

        dt = str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p"))
        # writer = tf.summary.FileWriter(settings['summary_dir'] + '/summary_md_' + dt, sess.graph)
        video_dir = self.videos_dir + '/video_mb1step_' + dt
        if dir is not None:
            video_dir = dir
        try:
            os.makedirs(video_dir)
        except:
            print("directory '{}' already exists")

        sys.stdout = Logger(video_dir + "/log.txt")

        init_states = []
        refs = []
        for _ in range(self.minibatch_size):
            state = env.reset()

            state = env.normalize(state, env.state_bound)
            ref = env.get_current_reference()
            init_states.append(state)
            refs.append(ref)
        init_state = torch.from_numpy(np.array(init_states)).float().to(self.device).view(self.minibatch_size, -1)
        initital_cur_step = env.current_step

        refs = np.array(refs)
        refs = env.normalize(refs, env.goal_bound)
        refs = torch.from_numpy(refs).float().to(self.device)

        for episode in range(num_episodes):
            ep_states = []
            ep_states_pred = []
            ep_actions = []
            misses = []
            # rollout
            T = len(env.get_current_reference())
            state0 = env.reset()
            state = state0
            env.render()
            ep_states.append(state)
            ep_states_pred.append(state[:-env.goal_dim])
            state = env.normalize(state, env.state_bound)



            ep_states_normed = []
            ep_states_normed.append(state)

            score = 0.

            self.agent.eval()
            self.model_dynamics.eval()
            step = 0
            miss_max_idx = -1
            while True:
                state_tensor = torch.from_numpy(state).float().to(self.device).view(1, -1)
                action = self.agent(state_tensor).detach().cpu().numpy().squeeze()
                action = action + self.noise_gamma * action_noise()
                ep_actions.append(action)
                action_denorm = env.denormalize(action, env.action_bound)
                next_state, reward, done, _ = env.step(action_denorm)

                next_state_pred, _ = self.model_dynamics(state_tensor, torch.from_numpy(action).float().to(self.device).view(1,-1))
                next_state_pred = env.denormalize(next_state_pred.detach().cpu().numpy().squeeze(), env.state_bound[:-env.goal_dim])

                ep_states.append(next_state)
                ep_states_pred.append(next_state_pred)
                next_state = env.normalize(next_state, env.state_bound)
                ep_states_normed.append(next_state)
                env.render()
                if episode % 10 != 0:
                    self.replay_buffer.add((state, action, next_state))

                score += reward
                miss = torch.abs(torch.from_numpy(next_state).float().to(self.device)[:-env.goal_dim][torch.from_numpy(np.array(env.state_goal_mask, dtype=np.uint8)).byte()] -
                                 torch.from_numpy(state).float().to(self.device)[-env.goal_dim:])

                misses.append(miss[:].max().detach().cpu().numpy())
                if len(misses) > 3 and np.all(np.array(misses[-3:]) > 0.1) and episode % 10 != 0:
                    miss_max_idx = np.argmax(miss[:].detach().cpu().numpy())
                    break

                if np.any(done):
                    break
                state = next_state
                step += 1
            scores.append(score)

            # if episode % 10 != 0:
            #     self.replay_buffer.add(zip(ep_states_normed[:-1], ep_actions, ep_states_normed[1:]))

            if episode % 10 == 0:
                self.noise_gamma *= self.noise_decay

            if episode % 10 == 0 and episode > 1:
                n_audio = 26
                n_artic = 24

                n_artic_goal = 12
                # show fully predicted rollout given s0  and list of actions
                pred_states = []
                state = state0
                for idx, a in enumerate(ep_actions):
                    state = env.normalize(state, env.state_bound)
                    pred_states.append(state)

                    state_tensor = torch.from_numpy(state).float().to(self.device).view(1, -1)
                    next_state_pred, _ = self.model_dynamics(state_tensor,
                                                             torch.from_numpy(a).float().to(self.device).view(1, -1))
                    next_state_pred = env.denormalize(next_state_pred.detach().cpu().numpy().squeeze(),
                                                      env.state_bound[:-env.goal_dim])
                    state = np.concatenate((next_state_pred, ep_states[idx][-env.goal_dim:]))

                # Share a X axis with each column of subplots
                fig, axes = plt.subplots(5, 2, figsize=(10, 10))
                cb = None
                # plt.ion()
                # plt.show()



                ep_states = env.normalize(np.array(ep_states), env.state_bound)
                im0 = axes[0, 0].imshow(np.array(ep_states)[:, :n_artic].T, vmin=-1., vmax=1.)
                axes[0, 0].set_title('rollout_artic')
                plt.colorbar(im0, ax=axes[0, 0])

                im0 = axes[0, 1].imshow(np.array(ep_states)[:, n_artic: n_artic+n_audio].T, vmin=-1., vmax=1.)
                axes[0, 1].set_title('rollout_acoustic')
                plt.colorbar(im0, ax=axes[0, 1])
                # im_pred = axes[1].imshow(np.array(ep_states_pred)[:, -n_audio:].T, vmin=vmin, vmax=vmax)

                im_pred = axes[1, 0].imshow(np.array(pred_states)[:, :n_artic].T, vmin=-1., vmax=1.)
                axes[1, 0].set_title('pred_rollout_artic')
                plt.colorbar(im_pred, ax=axes[1, 0])

                im_pred = axes[1, 1].imshow(np.array(pred_states)[:, n_artic: n_artic+n_audio].T, vmin=-1., vmax=1.)
                axes[1, 1].set_title('pred_rollout_acoustic')
                plt.colorbar(im_pred, ax=axes[1, 1])

                if n_artic_goal > 0:
                    im1 = axes[2, 0].imshow(ep_states[:, -env.goal_dim: -env.goal_dim + n_artic_goal].T, vmin=-1., vmax=1.)
                    axes[2, 0].set_title('reference_artic')
                    plt.colorbar(im1, ax=axes[2, 0])

                im1 = axes[2, 1].imshow(ep_states[:, -env.goal_dim + n_artic_goal:].T, vmin=-1., vmax=1.)
                axes[2, 1].set_title('reference_acoustic')
                plt.colorbar(im1, ax=axes[2, 1])

                diff_img = np.abs(np.array([ep_states[i, -env.goal_dim:] - np.array(ep_states)[i + 1, :-env.goal_dim][env.state_goal_mask] for i in range(len(ep_states)-1)]))
                # diff_img_normed = env.normalize(diff_img.T, env.state_bound[:-env.goal_dim])
                diff_img_normed = diff_img

                if n_artic_goal > 0:
                    im2 = axes[3, 0].imshow(np.array(diff_img_normed[:, :n_artic_goal].T))
                    axes[3, 0].set_title('error_artic')
                    plt.colorbar(im2, ax=axes[3, 0])

                im2 = axes[3, 1].imshow(np.array(diff_img_normed[:, -n_audio:].T))
                axes[3, 1].set_title('error_acoustic')
                plt.colorbar(im2, ax=axes[3, 1])

                pred_err_img = np.abs(np.array(
                    [ep_states[i, :-env.goal_dim] - np.array(pred_states)[i, :-env.goal_dim] for i in
                     range(len(ep_states) - 1)]))
                # diff_img_normed = env.normalize(diff_img.T, env.state_bound[:-env.goal_dim])
                im3 = axes[4, 0].imshow(np.array(pred_err_img[:, :n_artic].T))
                axes[4, 0].set_title('pred_error_artic')
                plt.colorbar(im3, ax=axes[4, 0])

                im3 = axes[4, 1].imshow(np.array(pred_err_img[:, n_artic:n_artic + n_audio].T))
                axes[4, 1].set_title('pred_error_acoustic')
                plt.colorbar(im3, ax=axes[4, 1])

                # if cb is None:
                # cb = plt.colorbar(im0, ax=axes[0, 1])
                # plt.colorbar(im_pred, ax=axes[1, 1])
                # plt.colorbar(im1, ax=axes[2, 1])
                # plt.colorbar(im2, ax=axes[3, 1])
                # plt.colorbar(im3, ax=axes[4, 1])
                plt.tight_layout()
                # plt.draw()
                # plt.pause(.001)

                fname = video_dir + '/episode_' + str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p_%S"))
                env.dump_episode(fname)
                fig.savefig(fname+".png")
                plt.close('all')

                sys.stdout.flush()

            if self.replay_buffer.size() > 2 * self.minibatch_size:
                # train nets couple of times relative to the increase of replay buffer
                n_train_steps = round(
                    self.replay_buffer.size() / self.minibatch_size * self.num_epochs_model_dynamics + 1)
                ##############################################################
                # train model dynamics
                ##############################################################
                self.model_dynamics.train()
                for _ in range(n_train_steps):
                    train_step_i += 1
                    s0_batch, a_batch, s1_batch = self.replay_buffer.sample_batch(self.minibatch_size)

                    self.model_dynamics_optim.zero_grad()
                    s1_pred, _ = self.model_dynamics(s0_batch.float().to(self.device), a_batch.float().to(self.device))
                    md_loss = torch.nn.SmoothL1Loss(reduction="sum")(s1_pred, s1_batch[:, :-env.goal_dim].float().to(self.device)) / self.minibatch_size
                    md_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model_dynamics.parameters(), self.clip_grad)
                    self.model_dynamics_optim.step()

                ##############################################################
                # train policy
                ##############################################################

                n_train_steps = round(
                    self.replay_buffer.size() / self.minibatch_size * self.num_epochs_actor + 1)
                n_train_steps = 20
                self.agent.train()
                self.model_dynamics.eval()

                cur_step = initital_cur_step
                cur_steps = np.array([cur_step]*self.minibatch_size)

                state = init_state
                misses = deque(maxlen=3)
                for _ in range(n_train_steps):

                    action = self.agent(state)
                    next_state_pred, _ = self.model_dynamics(state, action)
                    next_state_ref = torch.cat([refs[l, cur_steps[l] + 1, :] for l in range(self.minibatch_size)]).view(self.minibatch_size, -1)
                    next_state_pred_full = torch.cat((next_state_pred, next_state_ref), -1)

                    actor_loss = torch.nn.SmoothL1Loss(reduction="sum")(
                        next_state_pred[:, torch.from_numpy(np.array(env.state_goal_mask, dtype=np.uint8)).byte()],
                        state[:, -env.goal_dim:]) / self.minibatch_size

                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.clip_grad)
                    self.actor_optim.step()

                    state = next_state_pred_full
                    cur_steps += 1

                    miss = torch.abs(next_state_pred_full[:, :-env.goal_dim][:,
                                     torch.from_numpy(np.array(env.state_goal_mask, dtype=np.uint8)).byte()] -
                                     state[:, -env.goal_dim:])
                    max_miss, _ = torch.max(miss, -1)
                    misses.append(max_miss.detach().cpu().numpy().squeeze())
                    misses_arr = np.array(misses)
                    for j in range(self.minibatch_size):
                        if (misses_arr.shape[0] > 2 and np.all(misses_arr[:, j] > 0.1)) or cur_steps[j] > refs.shape[1] - 2:
                            # substitute collapsed trajectory with new one
                            misses_arr[:, j] = 0.
                            state[j, :] = init_state[j, :]
                            cur_steps[j] = initital_cur_step
                    misses.clear()
                    for k in range(misses_arr.shape[0]):
                        misses.append(misses_arr[k, :])

                print(np.mean(cur_steps), end='; ')

                print("|episode: {}| train step: {}| model_dynamics loss: {:.8f}| policy loss: {:.5f}| score:{:.2f} | steps {}| miss_max_idx {}".format(episode,
                                                                                                                                                        train_step_i,
                                                                                                                                                        md_loss.detach().cpu().numpy(),
                                                                                                                                                        actor_loss.detach().cpu().numpy(),
                                                                                                                                                        score,
                                                                                                                                                        step,
                                                                                                                                                        miss_max_idx))

        print("Training finished. Result score: ", score)
        return scores