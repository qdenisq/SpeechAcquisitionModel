import json
from pprint import pprint
import os
import numpy as np
import pandas as pd
import datetime
import sys
from torch.distributions import Normal

import torch
from src.reinforcement.goal_directed_model_based_rl.replay_buffer import ReplayBuffer

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'], 'size': 10})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

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


class ModelBasedMultiStepBackPropWithEnsembleClassifier:
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
        self.cdf_beta = kwargs['cdf_beta']


        self.action_penalty = kwargs['action_penalty']

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
        video_dir = self.videos_dir + '/video_ensemble_multi_step_' + dt
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

            # axes[0].cla()

            score = 0.

            self.agent.eval()
            self.model_dynamics.eval()
            step = 0
            miss_max_idx = -1
            terminated = False

            probs = []
            entropies = []

            while True:
                state_tensor = torch.from_numpy(state).float().to(self.device).view(1, -1)
                action = self.agent(state_tensor).detach().cpu().numpy().squeeze()
                action = action + self.noise_gamma * action_noise()
                ep_actions.append(action)
                action_denorm = env.denormalize(action, env.action_bound)
                next_state, reward, done, _ = env.step(action_denorm)

                probs.append(reward[0].detach().cpu().numpy())
                entropies.append(reward[1].detach().cpu().numpy())

                next_state_pred, next_state_pred_std, _ = self.model_dynamics(state_tensor, torch.from_numpy(action).float().to(self.device).view(1,-1))
                next_state_pred = env.denormalize(next_state_pred.detach().cpu().numpy().squeeze(), env.state_bound[:-env.goal_dim])

                ep_states.append(next_state)
                ep_states_pred.append(next_state_pred)
                next_state = env.normalize(next_state, env.state_bound)
                env.render()
                if episode % 10 != 0:
                    self.replay_buffer.add((state, action, next_state))

                miss = torch.abs(torch.from_numpy(next_state).float().to(self.device)[:-env.goal_dim][torch.from_numpy(np.array(env.state_goal_mask, dtype=np.uint8)).byte()] -
                                 torch.from_numpy(state).float().to(self.device)[-env.goal_dim:])

                misses.append(miss[:].max().detach().cpu().numpy())

                if len(misses) > 3 and np.all(np.array(misses[-3:]) > 0.1) and not terminated:
                    terminated = True
                    res_step = step
                    miss_max_idx = np.argmax(miss[:].detach().cpu().numpy())
                elif not terminated:
                    score = step
                if len(misses) > 3 and np.all(np.array(misses[-3:]) > 0.1) and (episode % 10 != 0 or self.replay_buffer.size() < 2 * self.minibatch_size):
                    miss_max_idx = np.argmax(miss[:].detach().cpu().numpy())
                    break
                if np.any(done):

                    break
                state = next_state
                step += 1
            scores.append(score)
            if not terminated:
                res_step = step
            if episode % 10 == 0:
                self.noise_gamma *= self.noise_decay

            if episode % 10 == 0 and episode > 1 and self.replay_buffer.size() > 2 * self.minibatch_size:
                n_audio = 26
                n_artic = 24
                n_artic_goal = 6

                # show fully predicted rollout given s0  and list of actions
                pred_states = []
                pred_states_std = []
                state = state0
                state_std = np.zeros(env.state_dim - env.goal_dim)
                pred_states_probs = []
                pred_state_prob = 1.
                for idx, a in enumerate(ep_actions):
                    state = env.normalize(state, env.state_bound)

                    pred_states.append(state)
                    pred_states_std.append(state_std)
                    state_tensor = torch.from_numpy(state).float().to(self.device).view(1, -1)
                    next_state_pred, next_state_pred_std, _ = self.model_dynamics(state_tensor,
                                                             torch.from_numpy(a).float().to(self.device).view(1, -1))

                    next_state_distr = Normal(next_state_pred, next_state_pred_std)
                    next_state_sampled_prob = (
                                next_state_distr.cdf(next_state_pred + self.cdf_beta) - next_state_distr.cdf(
                            next_state_pred - self.cdf_beta)).prod(dim=-1).detach().cpu().numpy().squeeze()
                    pred_state_prob *= next_state_sampled_prob
                    pred_states_probs.append(pred_state_prob)

                    next_state_pred = env.denormalize(next_state_pred.detach().cpu().numpy().squeeze(),
                                                      env.state_bound[:-env.goal_dim])
                    state = np.concatenate((next_state_pred, ep_states[idx][-env.goal_dim:]))
                    state_std = next_state_pred_std.detach().cpu().numpy().squeeze()

                # Share a X axis with each column of subplots
                fig, axes = plt.subplots(8, 2, figsize=(5, 13))
                cb = None
                # plt.ion()
                # plt.show()



                ep_states = env.normalize(np.array(ep_states), env.state_bound)
                im0 = axes[0, 0].imshow(np.array(ep_states)[:, :n_artic].T, vmin=-1., vmax=1.)
                axes[0, 0].set_title('rollout artic')
                plt.colorbar(im0, ax=axes[0, 0])

                im0 = axes[0, 1].imshow(np.array(ep_states)[:, n_artic: n_artic+n_audio].T, vmin=-1., vmax=1.)
                axes[0, 1].set_title('rollout acoustic')
                plt.colorbar(im0, ax=axes[0, 1])
                # im_pred = axes[1].imshow(np.array(ep_states_pred)[:, -n_audio:].T, vmin=vmin, vmax=vmax)

                im_pred = axes[1, 0].imshow(np.array(pred_states)[:, :n_artic].T, vmin=-1., vmax=1.)
                axes[1, 0].set_title('pred rollout artic')
                plt.colorbar(im_pred, ax=axes[1, 0])

                im_pred = axes[1, 1].imshow(np.array(pred_states)[:, n_artic: n_artic+n_audio].T, vmin=-1., vmax=1.)
                axes[1, 1].set_title('pred rollout acoustic')
                plt.colorbar(im_pred, ax=axes[1, 1])

                im_pred = axes[2, 0].imshow(np.array(pred_states_std)[:, :n_artic].T)
                axes[2, 0].set_title('pred rollout artic std')
                plt.colorbar(im_pred, ax=axes[2, 0])

                im_pred = axes[2, 1].imshow(np.array(pred_states_std)[:, n_artic: n_artic+n_audio].T)
                axes[2, 1].set_title('pred rollout acoustic std')
                plt.colorbar(im_pred, ax=axes[2, 1])

                if n_artic_goal > 0:
                    im1 = axes[3, 0].imshow(ep_states[:, -env.goal_dim: -env.goal_dim + n_artic_goal].T, vmin=-1., vmax=1.)
                    axes[3, 0].set_title('reference artic')
                    plt.colorbar(im1, ax=axes[3, 0])

                im1 = axes[3, 1].imshow(ep_states[:, -env.goal_dim + n_artic_goal:].T, vmin=-1., vmax=1.)
                axes[3, 1].set_title('reference acoustic')
                plt.colorbar(im1, ax=axes[3, 1])

                diff_img = np.abs(np.array([ep_states[i, -env.goal_dim:] - np.array(ep_states)[i + 1, :-env.goal_dim][env.state_goal_mask] for i in range(len(ep_states)-1)]))
                # diff_img_normed = env.normalize(diff_img.T, env.state_bound[:-env.goal_dim])
                diff_img_normed = diff_img

                if n_artic_goal > 0:
                    im2 = axes[4, 0].imshow(np.array(diff_img_normed[:, :n_artic_goal].T))
                    axes[4, 0].set_title('error artic')
                    plt.colorbar(im2, ax=axes[4, 0])

                im2 = axes[4, 1].imshow(np.array(diff_img_normed[:, -n_audio:].T))
                axes[4, 1].set_title('error acoustic')
                plt.colorbar(im2, ax=axes[4, 1])

                pred_err_img = np.abs(np.array(
                    [ep_states[i, :-env.goal_dim] - np.array(pred_states)[i, :-env.goal_dim] for i in
                     range(len(ep_states) - 1)]))
                # diff_img_normed = env.normalize(diff_img.T, env.state_bound[:-env.goal_dim])
                im3 = axes[5, 0].imshow(np.array(pred_err_img[:, :n_artic].T))
                axes[5, 0].set_title('pred error artic')
                plt.colorbar(im3, ax=axes[5, 0])

                im3 = axes[5, 1].imshow(np.array(pred_err_img[:, n_artic:n_artic + n_audio].T))
                axes[5, 1].set_title('pred error acoustic')
                plt.colorbar(im3, ax=axes[5, 1])

                im4 = axes[6, 0].imshow(np.array(probs).T, vmin=0., vmax=np.array(probs).T.max())
                # axes[5, 1].ylim((0, 1.0))
                axes[6, 0].set_title('pred prob')
                plt.colorbar(im4, ax=axes[6, 0])

                im4 = axes[6, 1].plot(np.array(entropies))
                axes[6, 1].set_ylim(bottom=0, top=np.array(entropies).max()+2)
                axes[6, 1].set_title('pred entropy')
                # plt.colorbar(im4, ax=axes[4, 1])

                axes[7, 1].plot(np.array(pred_states_probs))
                axes[7, 1].set_ylim(bottom=0, top=1.2)
                axes[7, 1].set_title('pred state probability')

                # if cb is None:
                # cb = plt.colorbar(im0, ax=axes[0, 1])
                # plt.colorbar(im_pred, ax=axes[1, 1])
                # plt.colorbar(im1, ax=axes[2, 1])
                # plt.colorbar(im2, ax=axes[3, 1])
                # plt.colorbar(im3, ax=axes[4, 1])
                plt.tight_layout()
                # plt.draw()
                # plt.pause(.001)

                fname = video_dir + '/episode_' + str(episode) + '_' + str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p_%S"))
                env.dump_episode(fname)
                fig.savefig(fname+".png")
                plt.close('all')

                sys.stdout.flush()

            # save model
            if episode % 50 == 0:
                with open(video_dir + '/model_dynamics.pickle', 'wb') as f:
                    torch.save(self.model_dynamics, f)
                with open(video_dir + '/agent.pickle', 'wb') as f:
                    torch.save(self.agent, f)

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
                    s1_pred, _, s1_pred_ensemble = self.model_dynamics(s0_batch.float().to(self.device), a_batch.float().to(self.device))
                    md_loss = torch.nn.L1Loss(reduce=False)(s1_pred_ensemble,
                                                                     s1_batch[:, :-env.goal_dim].float().to(self.device).repeat(s1_pred_ensemble.shape[0], 1, 1))
                    md_loss = md_loss.sum() / self.minibatch_size

                    # md_loss = torch.nn.SmoothL1Loss(reduction="sum")(s1_pred_ensemble[0,:,:], s1_batch[:, :-env.goal_dim].float().to(self.device)) / self.minibatch_size


                    md_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model_dynamics.parameters(), self.clip_grad)
                    self.model_dynamics_optim.step()

                    # s1_pred, _, s1_pred_ensemble = self.model_dynamics(s0_batch.float().to(self.device),
                    #                                                    a_batch.float().to(self.device))
                    # md_loss = torch.nn.SmoothL1Loss(reduce=False)(s1_pred_ensemble,
                    #                                               s1_batch[:, :-env.goal_dim].float().to(
                    #                                                   self.device).repeat(s1_pred_ensemble.shape[0], 1,
                    #                                                                       1))
                    #
                    # for md_idx in range(md_loss.shape[0]):
                    #     single_md_loss = md_loss[md_idx, :, :].sum() / self.minibatch_size
                    #
                    #     # md_loss = torch.nn.SmoothL1Loss(reduction="sum")(s1_pred_ensemble[0,:,:], s1_batch[:, :-env.goal_dim].float().to(self.device)) / self.minibatch_size
                    #
                    #     self.model_dynamics_optim.zero_grad()
                    #     single_md_loss.backward(retain_graph=True)
                    #     torch.nn.utils.clip_grad_norm_(self.model_dynamics.nets[md_idx].parameters(), self.clip_grad)
                    #     self.model_dynamics_optim.step()

                ##############################################################
                # train policy
                ##############################################################

                n_train_steps = round(
                    self.replay_buffer.size() / self.minibatch_size * self.num_epochs_actor + 1)
                self.agent.train()
                self.model_dynamics.eval()

                s1_pred_log_probs = []
                for _ in range(n_train_steps):
                    # train_step_i += 1
                    s0_batch, a_batch, s1_batch = self.replay_buffer.sample_batch(self.minibatch_size)

                    self.actor_optim.zero_grad()
                    actions_predicted = self.agent(s0_batch.float().to(self.device))
                    # predict state if predicted actions will be applied
                    s1_pred, s1_pred_std, s1_pred_ensmble = self.model_dynamics(s0_batch.float().to(self.device), actions_predicted)
                    # s1_pred = s1_pred_ensmble[0,:,:]
                    actor_loss = torch.nn.SmoothL1Loss(reduction='none')(
                        s1_pred[:, torch.from_numpy(np.array(env.state_goal_mask, dtype=np.uint8)).byte()],
                        s0_batch[:, -env.goal_dim:].float().to(self.device))

                    # s1_pred_log_prob = Normal(s1_pred, s1_pred_std).log_prob(s1_pred).sum(dim=-1).exp()

                    # TODO find tha way to scale density probability function
                    s1_pred_log_prob = (Normal(s1_pred, s1_pred_std).cdf(s1_pred + self.cdf_beta) - Normal(s1_pred, s1_pred_std).cdf(s1_pred - self.cdf_beta)).prod(dim=-1)

                    s1_pred_log_probs.append(s1_pred_log_prob.detach().cpu().numpy())
                    s1_pred_log_prob = torch.clamp(s1_pred_log_prob, max=1.0).detach()
                    actor_loss = actor_loss * s1_pred_log_prob.unsqueeze(1)
                    actor_loss = actor_loss.sum() / self.minibatch_size

                    # study this penalty
                    action_penalty = self.action_penalty * torch.mean(torch.abs(actions_predicted))
                    actor_loss += action_penalty

                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.clip_grad)
                    self.actor_optim.step()

                # if episode < 50:
                #     print(
                #         "|episode: {}| train step: {}| model_dynamics loss: {:.8f}| score:{:.2f} | steps {}| miss_max_idx {}".format(
                #             episode,
                #             train_step_i,
                #             md_loss.detach().cpu().numpy(),
                #             score,
                #             res_step,
                #             miss_max_idx))
                #     continue

                # n_train_steps = round(
                #     self.replay_buffer.size() / self.minibatch_size * self.num_epochs_actor + 1)
                # self.agent.train()
                # self.model_dynamics.eval()
                #
                # n_train_steps = n_train_steps
                # s1_pred_log_probs = []
                # avg_length = []
                # for _ in range(n_train_steps):
                #
                #     cur_state = init_state
                #     init_step = initital_cur_step
                #
                #     actor_loss = torch.Tensor([0.])
                #
                #     last_prob = torch.ones([self.minibatch_size, 1])
                #
                #     for i in range(init_step, refs.shape[1] - 1):
                #
                #
                #         action = self.agent(cur_state.to(self.device))
                #
                #         # predict state if predicted actions will be applied
                #         next_state_pred, next_state_pred_std, _ = self.model_dynamics(cur_state.float().to(self.device), action)
                #
                #         # next_state_pred_std = torch.clamp(next_state_pred_std, min=0.00001)
                #         next_state_pred_dist = Normal(next_state_pred, next_state_pred_std)
                #         # next_state_pred_sampled = torch.clamp(next_state_pred_sampled, min=-1.0, max=1.0)
                #         next_state_pred_prob = (next_state_pred_dist.cdf(next_state_pred + 5e-2)
                #                             - next_state_pred_dist.cdf(next_state_pred - 5e-2)).cumprod(dim=-1)[:, -1]
                #         last_prob = last_prob * next_state_pred_prob.detach().unsqueeze(1)
                #         s1_pred_log_probs.append(last_prob.detach().cpu().numpy())
                #         if torch.mean(last_prob) < 5e-1:
                #             break
                #
                #         next_state_ref = refs[:, i+1, :]
                #         next_state_pred_full = torch.cat((next_state_pred, next_state_ref), -1)
                #
                #         loss = torch.nn.SmoothL1Loss(reduction='none')(
                #             next_state_pred[:, torch.from_numpy(np.array(env.state_goal_mask, dtype=np.uint8)).byte()],
                #             cur_state[:, -env.goal_dim:]) * last_prob
                #
                #         loss = loss.sum() / self.minibatch_size
                #         # print(action.mean())
                #         # print(next_state_pred.mean())
                #         # print(next_state_pred_std.mean())
                #         # print(loss)
                #         # self.actor_optim.zero_grad()
                #         # loss.backward()
                #         # torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.clip_grad)
                #         # self.actor_optim.step()
                #
                #         # if torch.isnan(loss):
                #         #     k = 123
                #         actor_loss += loss
                #         cur_state = next_state_pred_full.detach()
                #
                #     actor_loss = actor_loss / (i - init_step + 1)
                #     avg_length.append(i)
                #
                #     # actor_loss = actor_loss.mean()
                #     if torch.min(actor_loss) > 1e-10:
                #         self.actor_optim.zero_grad()
                #         actor_loss.backward()
                #         torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.clip_grad)
                #         self.actor_optim.step()
                #
                # print(f"| roll len{np.mean(np.array(avg_length)):.2f} ", end='')


                print("|episode: {}| train step: {}| model_dynamics loss: {:.8f}| policy loss: {:.5f}| score:{:.2f} | steps {}| miss_max_idx {} | md_prob_mean {:.2f}".format(episode,
                                                                                                                              train_step_i,
                                                                                                                              md_loss.mean().detach().cpu().numpy(),
                                                                                                                              actor_loss.detach().cpu().numpy().squeeze(),
                                                                                                                              score,
                                                                                                                              res_step,
                                                                                                                              miss_max_idx,
                                                                                                                                np.mean(np.array(s1_pred_log_probs))))

        print("Training finished. Result score: ", score)
        return scores