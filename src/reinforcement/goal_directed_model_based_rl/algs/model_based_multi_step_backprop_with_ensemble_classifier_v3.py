import json
from pprint import pprint
import os
import numpy as np
import pandas as pd
import datetime
import sys
from torch.distributions import Normal

import copy
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


class ModelBasedMultiStepBackPropWithEnsembleClassifierV3:
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

        self.action_penalty = kwargs['action_penalty']

        self.cdf_beta = kwargs['cdf_beta']

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
        video_dir = self.videos_dir + '/video_ensemble_multi_step_V3_' + dt
        if dir is not None:
            video_dir = dir
        try:
            os.makedirs(video_dir)
        except:
            print("directory '{}' already exists")

        sys.stdout = Logger(video_dir + "/log.txt")

        for episode in range(num_episodes):
            ep_states = []
            ep_states_pred = []
            ep_actions = []
            misses = []
            # rollout
            T = len(env.get_current_reference())
            states = np.zeros((T, env.state_dim))
            actions = np.zeros((T, env.action_dim))
            next_states = np.zeros((T, env.state_dim))
            dones = np.zeros(T)


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

            reference_probs = []
            reference_entropies = []
            reference_classify_hidden = None

            while True:
                state_tensor = torch.from_numpy(state).float().to(self.device).view(1, -1)
                action = self.agent(state_tensor).detach().cpu().numpy().squeeze()
                action = action + self.noise_gamma * action_noise()
                ep_actions.append(action)
                action_denorm = env.denormalize(action, env.action_bound)
                next_state, reward, done, _ = env.step(action_denorm)

                # reference classify
                sound_ref = env.get_current_reference()[step, -env.audio_dim:]
                ref_prob, ref_entropy, reference_classify_hidden = env.classify(torch.from_numpy(sound_ref).float().to(self.device).view(1,1,-1), reference_classify_hidden)
                reference_probs.append(ref_prob.detach().cpu().numpy())
                reference_entropies.append(ref_entropy.detach().cpu().numpy())

                if isinstance(reward, float):
                    probs.append(0)
                    entropies.append(0)
                else:
                    probs.append(reward[0].detach().cpu().numpy())
                    entropies.append(reward[1].detach().cpu().numpy())

                next_state_pred, next_state_pred_std, _ = self.model_dynamics(state_tensor, torch.from_numpy(action).float().to(self.device).view(1,-1))
                next_state_pred = env.denormalize(next_state_pred.detach().cpu().numpy().squeeze(), env.state_bound[:-env.goal_dim])

                ep_states.append(next_state)
                ep_states_pred.append(next_state_pred)
                next_state = env.normalize(next_state, env.state_bound)

                states[step, :] = state
                actions[step, :] = action
                next_states[step, :] = next_state
                dones[step] = 1

                env.render()

                miss = torch.abs(torch.from_numpy(next_state).float().to(self.device)[:-env.goal_dim][torch.from_numpy(np.array(env.state_goal_mask, dtype=np.uint8)).byte()] -
                                 torch.from_numpy(state).float().to(self.device)[-env.goal_dim:])

                misses.append(miss[:].max().detach().cpu().numpy())

                if len(misses) > 3 and np.all(np.array(misses[-3:]) > 0.1) and not terminated:
                    terminated = True
                    res_step = step
                    miss_max_idx = np.argmax(miss[:].detach().cpu().numpy())
                elif not terminated:
                    score = step
                if len(misses) > 3 and np.all(np.array(misses[-3:]) > 0.1) and (episode % 10 != 0 or self.replay_buffer.size() * 3 < 2 * self.minibatch_size):
                    miss_max_idx = np.argmax(miss[:].detach().cpu().numpy())
                    break
                if np.any(done):

                    break
                state = next_state
                step += 1




            if episode % 10 != 0:
                self.replay_buffer.add((states, actions, next_states, dones))

            scores.append(score)
            if not terminated:
                res_step = step
            if episode % 10 == 0:
                self.noise_gamma *= self.noise_decay

            if episode % 10 == 0 and episode > 1 and self.replay_buffer.size() * 3 > 2 * self.minibatch_size:
                n_audio = 26
                n_artic = 24
                n_artic_goal = 24

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
                    next_state_sampled_prob = (next_state_distr.cdf(next_state_pred + self.cdf_beta) - next_state_distr.cdf(
                        next_state_pred - self.cdf_beta)).prod(dim=-1).detach().cpu().numpy().squeeze()
                    pred_state_prob *= next_state_sampled_prob
                    pred_states_probs.append(pred_state_prob)

                    next_state_pred = env.denormalize(next_state_pred.detach().cpu().numpy().squeeze(),
                                                      env.state_bound[:-env.goal_dim])
                    state = np.concatenate((next_state_pred, ep_states[idx][-env.goal_dim:]))
                    state_std = next_state_pred_std.detach().cpu().numpy().squeeze()





                # Share a X axis with each column of subplots
                fig, axes = plt.subplots(9, 2, figsize=(5, 17))
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
                axes[6, 0].set_title('classify prob')
                plt.colorbar(im4, ax=axes[6, 0])

                im4 = axes[6, 1].plot(np.array(entropies))
                axes[6, 1].set_ylim(bottom=0, top=np.array(entropies).max()+2)
                axes[6, 1].set_title('classify entropy')
                # plt.colorbar(im4, ax=axes[4, 1])

                axes[7, 1].plot(np.array(pred_states_probs))
                axes[7, 1].set_ylim(bottom=0, top=1.2)
                axes[7, 1].set_title('pred state probability')

                im = axes[8, 0].imshow(np.array(reference_probs).T, vmin=0., vmax=np.array(reference_probs).T.max())
                # axes[5, 1].ylim((0, 1.0))
                axes[8, 0].set_title('classify ref prob')
                plt.colorbar(im, ax=axes[8, 0])

                im = axes[8, 1].plot(np.array(reference_entropies).T)
                axes[8, 1].set_ylim(bottom=0, top=np.array(reference_entropies).max()+2)
                # axes[5, 1].ylim((0, 1.0))
                axes[8, 1].set_title('classify ref entropy')



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

            if self.replay_buffer.size() * 3 > 2 * self.minibatch_size:
                # train nets couple of times relative to the increase of replay buffer

                ##############################################################
                # train model dynamics
                ##############################################################


                s0_batch, a_batch, s1_batch, dones_batch = self.replay_buffer.sample_batch(self.replay_buffer.size())

                state_mask = dones_batch.repeat(s0_batch.shape[-1], 1, 1).permute(1, 2, 0).byte()
                action_mask = dones_batch.repeat(a_batch.shape[-1], 1, 1).permute(1, 2, 0).byte()

                s0_batch_masked = s0_batch.masked_select(state_mask).view(-1, env.state_dim)
                a_batch_masked = a_batch.masked_select(action_mask).view(-1, env.action_dim)
                s1_batch_masked = s1_batch.masked_select(state_mask).view(-1, env.state_dim)

                md_replay_buffer = ReplayBuffer(self.replay_buffer.size() * 30)

                [md_replay_buffer.add((s0_batch_masked[i].detach().cpu().numpy(),
                                          a_batch_masked[i].detach().cpu().numpy(),
                                          s1_batch_masked[i].detach().cpu().numpy())) for i in range(s0_batch_masked.shape[0])]

                n_train_steps = round(
                    md_replay_buffer.size() / self.minibatch_size * self.num_epochs_model_dynamics + 1)

                self.model_dynamics.train()
                for _ in range(n_train_steps):
                    train_step_i += 1
                    s0_batch, a_batch, s1_batch = md_replay_buffer.sample_batch(self.minibatch_size)

                    self.model_dynamics_optim.zero_grad()
                    s1_pred, _, s1_pred_ensemble = self.model_dynamics(s0_batch.float().to(self.device), a_batch.float().to(self.device))
                    md_loss = torch.nn.SmoothL1Loss(reduce=False)(s1_pred_ensemble,
                                                            s1_batch[:, :-env.goal_dim].float().to(self.device).repeat(s1_pred_ensemble.shape[0], 1, 1))
                    md_loss = md_loss.sum() / s0_batch.shape[0]

                    md_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model_dynamics.parameters(), self.clip_grad)
                    self.model_dynamics_optim.step()

                ##############################################################
                # train policy
                ##############################################################

                self.model_dynamics.eval()

                s0_batch, a_batch, s1_batch, dones_batch = self.replay_buffer.sample_batch(self.replay_buffer.size())

                # calculate probabilities of trajectories
                next_state_pred, next_state_pred_std, _ = self.model_dynamics(s0_batch.view(-1, env.state_dim).float(),
                                                                              a_batch.view(-1, env.action_dim).float())
                next_state_dist = Normal(next_state_pred, next_state_pred_std)
                next_state_prob = (next_state_dist.cdf(s1_batch.float().view(-1, env.state_dim)[:, :-env.goal_dim] + self.cdf_beta)
                                   - next_state_dist.cdf(s1_batch.float().view(-1, env.state_dim)[:, :-env.goal_dim] - self.cdf_beta)).prod(dim=-1)

                # compute cum prod of trajectory probability
                next_state_prob = next_state_prob.view(s0_batch.shape[0], -1).cumprod(dim=-1)

                state_mask = dones_batch.repeat(s0_batch.shape[-1], 1, 1).permute(1, 2, 0).byte()
                action_mask = dones_batch.repeat(a_batch.shape[-1], 1, 1).permute(1, 2, 0).byte()

                s0_batch_masked = s0_batch.masked_select(state_mask).view(-1, env.state_dim)
                a_batch_masked = a_batch.masked_select(action_mask).view(-1, env.action_dim)
                s1_batch_masked = s1_batch.masked_select(state_mask).view(-1, env.state_dim)
                s0_prob = torch.cat((torch.ones(s0_batch.shape[0], 1), next_state_prob[:, :-1]), dim=-1)
                s0_prob_masked = s0_prob.masked_select(dones_batch.byte())

                agent_replay_buffer = ReplayBuffer(self.replay_buffer.size() * 30)

                [agent_replay_buffer.add((s0_batch_masked[i].detach().cpu().numpy(),
                                          a_batch_masked[i].detach().cpu().numpy(),
                                          s1_batch_masked[i].detach().cpu().numpy(),
                                          s0_prob_masked[i].detach().cpu().numpy())) for i in range(s0_batch_masked.shape[0])]

                s1_pred_log_probs = []
                self.agent.train()
                self.model_dynamics.eval()
                for _ in range(n_train_steps):
                    # train_step_i += 1

                    s0_batch, a_batch, s1_batch, s0_prob_batch = agent_replay_buffer.sample_batch(self.minibatch_size)

                    s0_batch = s0_batch.detach()
                    s0_prob_batch = s0_prob_batch.detach()

                    actions_predicted = self.agent(s0_batch.float().to(self.device))
                    # predict state if predicted actions will be applied
                    s1_pred, s1_pred_std, s1_pred_ensmble = self.model_dynamics(s0_batch.float().to(self.device), actions_predicted)
                    # s1_pred = s1_pred_ensmble[0,:,:]
                    actor_loss = torch.nn.MSELoss(reduction='none')(
                        s1_pred[:, torch.from_numpy(np.array(env.state_goal_mask, dtype=np.uint8)).byte()],
                        s0_batch[:, -env.goal_dim:].float().to(self.device))

                    # s1_pred_log_prob = Normal(s1_pred, s1_pred_std).log_prob(s1_pred).sum(dim=-1).exp()

                    s1_pred_prob = (Normal(s1_pred, s1_pred_std).cdf(s1_pred + self.cdf_beta) - Normal(s1_pred, s1_pred_std).cdf(s1_pred - self.cdf_beta)).prod(dim=-1)
                    s1_pred_prob = s1_pred_prob * s0_prob_batch
                    # s1_pred_prob = s1_pred_prob

                    s1_pred_log_probs.append(s1_pred_prob.detach().cpu().numpy().squeeze())
                    actor_loss = actor_loss * s1_pred_prob.detach().unsqueeze(1)
                    actor_loss = actor_loss.sum() / self.minibatch_size
                    if torch.isnan(actor_loss):
                        k = 7
                    actor_loss = actor_loss
                    # study this penalty
                    action_penalty = self.action_penalty * torch.mean(torch.abs(actions_predicted))
                    # actor_loss += action_penalty
                    self.actor_optim.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.clip_grad)
                    self.actor_optim.step()

                print("|episode: {}| train step: {}| model_dynamics loss: {:.8f}| policy loss: {:.5f}| score:{:.2f} | steps {}| miss_max_idx {} | md_prob_mean {:.2f}".format(episode,
                                                                                                                              train_step_i,
                                                                                                                              np.mean(md_loss.detach().cpu().numpy()),
                                                                                                                              actor_loss.detach().cpu().numpy().squeeze(),
                                                                                                                              score,
                                                                                                                              res_step,
                                                                                                                              miss_max_idx,
                                                                                                                                np.mean(np.array(s1_pred_log_probs))))

        print("Training finished. Result score: ", score)
        return scores