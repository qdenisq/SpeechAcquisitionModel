from random import seed
from random import random as rnd
from random import randrange
import random
import datetime
import os
import pickle
from collections import deque
from math import sqrt
from sklearn.metrics import mean_squared_error
import numpy as np
import tensorflow as tf

import torch as torch
import torch.nn as nn
import torch.nn.functional as F

from src.VTL.vtl_environment import VTLEnv

class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=0):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s0, g0, a, s1, g1, target):
        if self.count < self.buffer_size:
            self.buffer.append((s0, g0, a, s1, g1, target))
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append((s0, g0, a, s1, g1, target))

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s0_batch = np.array([_[0] for _ in batch])
        g0_batch = np.array([_[1] for _ in batch])
        a_batch = np.array([_[2] for _ in batch])
        s1_batch = np.array([_[3] for _ in batch])
        g1_batch = np.array([_[4] for _ in batch])
        target_batch = np.array([_[5] for _ in batch])

        return s0_batch, g0_batch, a_batch, s1_batch, g1_batch, target_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0


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


def normalize(data, bound):
    largest = np.array([max(abs(y[0]), abs(y[1])) for y in bound])
    normed_data = data / largest
    return normed_data


def denormalize(normed_data, bound):
    largest = np.array([max(abs(y[0]), abs(y[1])) for y in bound])
    data = normed_data * largest
    return data



def train(settings, env, replay_buffer, reference_trajectory):

    # setup all summaries and dump directories

    num_episodes = 5000
    action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(env.action_dim), sigma=0.001)
    s_dim = settings['state_dim']
    g_dim = settings['goal_dim']
    a_dim = settings['action_dim']
    s_bound = settings['state_bound']
    a_bound = settings['action_bound']
    g_bound = settings['goal_bound']

    ###########################################
    # Create model dynamics and policy
    ###########################################


    ################################################################
    # Model Dynamics
    ################################################################


    class ModelDynamicsNet(nn.Module):
        def __init__(self, s_dim, g_dim, a_dim):
            super(ModelDynamicsNet, self).__init__()
            # an affine operation: y = Wx + b
            self.fc1 = nn.Linear((s_dim + g_dim + a_dim), (s_dim + g_dim + a_dim) * 2)
            torch.nn.init.xavier_uniform_(self.fc1.weight)

            self.fc2 = nn.Linear((s_dim + g_dim + a_dim) * 2, s_dim * 4)
            torch.nn.init.xavier_uniform_(self.fc2.weight)

            self.fc3 = nn.Linear(s_dim * 4, s_dim + g_dim)
            torch.nn.init.xavier_uniform_(self.fc3.weight)

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            return x

        def num_flat_features(self, x):
            size = x.size()[1:]  # all dimensions except the batch dimension
            num_features = 1
            for s in size:
                num_features *= s
            return num_features

    md_net = ModelDynamicsNet(s_dim, g_dim, a_dim)
    md_criterion = nn.MSELoss()
    md_optimizer = torch.optim.Adadelta(md_net.parameters())

    ################################################################
    # Policy
    ################################################################


    class PolicyNet(nn.Module):
        def __init__(self, s_dim, g_dim, a_dim):
            super(PolicyNet, self).__init__()
            # an affine operation: y = Wx + b

            self.fc1 = nn.Linear(s_dim + 2 * g_dim, 2 * (s_dim + 2 * g_dim))
            torch.nn.init.xavier_uniform_(self.fc1.weight)

            self.fc2 = nn.Linear(2 * (s_dim + 2 * g_dim), (s_dim + 2 * g_dim))
            torch.nn.init.xavier_uniform_(self.fc2.weight)

            self.fc3 = nn.Linear((s_dim + 2 * g_dim), a_dim)
            torch.nn.init.xavier_uniform_(self.fc3.weight)

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            return x

        def num_flat_features(self, x):
            size = x.size()[1:]  # all dimensions except the batch dimension
            num_features = 1
            for s in size:
                num_features *= s
            return num_features

    policy_net = PolicyNet(s_dim, g_dim, a_dim)
    policy_optimizer = torch.optim.Adadelta(policy_net.parameters())

    dt = str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p"))
    # writer = tf.summary.FileWriter(settings['summary_dir'] + '/summary_md_' + dt, sess.graph)
    video_dir = settings['videos_dir'] + '/video_md_' + dt
    try:
        os.makedirs(video_dir)
    except:
        print("directory '{}' already exists")
    minibatch_size = settings['minibatch_size']
    save_step = settings['save_video_step']
    ###########################################
    # main train loop
    ###########################################
    train_step_i = 0

    for i in range(num_episodes):
        # pick random initial state from the reference trajectory
        s0_index = randrange(0, reference_trajectory.shape[0] - 1)
        if i % save_step == 0:
            s0_index = 0

        s0 = reference_trajectory[s0_index]
        g0 = s0
        env.reset(s0)

        r = []
        # rollout episode
        for j in range(s0_index, len(reference_trajectory) - 1):
            target = reference_trajectory[j + 1]
            # normalize data before feeding it to policy
            s0_normed = normalize(np.reshape(s0, (1, s_dim)), s_bound)
            g0_normed = normalize(np.reshape(g0, (1, g_dim)), g_bound)
            target_normed = normalize(np.reshape(target, (1, g_dim)), g_bound)
            policy_X = torch.from_numpy(np.concatenate((s0_normed, g0_normed, target_normed), axis=1)).float()


            action_normed = policy_net(policy_X).detach().numpy()
            # add noise
            a_noise = action_noise()
            if i % save_step == 0:
                action_normed += a_noise
            else:
                action_normed += a_noise

            # denormalize action
            action = denormalize(action_normed, settings['action_bound'])
            action = np.reshape(action, (a_dim))
            # make a step
            s1, _ = env.step(action)
            g1 = s1

            env.render()

            # calc reward
            g1_normed = normalize(g1, g_bound)
            tar_normed = normalize(target, g_bound)
            miss = abs(tar_normed - g1_normed)
            if i % save_step != 0:
                replay_buffer.add(s0, g0, action, s1, g1, target)
            s0 = s1
            g0 = g1
            # some sort of early stopping
            # Need to change for sure in order to avoid going out of bounds at least
            if max(miss) > 0.1 and (i % save_step != 0 or replay_buffer.size() < minibatch_size):
                break

        # dump video
        if i % save_step == 0 and replay_buffer.size() > minibatch_size:
            fname = video_dir + '/episode_' + str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p_%S"))
            env.dump_episode(fname)
        # train model_dynamics and policy

        if replay_buffer.size() > minibatch_size:
            # train nets couple of times relative to the increase of replay buffer
            n_train_steps = round(replay_buffer.size() / replay_buffer.buffer_size * settings['max_train_per_simulation'] + 1)
            for _ in range(n_train_steps):
                train_step_i += 1
                s0_batch, g0_batch, a_batch, s1_batch, g1_batch, target_batch = \
                    replay_buffer.sample_batch(minibatch_size)

                # train model_dynamics
                # normalize data before train
                s0_batch_normed = normalize(s0_batch, s_bound)
                g0_batch_normed = normalize(g0_batch, g_bound)
                a_batch_normed = normalize(a_batch, a_bound)
                s1_batch_normed = normalize(s1_batch, s_bound)
                g1_batch_normed = normalize(g1_batch, g_bound)
                target_batch_normed = normalize(target_batch, g_bound)

                # zero grad all the staff
                md_optimizer.zero_grad()
                # train model dynamics
                md_input_X = np.concatenate((s0_batch_normed,
                                             g0_batch_normed,
                                             a_batch_normed), axis=1)
                md_input_X = torch.from_numpy(md_input_X).float()

                md_pred = md_net(md_input_X)

                md_target_Y = np.concatenate((s1_batch_normed,
                                              g1_batch_normed), axis=1)

                md_target_Y = torch.from_numpy(md_target_Y).float()

                md_loss = md_criterion(md_pred, md_target_Y)
                md_loss.backward()
                md_optimizer.step()


                #############################################################
                # train policy
                ##############################################################

                # zero grad
                policy_optimizer.zero_grad()
                # predict actions
                policy_input_X = np.concatenate((s0_batch_normed,
                                                 g0_batch_normed,
                                                 target_batch_normed), axis=1)
                policy_input_X = torch.from_numpy(policy_input_X).float()


                actions_normed = policy_net(policy_input_X).float()

                # predict state if predicted actions will be applied
                md_input_X = np.concatenate((s0_batch_normed,
                                             g0_batch_normed), axis=1)

                md_input_X = torch.from_numpy(md_input_X).float()

                # now stack the rest action tensor
                md_input_X = torch.cat((md_input_X, actions_normed), 1)

                md_target_Y = np.concatenate((s1_batch_normed,
                                             target_batch_normed), axis=1)
                md_target_Y = torch.from_numpy(md_target_Y).float()

                md_pred = md_net(md_input_X)
                loss = md_criterion(md_pred[:, s_dim:], md_target_Y[:, s_dim:])
                #
                # action_grads = torch.autograd.grad(loss, actions_normed, allow_unused=True)
                #
                # torch.autograd.backward([actions_normed], [action_grads[0]])
                loss.backward()
                policy_optimizer.step()

                expected_actions_normed = target_batch_normed - g0_batch_normed
                policy_loss_out = np.mean(np.sum(np.square(expected_actions_normed - actions_normed.detach().numpy()), axis=1), axis=0)

                print("|episode: {}| train step: {}| model_dynamics loss: {:.8f}| policy loss: {:.5f}".format(i, train_step_i, md_loss.detach().numpy(), policy_loss_out))



def main():
    speaker_fname = os.path.join(r'C:\Study\SpeechAcquisitionModel\src\VTL', 'JD2.speaker')
    lib_path = os.path.join(r'C:\Study\SpeechAcquisitionModel\src\VTL', 'VocalTractLab2.dll')
    ep_duration = 5000
    timestep = 20
    env = VTLEnv(lib_path, speaker_fname, timestep, max_episode_duration=ep_duration)

    settings = {
            'state_dim': env.state_dim,
            'action_dim': env.action_dim,
            'state_bound': env.state_bound,
            'action_bound': env.action_bound,
            'goal_dim': env.state_dim,
            'goal_bound': env.state_bound,
            'episode_length': 40,
            'minibatch_size': 512,
            'max_train_per_simulation': 50,
            'save_video_step': 200,

            'actor_tau': 0.01,
            'actor_learning_rate': 0.001,

            'model_dynamics_learning_rate': 0.05,

            'summary_dir': r'C:\Study\SpeechAcquisitionModel\reports\summaries',
            'videos_dir': r'C:\Study\SpeechAcquisitionModel\reports\videos'
        }

    replay_buffer = ReplayBuffer(100000)

    reference_fname = r'C:\Study\SpeechAcquisitionModel\src\VTL\references\a_o.pkl'
    with open(reference_fname, 'rb') as f:
        (tract_params, glottis_params) = pickle.load(f)
        target_trajectory = np.hstack((np.array(tract_params), np.array(glottis_params)))
    train(settings, env, replay_buffer, target_trajectory)
    return


if __name__ == '__main__':
    np.random.seed(1)
    tf.set_random_seed(1)
    torch.manual_seed(10)
    main()