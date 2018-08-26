import random
import datetime
import os
import pickle
from collections import deque
import numpy as np

import torch as torch
import torch.nn as nn
import torch.nn.functional as F

from src.VTL.vtl_environment import VTLEnv
from src.speech_classification.audio_processing import AudioPreprocessor


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

################################################################
# Model Dynamics
################################################################


class ModelDynamicsWithAudioGoalNet(torch.nn.Module):
    def __init__(self, s_dim, g_dim, a_dim):
        super(ModelDynamicsWithAudioGoalNet, self).__init__()
        # an affine operation: y = Wx + b
        num_units = [s_dim + g_dim + a_dim,  # input
                     (s_dim + g_dim + a_dim) * 3,
                     (s_dim + g_dim + a_dim) * 6,
                     (s_dim + g_dim + a_dim) * 3,
                     g_dim]  # output
        self.__fc_layers = nn.ModuleList(
            [nn.Linear(num_units[i - 1], num_units[i]) for i in range(1, len(num_units))])
        [torch.nn.init.xavier_uniform_(layer.weight) for layer in self.__fc_layers]
        self.__batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(num_units[i]) for i in range(1, len(num_units))])

    def forward(self, x):
        for i in range(len(self.__fc_layers) - 2):
            x = self.__fc_layers[i](x)
            # x = self.__batch_norm_layers[i](x)
            x = torch.nn.ReLU()(x)
        x = self.__fc_layers[-2](x)
        x = torch.nn.Tanh()(x)
        x = self.__fc_layers[-1](x)
        return x


class ModelDynamicsNet(nn.Module):
    def __init__(self, s_dim, g_dim, a_dim):
        super(ModelDynamicsNet, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear((s_dim + g_dim + a_dim), (s_dim + g_dim + a_dim) * 2)
        torch.nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear((s_dim + g_dim + a_dim) * 2, s_dim * 4)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(s_dim * 4, s_dim)
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


################################################################
# Policy
################################################################


class PolicyNet(nn.Module):
    def __init__(self, s_dim, g_dim, a_dim, s_bound, a_bound):
        super(PolicyNet, self).__init__()
        # an affine operation: y = Wx + b
        self.__s_dim = s_dim
        num_units = [s_dim + g_dim + g_dim,  # input
                     (s_dim + g_dim + g_dim) * 3,
                     (s_dim + g_dim + g_dim) * 6,
                     (s_dim + g_dim + g_dim) * 3,
                     a_dim]  # output
        self.__fc_layers = nn.ModuleList(
            [nn.Linear(num_units[i - 1], num_units[i]) for i in range(1, len(num_units))])
        # [torch.nn.init.xavier_uniform_(layer.weight) for layer in self.__fc_layers]
        self.__batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(num_units[i]) for i in range(1, len(num_units))])

    #     test truncating actions inside policy so that state remains within bounds
        largest_state = np.array([max(abs(y[0]), abs(y[1])) for y in s_bound])
        state_min, state_max = zip(*s_bound)
        largest_action = np.array([max(abs(y[0]), abs(y[1])) for y in a_bound])

        self.__state_scaling_factor = torch.from_numpy(largest_state).float()
        self.__action_scaling_factor = torch.from_numpy(largest_action).float()
        self.__state_max = torch.tensor(np.reshape(state_max, (1, -1)), requires_grad=False).float()
        self.__state_min = torch.tensor(np.reshape(state_min, (1, -1)), requires_grad=False).float()

    def forward(self, x):
        inp = x
        for i in range(len(self.__fc_layers) - 2):
            x = self.__fc_layers[i](x)
            # x = self.__batch_norm_layers[i](x)
            x = torch.nn.ReLU()(x)
        x = self.__fc_layers[-2](x)
        x = torch.nn.Tanh()(x)
        x = self.__fc_layers[-1](x)

        # test truncate here
        ds = (self.__state_min - inp[:, :self.__s_dim] * self.__state_scaling_factor) / self.__action_scaling_factor
        x = torch.clamp(x - ds, min=0.)
        x = x + ds

        ds_1 = (self.__state_max - inp[:, :self.__s_dim] * self.__state_scaling_factor) / self.__action_scaling_factor
        x = torch.clamp(x - ds_1, max=0.)
        x = x + ds_1

        return x


def train(settings, env, replay_buffer, preproc, reference_trajectory):

    # setup all summaries and dump directories

    num_episodes = 50000
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

    md_goal_net = ModelDynamicsWithAudioGoalNet(s_dim, g_dim, a_dim)
    md_goal_criterion = nn.MSELoss()
    md_goal_optimizer = torch.optim.Adam(md_goal_net.parameters())

    # md_state_net = ModelDynamicsNet(s_dim, g_dim, a_dim)
    # md_state_criterion = nn.MSELoss()
    # md_state_optimizer = torch.optim.Adam(md_state_net.parameters())

    policy_net = PolicyNet(s_dim, g_dim, a_dim, s_bound, a_bound)
    policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.00001)

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

    (actions_target_traj, goal_target_traj, state_target_traj) = reference_trajectory

    for i in range(num_episodes):
        # pick random initial state from the reference trajectory
        # s0_index = randrange(0, reference_trajectory.shape[0] - 1)
        # if i % save_step == 0:
        #     s0_index = 0
        # for nwo always start with 0 step
        s0_index = 0
        s0 = state_target_traj[s0_index]
        g0 = goal_target_traj[s0_index]
        env.reset(s0)
        env.render()

        # rollout episode
        for j in range(s0_index, len(goal_target_traj) - 1):
            target = goal_target_traj[j + 1]
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
            s1, audio = env.step(action)

            wav_audio = np.int16(audio * (2 ** 15 - 1))
            mfcc = preproc(wav_audio, env.audio_sampling_rate)
            isnans = np.isnan(mfcc)
            if isnans.any():
                print(mfcc)
                print("NAN OCCURED")
            g1 = np.reshape(mfcc, (g_dim))
            env.render()

            g1_normed = normalize(g1, g_bound)
            tar_normed = normalize(target, g_bound)
            miss = abs(tar_normed - g1_normed)
            miss_thresh = 0.3
            s1_norm = normalize(s1, s_bound)
            out_of_bound = [abs(s1[k]-s_bound[k][1]) + abs(s1[k]-s_bound[k][0]) > 1.01 * abs(s_bound[k][0]-s_bound[k][1]) for k in range(s_dim)]
            mean_miss = np.mean(miss)
            # add to replay buffer
            # avoid nans from VTL(bug) and also skip first step because of dubious g0= 0
            # if not isnans.any() and j > 0 and np.mean(miss) < miss_thresh and not any(out_of_bound):
            if i % save_step != 0:
                replay_buffer.add(s0, g0, action, s1, g1, target)

            s0 = s1
            g0 = g1

            # some sort of early stopping
            # Need to change for sure in order to avoid going out of bounds at least
            if (np.mean(miss) > miss_thresh or any(out_of_bound)) and (i % save_step != 0 or replay_buffer.size() < minibatch_size):
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
                md_goal_optimizer.zero_grad()
                # train model dynamics
                md_input_X = np.concatenate((s0_batch_normed,
                                             g0_batch_normed,
                                             a_batch_normed), axis=1)
                md_input_X = torch.from_numpy(md_input_X).float()

                md_pred = md_goal_net(md_input_X)

                md_target_Y = torch.from_numpy(g1_batch_normed).float()

                md_loss = md_goal_criterion(md_pred, md_target_Y)
                md_loss.backward()
                md_goal_optimizer.step()

                # calc denormalize loss

                denormed_pred = torch.from_numpy(denormalize(md_pred.detach().numpy(), g_bound))
                denormed_y = torch.from_numpy(denormalize(md_target_Y.detach().numpy(), g_bound))
                md_loss_denormed = torch.nn.MSELoss()(denormed_pred, denormed_y)

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

                md_target_Y = torch.from_numpy(target_batch_normed).float()

                md_pred = md_goal_net(md_input_X)
                loss = md_goal_criterion(md_pred, md_target_Y)
                #
                # action_grads = torch.autograd.grad(loss, actions_normed, allow_unused=True)
                #
                # torch.autograd.backward([actions_normed], [action_grads[0]])
                loss.backward()
                policy_optimizer.step()

                # note that policy loss here is calculated in accordance with the model dynamics
                policy_loss_out = loss.detach().numpy()

                print("|episode: {}| train step: {}| model_dynamics loss: {:.8f}| model_dynamics denormed loss: {:.8f}| policy loss: {:.5f}"
                      .format(i, train_step_i, md_loss.detach().numpy(), md_loss_denormed.detach().numpy(), policy_loss_out))



def main():
    speaker_fname = os.path.join(r'C:\Study\SpeechAcquisitionModel\src\VTL', 'JD2.speaker')
    lib_path = os.path.join(r'C:\Study\SpeechAcquisitionModel\src\VTL', 'VocalTractLab2.dll')
    ep_duration = 5000
    timestep = 20
    env = VTLEnv(lib_path, speaker_fname, timestep, max_episode_duration=ep_duration)
    preproc = AudioPreprocessor(numcep=13, winlen=timestep / 1000)
    settings = {
            'state_dim': env.state_dim,
            'action_dim': env.action_dim,
            'state_bound': env.state_bound,
            'action_bound': [(p[0] / 5, p[1] / 5) for p in env.action_bound ], #env.action_bound,
            'goal_dim': preproc.get_dim(),
            'goal_bound': [(-50, 50) for _ in range(preproc.get_dim())],
            'episode_length': 40,
            'minibatch_size': 512,
            'max_train_per_simulation': 50,
            'save_video_step': 200,

            'actor_tau': 0.01,
            'actor_learning_rate': 0.000001,

            'model_dynamics_learning_rate': 0.05,

            'summary_dir': r'C:\Study\SpeechAcquisitionModel\reports\summaries',
            'videos_dir': r'C:\Study\SpeechAcquisitionModel\reports\videos'
        }

    replay_buffer = ReplayBuffer(100000)

    reference_fname = r'C:\Study\SpeechAcquisitionModel\src\VTL\references\a_i.pkl'
    with open(reference_fname, 'rb') as f:
        (tract_params, glottis_params) = pickle.load(f)
        target_trajectory = np.hstack((np.array(tract_params), np.array(glottis_params)))
    # generate audio and then goal target trajectpry based on given state space target trajectory
    s0 = env.reset(target_trajectory[0])
    g0 = np.zeros(preproc.get_dim())
    target_actions = []
    target_goals = []
    target_goals.append(g0)
    target_states = []
    target_states.append(s0)

    for i in range(1, len(target_trajectory) - 1):
        action = np.subtract(target_trajectory[i], s0)
        s1, audio = env.step(action)
        wav_audio = np.int16(audio * (2 ** 15 - 1))
        mfcc = preproc(wav_audio, env.audio_sampling_rate)
        isnans = np.isnan(mfcc)
        if isnans.any():
            print(mfcc)
            print("NAN OCCURED")
            raise TypeError("NAN in target")
        g1 = np.reshape(mfcc, (preproc.get_dim()))

        target_actions.append(action)
        target_goals.append(g1)
        target_states.append(s1)



        s0 = s1
        g0 = g1

    target_goals[1] = target_goals[2]
    target_trajectory = (target_actions, target_goals, target_states)
    train(settings, env, replay_buffer, preproc, target_trajectory)
    return


if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(10)
    main()