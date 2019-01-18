import random
import os
import pickle
import datetime
from collections import deque
from math import sqrt
from sklearn.metrics import mean_squared_error
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.VTL.vtl_environment import VTLEnv
from src.VTL.pyvtl_v2 import get_cf
from src.speech_classification.audio_processing import AudioPreprocessor

from src.speech_classification.pytorch_conv_lstm import LstmNet

class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=0):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s0, g0, a, s1, g1, t):
        if self.count < self.buffer_size:
            self.buffer.append((s0, g0, a, s1, g1, t))
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append((s0, g0, a, s1, g1, t))

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
        t_batch = np.array([_[5] for _ in batch])

        return s0_batch, g0_batch, a_batch, s1_batch, g1_batch, t_batch

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


class ModelDynamicsWithAudioGoalNet(torch.nn.Module):
    def __init__(self, s_dim, g_dim, a_dim):
        super(ModelDynamicsWithAudioGoalNet, self).__init__()
        # an affine operation: y = Wx + b
        num_units = [s_dim + g_dim + a_dim, #input
                     (s_dim + 2*g_dim + a_dim) * 3,
                     (s_dim + 2*g_dim + a_dim) * 6,
                     (s_dim + 2*g_dim + a_dim) * 3,
                     g_dim] #output
        self.__fc_layers = nn.ModuleList([nn.Linear(num_units[i-1], num_units[i]) for i in range(1, len(num_units))])
        [torch.nn.init.xavier_uniform_(layer.weight) for layer in self.__fc_layers]
        self.__batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(num_units[i]) for i in range(1, len(num_units))])

    def forward(self, x):
        for i in range(len(self.__fc_layers)-2):
            x = self.__fc_layers[i](x)
            # x = self.__batch_norm_layers[i](x)
            x = torch.nn.ReLU()(x)
        x = self.__fc_layers[-2](x)
        x = torch.nn.Tanh()(x)
        x = self.__fc_layers[-1](x)

        return x

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

        ds_1 = (self.__state_max - inp[:,
                                   :self.__s_dim] * self.__state_scaling_factor) / self.__action_scaling_factor
        x = torch.clamp(x - ds_1, max=0.)
        x = x + ds_1

        return x

def train(settings, env, replay_buffer, preproc, lstm_net, target_trajectory, reference_s0):
    # instantiate environment and its properties
    s_dim = settings['state_dim']
    a_dim = settings['action_dim']
    # remember that lstm hidden state is a tuple h, c so we have to predict tuple (h, c)
    g_dim = settings['goal_dim']

    s_bound = settings['state_bound']
    a_bound = settings['action_bound']
    g_bound = settings['goal_bound']

    episode_length = settings['episode_length']

    minibatch_size = settings['minibatch_size']
    save_step = settings['save_video_step']

    dt = str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p"))
    # writer = tf.summary.FileWriter(settings['summary_dir'] + '/summary_md_' + dt, sess.graph)
    video_dir = settings['videos_dir'] + '/video_md_' + dt
    try:
        os.makedirs(video_dir)
    except:
        pass

    action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(env.action_dim), sigma=0.01)

    n_minibatch_size = 512

    md_goal_net = ModelDynamicsWithAudioGoalNet(s_dim, g_dim, a_dim)
    md_goal_optimizer = torch.optim.RMSprop(md_goal_net.parameters(), lr=0.001) #rms prop lr=0.001

    policy_net = PolicyNet(s_dim, g_dim, a_dim, s_bound, a_bound)
    policy_optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=0.001)

    num_episodes = 5000
    train_step_i = 0
    miss_thresh = 0.2
    md_loss_thresh = 0.01
    # now at each step make 1 rollout, save it to a replay buffer and train model dynamics on a batch after each episode
    for i in range(num_episodes):
        md_goal_net.eval()
        policy_net.eval()


        s0 = env.reset(reference_s0)
        g0 = np.zeros(g_dim)

        # rollout episode
        for j in range(episode_length):
            target = target_trajectory[j + 1]
            # normalize data before feeding it to policy
            s0_normed = normalize(np.reshape(s0, (1, s_dim)), s_bound)
            g0_normed = normalize(np.reshape(g0, (1, g_dim)), g_bound)
            target_normed = normalize(np.reshape(target, (1, g_dim)), g_bound)
            policy_X = torch.from_numpy(np.concatenate((s0_normed, g0_normed, target_normed), axis=1)).float()

            action_normed = policy_net(policy_X).detach().numpy()
            # add noise
            a_noise = action_noise()
            a_noise[24:] = 0.
            action_normed += a_noise

            # denormalize action
            action = denormalize(action_normed, a_bound)
            action = np.reshape(action, (a_dim))

            # make a step
            s1, audio = env.step(action)
            # get audio
            wav_audio = np.int16(audio * (2 ** 15 - 1))
            # get mfcc
            mfcc = preproc(wav_audio, env.audio_sampling_rate)
            isnans = np.isnan(mfcc)
            if isnans.any():
                print(mfcc)
                print("NAN OCCURRED")
            mfcc = torch.from_numpy(np.expand_dims(mfcc, axis=0)).float()
            # get hidden state from lstm net
            hidden = None
            pred, hidden, lstm_out = lstm_net(mfcc, np.array([1]), hidden=hidden)
            g1 = np.concatenate([hidden[0].detach().numpy().flatten(), hidden[0].detach().numpy().flatten()])
            env.render()

            # add to replay buffer
            if not isnans.any() and i % save_step != 0:
                replay_buffer.add(s0, g0, action, s1, g1, target)

            s0 = s1
            g0 = g1

            tt = torch.from_numpy(target.flatten()[np.newaxis, :]).float()
            tg1 = torch.from_numpy(g1.flatten()[np.newaxis, :]).float()
            miss = torch.nn.MSELoss()(tt, tg1).detach().numpy()
            if miss > miss_thresh and (i % save_step != 0 or replay_buffer.size() < minibatch_size) and j > 1:
                break
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
                md_goal_net.train()
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

                md_loss = torch.nn.MSELoss()(md_pred, md_target_Y)
                md_loss.backward()
                md_goal_optimizer.step()

                # calc denormalize loss

                denormed_pred = torch.from_numpy(denormalize(md_pred.detach().numpy(), g_bound))
                denormed_y = torch.from_numpy(denormalize(md_target_Y.detach().numpy(), g_bound))
                md_loss_denormed = torch.nn.MSELoss()(denormed_pred, denormed_y)

                policy_loss_out = 0.
                if md_loss.detach() < md_loss_thresh:

                    #############################################################
                    # train policy
                    ##############################################################
                    policy_net.train()
                    md_goal_net.eval()
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
                    loss = torch.nn.MSELoss()(md_pred, md_target_Y)

                    loss.backward()
                    # clip gradients
                    torch.nn.utils.clip_grad_norm(policy_net.parameters(), 0.001)

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
    preproc = AudioPreprocessor(numcep=12, winlen=timestep / 1000, winstep=timestep / 1000)
    # load lstm net for classification
    lstm_net_fname = r'C:\Study\SpeechAcquisitionModel\reports\VTL_sigmoid_transition_classification\checkpoints\simple_lstm_08_29_2018_03_13_PM_acc_0.9961.pt'
    lstm_net_classes = 25
    lstm_model_settings = {
        'dct_coefficient_count': 12,
        'label_count': lstm_net_classes + 2,
        'hidden_reccurent_cells_count': 50,
        'winlen': 0.02,
        'winstep': 0.02
    }

    lstm_net = LstmNet(lstm_model_settings)
    lstm_net.load_state_dict(torch.load(lstm_net_fname))

    settings = {
            'state_dim': env.state_dim,
            'action_dim': env.action_dim,
            'state_bound': env.state_bound,
            'action_bound': [(p[0] / 5, p[1] / 5) for p in env.action_bound ], #env.action_bound,
            'goal_dim': lstm_model_settings['hidden_reccurent_cells_count']*2,
            'goal_bound': [(-1., 1.) for _ in range(lstm_model_settings['hidden_reccurent_cells_count']*2)],
            'episode_length': 40,
            'minibatch_size': 512,
            'max_train_per_simulation': 50,
            'save_video_step': 200,

            'summary_dir': r'C:\Study\SpeechAcquisitionModel\reports\summaries',
            'videos_dir': r'C:\Study\SpeechAcquisitionModel\reports\videos'
        }

    replay_buffer = ReplayBuffer(100000)

    # load target sound
    reference_wav_fname = r'C:\Study\SpeechAcquisitionModel\data\raw\VTL_model_dynamics_sigmoid_transition_08_28_2018_03_57_PM_03\Videos\a_i\episode_08_28_2018_03_57_PM_06.wav'
    reference_s0 = get_cf('a')
    reference_mfcc = preproc(reference_wav_fname)
    # feed target sound to lstm net and  get target goal from hidden state

    hidden = None
    target_trajectory = []
    for i in range(reference_mfcc.shape[0]):
        net_input = torch.from_numpy(np.reshape(reference_mfcc[i, :], (1, 1, reference_mfcc.shape[1]))).float()
        _, hidden, _ = lstm_net(net_input, np.array([1]), hidden)
        t = np.concatenate([hidden[0].detach().numpy().flatten(), hidden[0].detach().numpy().flatten()])
        target_trajectory.append(t)

    train(settings, env, replay_buffer, preproc, lstm_net, target_trajectory, reference_s0)
    return


if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(10)
    main()