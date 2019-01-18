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

    def add(self, s0, g0, a, s1, g1):
        if self.count < self.buffer_size:
            self.buffer.append((s0, g0, a, s1, g1))
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append((s0, g0, a, s1, g1))

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

        return s0_batch, g0_batch, a_batch, s1_batch, g1_batch

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

# specify sound transitions that will be encountered in training set
sound_names = ['a', 'e', 'i', 'o', 'u']
sound_names_cap_vowels = ['A', 'I', 'O', 'U', 'Y', '@']
sound_names_consonants = ['tt-dental-fric(i)', 'll-labial-nas(a)', 'll-labial-nas(i)',
                          'tt-alveolar-lat(a)', 'tt-postalveolar-fric(i)', 'tb-palatal-fric(a)']
sound_cfs = [get_cf(_) for _ in sound_names]

wanted_sound_transition = ['a_a', 'a_i', 'a_u', 'a_o', 'a_e',
                                'i_a', 'i_i', 'i_u', 'i_o', 'i_e',
                                'u_a', 'u_i', 'u_u', 'u_o', 'u_e',
                                'o_a', 'o_i', 'o_u', 'o_o', 'o_e',
                                'e_a', 'e_i', 'e_u', 'e_o', 'e_e']
# load lstm net for classification
lstm_net_fname = r'C:\Study\SpeechAcquisitionModel\reports\VTL_sigmoid_transition_classification\checkpoints\simple_lstm_08_29_2018_03_13_PM_acc_0.9961.pt'

lstm_model_settings = {
    'dct_coefficient_count': 12,
    'label_count': len(wanted_sound_transition) + 2,
    'hidden_reccurent_cells_count': 50,
    'winlen': 0.02,
    'winstep': 0.02
}

lstm_net = LstmNet(lstm_model_settings)
lstm_net.load_state_dict(torch.load(lstm_net_fname))

# instantiate environment and its properties
speaker_fname = os.path.join(r'C:\Study\SpeechAcquisitionModel\src\VTL', 'JD2.speaker')
lib_path = os.path.join(r'C:\Study\SpeechAcquisitionModel\src\VTL', 'VocalTractLab2.dll')
ep_duration = 5000
timestep = 20
episode_length = 40
env = VTLEnv(lib_path, speaker_fname, timestep, max_episode_duration=ep_duration)
win_len = int(timestep * env.audio_sampling_rate)
preproc = AudioPreprocessor(numcep=12, winlen=timestep/1000)
replay_buffer = ReplayBuffer(1000000)

s_dim = env.state_dim
a_dim = env.action_dim
# remember that lstm hidden state is a tuple h, c so we have to predict tuple (h, c)
g_dim = 2 * lstm_model_settings['hidden_reccurent_cells_count']

s_bound = env.state_bound
a_bound = env.action_bound
g_bound = [(-1., 1.) for _ in range(g_dim)]

action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(env.action_dim), sigma=0.01)

n_minibatch_size = 512

md_net = ModelDynamicsWithAudioGoalNet(s_dim, g_dim, a_dim)
md_optimizer = torch.optim.Adam(md_net.parameters()) #rms prop lr=0.001


num_episodes = 5000
# now at each step make 1 rollout, save it to a replay buffer and train model dynamics on a batch after each episode
for i in range(num_episodes):
    # pick 2 random sounds to make a transition
    sound1, cf1 = random.choice(list(zip(sound_names, sound_cfs)))
    sound2, cf2 = random.choice(list(zip(sound_names, sound_cfs)))

    sound1 = sound_names[0]
    cf1 = sound_cfs[0]
    sound2 = sound_names[1]
    cf2 = sound_cfs[1]

    # cf1 = random.choice(sound_cfs)
    # cf2 = random.choice(sound_cfs)

    s0 = env.reset(cf1)
    g0 = np.zeros(g_dim)

    # rollout episode
    for j in range(episode_length):
        k = 1. + np.exp(30. * (-1. * float(j) / episode_length + 0.5))
        action = np.subtract(cf2, cf1) / k - s0 + cf1

        noise = action_noise()
        noise[24:] = 0.
        action_noise_denormed = denormalize(noise, a_bound)

        action += action_noise_denormed

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
        if not isnans.any():
            replay_buffer.add(s0, g0, action, s1, g1)

        s0 = s1
        g0 = g1

        if j > 1:
            break
    adapt_minibatch = n_minibatch_size * 2 ** ((i * 5) // num_episodes)
    if replay_buffer.size() < adapt_minibatch:
        continue
    # sample from replay buffer and train model_dynamics
    # collect data
    s0_batch, g0_batch, a_batch, s1_batch, g1_batch = \
        replay_buffer.sample_batch(adapt_minibatch)

    # normalize data before train
    s0_batch_normed = normalize(s0_batch, s_bound)
    g0_batch_normed = normalize(g0_batch, g_bound)
    a_batch_normed = normalize(a_batch, a_bound)
    s1_batch_normed = normalize(s1_batch, s_bound)
    g1_batch_normed = normalize(g1_batch, g_bound)
    # zero grad
    md_optimizer.zero_grad()
    # prepare input
    x = torch.from_numpy(np.concatenate((s0_batch_normed, g0_batch_normed, a_batch_normed), axis=1)).float()
    y = torch.from_numpy(g1_batch_normed).float()

    pred = md_net(x)
    loss = torch.nn.MSELoss()(pred, y)
    loss.backward()

    md_optimizer.step()

    denormed_pred = torch.from_numpy(denormalize(pred.detach().numpy(), g_bound))
    denormed_y = torch.from_numpy(denormalize(y.detach().numpy(), g_bound))
    denormed_loss = torch.nn.MSELoss()(denormed_pred, denormed_y)
    print("|train step: {}| loss: {:.4f}| denormalized loss: {:.4f}".format(i,
                                                                            loss.detach().numpy(),
                                                                            denormed_loss.detach().numpy()))
    # show some examples
    if i % 200 == 0:
        for e in range(5):
            print("|Example:{}|\n"
                  "|Initial: {}|\n"
                  "|Expected: {}|\n"
                  "|Predicted: {}|\n"
                  "|Error: {}|".format(e,
                                       g0_batch[e],
                                       g1_batch[e],
                                       denormed_pred[e].detach().numpy(),
                                       abs(np.subtract(g1_batch[e], denormed_pred[e].detach().numpy()))
                                       ))


