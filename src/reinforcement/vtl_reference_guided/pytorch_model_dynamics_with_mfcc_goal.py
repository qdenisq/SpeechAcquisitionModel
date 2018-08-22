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
                     (s_dim + g_dim + a_dim) * 3,
                     (s_dim + g_dim + a_dim) * 6,
                     (s_dim + g_dim + a_dim) * 3,
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


speaker_fname = os.path.join(r'C:\Study\SpeechAcquisitionModel\src\VTL', 'JD2.speaker')
lib_path = os.path.join(r'C:\Study\SpeechAcquisitionModel\src\VTL', 'VocalTractLab2.dll')
ep_duration = 5000
timestep = 20
episode_length = 40
env = VTLEnv(lib_path, speaker_fname, timestep, max_episode_duration=ep_duration)
win_len = int(timestep * env.audio_sampling_rate)
preproc = AudioPreprocessor(numcep=26, winlen=timestep/1000)

dir_name = r'C:\Study\SpeechAcquisitionModel\data\raw\VTL_model_dynamics_simple_transition_08_16_2018_03_22_AM_48'
video_dir = dir_name + r'\Videos'
buffer_fname = dir_name + r'\replay_buffer.pkl'
with open(buffer_fname, mode='rb') as f:
    replay_buffer = pickle.load(f)


s_dim = env.state_dim
a_dim = env.action_dim
g_dim = preproc.get_dim()

s_bound = env.state_bound
a_bound = env.action_bound
# a_bound = [(p[0] / 5, p[1] / 5) for p in env.action_bound ]
g_bound = [(-20, 20) for _ in range(g_dim)]


# setup training parameters
num_train_steps = 50000
n_minibatch_size = 512

net = ModelDynamicsWithAudioGoalNet(s_dim, g_dim, a_dim)
optimizer = torch.optim.Adam(net.parameters()) #rms prop lr=0.001

# train loop
for i in range(num_train_steps):
    adapt_minibatch = n_minibatch_size * 2 ** ((i * 5) // num_train_steps)
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
    optimizer.zero_grad()
    # prepare input
    x = torch.from_numpy(np.concatenate((s0_batch_normed, g0_batch_normed, a_batch_normed), axis=1)).float()
    y = torch.from_numpy(g1_batch_normed).float()

    pred = net(x)
    loss = torch.nn.MSELoss()(pred, y)
    loss.backward()

    optimizer.step()

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


