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

from src.VTL.pyvtl_v2 import get_cf

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


def generate_model_dynamics_training_data_random_policy(env, preproc, replay_buffer, num_samples, episode_length,
                                                        video_dir=None):
    num_episodes = num_samples // episode_length
    s_dim = env.state_dim
    a_dim = env.action_dim
    g_dim = preproc.get_dim()

    s_bound = env.state_bound
    a_bound = env.action_bound
    g_bound = [(-100, 100) for _ in range(g_dim)]

    action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(env.action_dim), sigma=0.01)

    for i in range(num_episodes):
        s0 = env.reset()
        g0 = np.zeros(g_dim)
        r = []
        # rollout episode
        for j in range(episode_length):
            action_normed = action_noise()

            # denormalize action
            action = denormalize(action_normed, a_bound)
            action = np.reshape(action, (a_dim))
            action[24:] = 0.
            # make a step
            s1, audio = env.step(action)
            wav_audio = np.int16(audio * (2 ** 15 - 1))
            mfcc = preproc(wav_audio, env.audio_sampling_rate)
            g1 = np.reshape(mfcc, (g_dim))
            env.render()

            # calc reward
            replay_buffer.add(s0, g0, action, s1, g1)

            s0 = s1
            g0 = g1

        if video_dir is not None:
            fname = video_dir + '/episode_' + str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p_%S"))
            env.dump_episode(fname)
        print('|episode: {} out of {}|'.format(i, num_episodes))


def generate_model_dynamics_training_data_simple_transition(env, preproc, replay_buffer, num_samples, episode_length,
                                                            video_dir=None):
    num_episodes = num_samples // episode_length
    s_dim = env.state_dim
    a_dim = env.action_dim
    g_dim = preproc.get_dim()

    s_bound = env.state_bound
    a_bound = env.action_bound
    g_bound = [(-50, 50) for _ in range(g_dim)]

    action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(env.action_dim), sigma=0.001)

    sound_names = ['a', 'e', 'i', 'o', 'u', 'A', 'I', 'O', 'U', 'Y', '@',
                   'tt-dental-fric(i)', 'll-labial-nas(a)', 'll-labial-nas(i)',
                   'tt-alveolar-lat(a)', 'tt-postalveolar-fric(i)', 'tb-palatal-fric(a)']
    sound_cfs = [get_cf(_) for _ in sound_names]

    for i in range(num_episodes):
        # pick 2 random sounds to make a transition
        cf1 = random.choice(sound_cfs)
        cf2 = random.choice(sound_cfs)

        s0 = env.reset(cf1)
        g0 = np.zeros(g_dim)
        # rollout episode
        for j in range(episode_length):

            action = np.subtract(cf2, cf1) / episode_length

            noise = action_noise()
            action_noise_denormed = denormalize(noise, a_bound)

            action += action_noise_denormed

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

            # add to replay buffer
            # avoid nans from VTL(bug) and also skip first step because of dubious g0= 0
            if not isnans.any() and j>0:
                replay_buffer.add(s0, g0, action, s1, g1)

            s0 = s1
            g0 = g1

        if video_dir is not None:
            fname = video_dir + '/episode_' + str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p_%S"))
            env.dump_episode(fname)

        print('|episode: {} out of {}|'.format(i, num_episodes))


def generate(mode='random'):
    speaker_fname = os.path.join(r'C:\Study\SpeechAcquisitionModel\src\VTL', 'JD2.speaker')
    lib_path = os.path.join(r'C:\Study\SpeechAcquisitionModel\src\VTL', 'VocalTractLab2.dll')
    ep_duration = 5000
    timestep = 20
    episode_length = 40
    env = VTLEnv(lib_path, speaker_fname, timestep, max_episode_duration=ep_duration)
    win_len = int(timestep * env.audio_sampling_rate)
    preproc = AudioPreprocessor(numcep=13, winlen=timestep/1000)
    replay_buffer = ReplayBuffer(1000000)

    num_samples = 50000

    dt = str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p_%S"))
    video_dir = r'C:\Study\SpeechAcquisitionModel\data\raw\VTL_model_dynamics_' + mode + '_' + dt + r'\Videos'
    buffer_fname = r'C:\Study\SpeechAcquisitionModel\data\raw\VTL_model_dynamics_' + mode + '_' + dt + r'\replay_buffer.pkl'
    os.makedirs(video_dir, exist_ok=True)

    if mode == 'random':
        generate_model_dynamics_training_data_random_policy(env, preproc, replay_buffer, num_samples, episode_length,
                                                            video_dir=video_dir)
    elif mode == 'simple_transition':
        generate_model_dynamics_training_data_simple_transition(env, preproc, replay_buffer, num_samples,
                                                                episode_length, video_dir=video_dir)
    with open(buffer_fname, mode='wb') as f:
        pickle.dump(replay_buffer, f)

if __name__ == '__main__':
    generate(mode='simple_transition')