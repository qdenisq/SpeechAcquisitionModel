import yaml
from pprint import pprint
import datetime
import copy
import os
from collections import defaultdict
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from src.reinforcement_v2.envs.env import EnvironmentManager

from src.reinforcement_v2.utils.timer import Timer
# from src.reinforcement_v2.common.tensorboard import DoubleSummaryWriter
# from src.common.nn import SoftQNetwork, PolicyNetwork
# from src.common.noise import OUNoise
# from src.common.replay_buffer import ReplayBuffer
from src.siamese_net_sound_similarity.train_v2 import SiameseDeepLSTMNet

if __name__ == '__main__':
    # Generate rollouts with action pattern a = gamma * a_true + (1- gamma) * noise with varying gamma
    with open('DTW_env_test.yaml', 'r') as data_file:
        kwargs = yaml.safe_load(data_file)
    pprint(kwargs)

    # create env
    env_mgr = EnvironmentManager()
    env_kwargs = copy.deepcopy(kwargs['env'])
    env_args = [kwargs['env']['lib_path'], kwargs['env']['speaker_fname']]

    env_id = env_kwargs.pop('env_id')
    env = env_mgr.make(env_id, *env_args, **env_kwargs)

    gammas = np.linspace(1, 0, 10)
    print(gammas)

    num_rollouts_per_gamma = 100

    for i, gamma in enumerate(gammas):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'gamma_{gamma:.2f}')
        try:
            os.makedirs(path)
        except:
            pass

        num_rollouts = num_rollouts_per_gamma // kwargs['env']['num_workers']
        for j in range(num_rollouts):
            env.reset()
            k = 0
            ref = env.get_attr('cur_reference')
            while True:
                action_noise = np.tile(env.action_space.sample(), (kwargs['env']['num_workers'], 1))
                action = [ref[m]['action'][k, :] for m in range(kwargs['env']['num_workers'])]
                action = gamma * np.stack(action) + (1 - gamma) * action_noise
                obs_dict, reward, done, info = env.step(action)
                env.render()
                k += 1
                print(f'{k} | r={reward.mean():.2f}')
                if np.any(done):
                    dt = str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p_%S"))

                    fname = os.path.join(path, dt)
                    ref_fname = os.path.join(path, 'ref_'+dt)
                    env.dump_episode(fname=fname)
                    env.env_method('dump_reference', fname=ref_fname)
                    break
