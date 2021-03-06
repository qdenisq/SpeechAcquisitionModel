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
    with open('configs/vtl_env_run_example.yaml', 'r') as data_file:
        kwargs = yaml.safe_load(data_file)
    pprint(kwargs)

    # create env
    env_mgr = EnvironmentManager()
    env_kwargs = copy.deepcopy(kwargs['env'])
    env_args = [kwargs['env']['lib_path'], kwargs['env']['speaker_fname']]

    env_id = env_kwargs.pop('env_id')
    env = env_mgr.make(env_id, *env_args, **env_kwargs)

    for i in range(10):
        env.reset()
        k = 0
        ref = env.get_attr('cur_reference')
        while True:
            action_noise = np.tile(env.action_space.sample(), (kwargs['env']['num_workers'], 1))
            # action = [ref[j]['action'][k, :] for j in range(kwargs['env']['num_workers'])]

            # action = np.stack(action) + action_noise * 0.
            action = action_noise
            obs_dict, reward, done, info = env.step(action)
            env.render()
            k += 1
            print(f'{k} | r={reward.mean():.2f}')
            if np.any(done):
                env.dump_episode()
                env.env_method('dump_reference')
                break