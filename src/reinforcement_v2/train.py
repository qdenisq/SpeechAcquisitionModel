import json
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
    env_args = []

    env_args.append(kwargs['env']['lib_path'])
    env_args.append(kwargs['env']['speaker_fname'])

    env_id = env_kwargs.pop('env_id')
    env = env_mgr.make(env_id, *env_args, **env_kwargs)

    for i in range(10):
        env.reset()
        k = 0
        while True:
            k += 1
            obs_dict, reward, done, info = env.step(np.tile(env.action_space.sample()*0.1, (kwargs['env']['num_workers'], 1)))
            env.render()
            print(f'{k} | r={reward.mean():.2f}')
            if np.any(done):
                # env.dump_episode()
                break



    kwargs['train'].update(kwargs['env'])
    # kwargs['train']['collect_data'] = kwargs['collect_data']
    # kwargs['train']['log_mode'] = kwargs['log_mode']
    #
    # if kwargs["init_data_fname"] is not None:
    #     kwargs['train']['init_data_fname'] = kwargs['init_data_fname']
    #
    # action_dim = env.action_space.shape[0]
    # state_dim = env.observation_space.shape[0]
    #
    # kwargs['soft_q_network']['state_dim'] = state_dim
    # kwargs['soft_q_network']['action_dim'] = action_dim
    #
    # kwargs['policy_network']['state_dim'] = state_dim
    # kwargs['policy_network']['action_dim'] = action_dim
    #
    # if kwargs['agent_fname'] is not None:
    #     # load agent
    #     agent_fname = kwargs['agent_fname']
    #     print(f'Loading agent from "{agent_fname}"')
    #     agent = torch.load(kwargs['agent_fname'])
    #     if not kwargs['use_alpha']:
    #         agent.noise_level = kwargs['noise_init_level']
    #         agent.noise = OUNoise(kwargs['soft_q_network']['action_dim'] * kwargs['env']['num_workers'],
    #                                  kwargs['env']['seed'])
    #     # to enable agent starting with custom (full) replay buffer
    #     if agent.replay_buffer_csv_filename is not None:
    #         agent.replay_buffer_csv_filename = os.path.splitext(agent.replay_buffer_csv_filename)[0] + "_new.csv"
    #         agent.replay_buffer = ReplayBuffer(agent.replay_buffer_size, agent.replay_buffer_csv_filename, None)
    # else:
    #     # create agent
    #     agent = AsyncrhonousSoftActorCritic(**kwargs)
    #
    # # train
    # agent.train(env, **kwargs['train'])
