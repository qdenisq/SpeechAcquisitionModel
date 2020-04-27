import yaml
from pprint import pprint
import argparse
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
# from src.common.nn import *
# from src.common.noise import OUNoise
# from src.common.replay_buffer import ReplayBuffer
from src.reinforcement_v2.algo.backprop_md_softDTW import SequentialBackpropIntoPolicy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs trained agent')
    parser.add_argument('--config', default=r'C:\Study\SpeechAcquisitionModel\runs\ref_masked_dtw_we_vtl_backprop_04_27_2020_07_32_PM\md_backprop.yaml', help='config to build environment')
    parser.add_argument('--agent', default=r'C:\Study\SpeechAcquisitionModel\models\ref_masked_dtw_we_vtl_backprop_04_27_2020_07_32_PM\ref_masked_dtw_we_vtl_BackpropIntoPolicy_200.bp', help='path to the saved agent')

    args = parser.parse_args()

    with open(args.config, 'r') as data_file:
        kwargs = yaml.safe_load(data_file)
    pprint(kwargs)


    agent = torch.load(args.agent)
    agent.policy_net.eval()

    # create env
    env_mgr = EnvironmentManager()
    env_kwargs = copy.deepcopy(kwargs['env'])
    env_kwargs['num_workers'] = 1
    env_args = [kwargs['env']['lib_path'], kwargs['env']['speaker_fname']]

    env_id = env_kwargs.pop('env_id')
    env = env_mgr.make(env_id, *env_args, **env_kwargs)

    for i in range(2):

        state = env.reset()
        env.render()
        k = 0
        ref = env.get_attr('cur_reference')
        total_reward = 0.
        while True:
            # action, pi_mean, pi_log_std, log_prob = agent.policy_net.get_action(state)
            action = agent.policy_net.get_action(state)

            # action *= 0
            # action_noise = np.tile(env.action_space.sample(), (kwargs['env']['num_workers'], 1))
            # action = [ref[j]['action'][k, :] for j in range(kwargs['env']['num_workers'])]

            # action = np.stack(action) + action_noise * 0.
            # action = action_noise
            state, reward, done, info = env.step(action.detach().cpu().numpy())
            env.render()
            k += 1
            total_reward += reward.squeeze()
            print(f'{k} | score={total_reward:.2f}')
            if np.any(done):
                env.dump_episode()
                env.env_method('dump_reference')
                break