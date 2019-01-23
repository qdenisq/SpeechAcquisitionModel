import json
from pprint import pprint
import os
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import torch

from src.reinforcement.goal_directed_model_based_rl.model import SimpleStochasticActorCritic
from src.reinforcement.goal_directed_model_based_rl.env import VTLEnvWithReferenceTransitionMasked
from src.reinforcement.goal_directed_model_based_rl.algs.ppo import PPO
from src.speech_classification.pytorch_conv_lstm import LstmNet



def train(*args, **kwargs):
    print(kwargs)

    device = 'cpu'
    kwargs['ppo']['device'] = device

    speaker_fname = os.path.join(kwargs['env']['vtl_dir'], 'JD2.speaker')
    lib_path = os.path.join(kwargs['env']['vtl_dir'], 'VocalTractLab2.dll')

    env = VTLEnvWithReferenceTransitionMasked(lib_path, speaker_fname, **kwargs['env'])
    env.reset(train_mode=True)

    state_dim = env.state_dim
    action_dim = env.action_dim
    kwargs['agent']['state_dim'] = state_dim
    kwargs['agent']['action_dim'] = action_dim

    agent = SimpleStochasticActorCritic(**kwargs['agent']).to(device)
    alg = PPO(agent=agent, **kwargs['ppo'])
    scores = alg.train(env, 200)

    agent.eval()
    for i in range(1):
        done = False
        state = env.reset(train_mode=True)
        rewards = []
        while not np.any(done):
            action, _, _, _ = agent(torch.from_numpy(state).float().to(device))
            action = action.detach().cpu().numpy()
            action = np.clip(action, -1, 1)
            env.step(action)
            state, reward, done = env.step(action)
            rewards.append(reward)
            score = np.mean(np.asarray(rewards).sum(axis=0))
            print("\r play #{} | score: {}".format(i + 1, score), end='')

    dt = str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p"))
    model_fname = "../models/ppo_reacher_{}.pt".format(dt)
    torch.save(agent, model_fname)

    scores_fname = "../reports/ppo_reacher_{}".format(dt)
    np.save(scores_fname, np.asarray(scores))

    plt.plot(scores)
    plt.plot(np.convolve(scores, np.ones(100)/100)[:200])
    fig_name = "../reports/ppo_reacher_{}.png".format(dt)
    plt.savefig(fig_name)


if __name__ == '__main__':
    with open('train_ppo_config_tract_goal_only.json') as data_file:
        kwargs = json.load(data_file)
    pprint(kwargs)
    train(**kwargs)