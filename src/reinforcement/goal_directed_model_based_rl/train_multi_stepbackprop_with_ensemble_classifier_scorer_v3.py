import json
from pprint import pprint
import os
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import torch

from src.reinforcement.goal_directed_model_based_rl.model import SimpleStochasticActorCritic,\
    SimpleStochasticModelDynamics, SimpleDeterministicPolicy, SimpleDeterministicModelDynamics, EnsembleDeterministicModelDynamicsDeltaPredict
from src.reinforcement.goal_directed_model_based_rl.env import VTLEnvWithReferenceTransitionMaskedEntropyScore, VTLEnvWithReferenceTransitionMasked
from src.reinforcement.goal_directed_model_based_rl.algs.model_based_multi_step_backprop_with_ensemble_classifier_v3 import ModelBasedMultiStepBackPropWithEnsembleClassifierV3
from src.speech_classification.pytorch_conv_lstm import LstmNet, LstmNetEnsemble


def train(*args, **kwargs):
    print(kwargs)

    dt = str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p"))
    # writer = tf.summary.FileWriter(settings['summary_dir'] + '/summary_md_' + dt, sess.graph)
    video_dir = kwargs['mbbackprop']['videos_dir'] + '/video_ensemble_multi_step_V3_' + dt
    try:
        os.makedirs(video_dir)
    except:
        print("directory '{}' already exists")
    with open(video_dir + "/config.json", 'w') as json_file:
        json.dump(kwargs, json_file,  indent=4, separators=(',', ': '))



    device = 'cpu'
    kwargs['mbbackprop']['device'] = device

    speaker_fname = os.path.join(kwargs['env']['vtl_dir'], 'JD2.speaker')
    lib_path = os.path.join(kwargs['env']['vtl_dir'], 'VocalTractLab2.dll')

    env = VTLEnvWithReferenceTransitionMaskedEntropyScore(lib_path, speaker_fname, **kwargs['env'])
    env.reset(train_mode=True)

    state_dim = env.state_dim
    action_dim = env.action_dim
    kwargs['agent']['state_dim'] = state_dim
    kwargs['agent']['action_dim'] = action_dim

    agent = SimpleDeterministicPolicy(**kwargs['agent']).to(device)
    md = EnsembleDeterministicModelDynamicsDeltaPredict(**kwargs['model_dynamics']).to(device)
    alg = ModelBasedMultiStepBackPropWithEnsembleClassifierV3(agent=agent, model_dynamics=md, **kwargs['mbbackprop'])
    scores = alg.train(env, 5000, dir=video_dir)

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
    model_fname = "../models/mbbackprop_vtl_{}.pt".format(dt)
    torch.save(agent, model_fname)

    scores_fname = "../reports/mbbackprop_vtl_{}".format(dt)
    np.save(scores_fname, np.asarray(scores))

    plt.plot(scores)
    plt.plot(np.convolve(scores, np.ones(100)/100)[:200])
    fig_name = "../reports/mbppo_vtl_{}.png".format(dt)
    plt.savefig(fig_name)


if __name__ == '__main__':
    with open('train_multi_stepbackprop_mfcc_config_with_ensemble_classifierv3.json') as data_file:
        kwargs = json.load(data_file)
    pprint(kwargs)
    train(**kwargs)
