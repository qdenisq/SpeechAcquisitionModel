import json
from pprint import pprint
import os
import numpy as np
import pandas as pd

import torch
from torch.nn import Linear, LSTM, Tanh, ReLU, Module, MSELoss
from torchvision import transforms

from src.speech_classification.audio_processing import AudioPreprocessor
from src.speech_classification.pytorch_conv_lstm import LstmNet
from src.reinforcement.goal_directed_model_based_rl.model import StochasticLstmModelDynamics, SimpleStochasticPolicy
from src.reinforcement.goal_directed_model_based_rl.replay_buffer import ReplayBuffer
from src.VTL.vtl_environment import VTLEnv


def train(*args, **kwargs):
    print(kwargs)

    torch.random.manual_seed(0)


    device = kwargs['train']['device']

    # 1. Init audio preprocessing
    preproc = AudioPreprocessor(**kwargs['preprocessing_params'])
    sr = kwargs['preprocessing_params']['sample_rate']

    # 2. Load preprocessing net
    preproc_net = torch.load(kwargs['preproc_net_fname']).to(device)

    # 3. Init model dynamics net
    md_net = StochasticLstmModelDynamics(**kwargs['model_dynamics_params']).to(device)
    optim = torch.optim.Adam(md_net.parameters(), lr=kwargs['train']['learning_rate'], eps=kwargs['train']['learning_rate_eps'])

    # 4. Init Policy
    policy = SimpleStochasticPolicy(**kwargs['policy_params']).to(device)

    # 5. Init environment
    speaker_fname = os.path.join(kwargs['vtl_dir'], 'JD2.speaker')
    lib_path = os.path.join(kwargs['vtl_dir'], 'VocalTractLab2.dll')
    num_episodes = 10
    ep_duration = 1000
    timestep = 20
    num_steps_per_ep = ep_duration // timestep

    env = VTLEnv(lib_path, speaker_fname, timestep, max_episode_duration=ep_duration)

    # 6. Load reference for policy
    reference_wav_fname = kwargs['reference_fname']
    reference_preproc = torch.from_numpy(preproc(reference_wav_fname)[np.newaxis]).float().to(device)
    _, _, reference = preproc_net(reference_preproc, seq_lens=np.array([reference_preproc.shape[1]]))
    reference = reference.detach().cpu().numpy().squeeze()

    # 7. Init replay buffer
    replay_buffer = ReplayBuffer(kwargs['buffer_size'])

    # 8. Train loop
    params = kwargs['train']
    policy.eval()
    md_net.train()
    for i in range(params['num_steps']):

        state = env.reset()
        goal_state = np.zeros(kwargs['model_dynamics_params']['goal_dim'])
        states = [state]
        actions = []
        goal_states = [goal_state]
        hidden = None
        for step in range(num_steps_per_ep):
            policy_input = np.concatenate((state, reference[step, :]))[np.newaxis]
            policy_input = torch.from_numpy(policy_input).float().to(device)
            action, _, _ = policy(policy_input)
            # action = (np.random.rand(action_space)) * 100.
            action = action.detach().cpu().numpy().squeeze()
            action[env.number_vocal_tract_parameters:] = 0.
            action = action * 0.1 # reduce amplitude for now
            new_state, audio = env.step(action, True)

            preproc_audio = preproc(audio, sr)[np.newaxis]
            preproc_audio = torch.from_numpy(preproc_audio).float().to(device)
            _, hidden, new_goal_state = preproc_net(preproc_audio, seq_lens=np.array([preproc_audio.shape[1]]), hidden=hidden)
            new_goal_state = new_goal_state.detach().cpu().numpy().squeeze()

            states.append(new_state)
            goal_states.append(new_goal_state)
            actions.append(action)

            state = new_state
            goal_state = new_goal_state

            env.render()

        replay_buffer.add(states, goal_states, actions, None, None)

        minibatch_size = kwargs['train']['minibatch_size']
        if replay_buffer.size() > minibatch_size:
            num_updates_per_epoch = kwargs['train']['updates_per_episode']
            for k in range(num_updates_per_epoch):
                # sample minibatch
                s0, g0, a, _, _ = replay_buffer.sample_batch(minibatch_size)

                # train
                seq_len = a.shape[1]
                goal_dim = kwargs['model_dynamics_params']["goal_dim"]

                s = torch.from_numpy(s0).float().to(device)
                g = torch.from_numpy(g0).float().to(device)
                a = torch.from_numpy(a).float().to(device)

                # forward prop
                s_pred, g_pred, s_prob, g_prob = md_net(s[:, :-1, :], g[:, :-1, :], a)

                # compute error
                loss = MSELoss(reduction='sum')(g_pred, g[:, 1:, :]) / (seq_len * kwargs['train']['minibatch_size'])

                # backprop
                optim.zero_grad()
                loss.backward()
                optim.step()

                dynamics = MSELoss(reduction='sum')(g[:, 1:, :], g[:, :-1, :]) / (seq_len * kwargs['train']['minibatch_size'])

            print("\rstep: {} | loss: {:.4f}| actual_dynamics: {:.4f}".format(i, loss.detach().cpu().item(), dynamics.detach().cpu().item()), end="")


if __name__ == '__main__':
    with open('rnn_md_config.json') as data_file:
        data = json.load(data_file)

    pprint(data)
    kwargs = data

    train(**kwargs)