import os
import time
import datetime
import numpy as np
import random
import pandas as pd
import pickle

from src.VTL.vtl_environment import VTLEnv
from src.speech_classification.audio_processing import AudioPreprocessor

import matplotlib.pyplot as plt

def create_reference(env, ep_duration, timestep, initial_state=None, end_state=None, directory=None, name=None):
    num_steps_per_ep = ep_duration // timestep
    action_space = env.number_glottis_parameters + env.number_vocal_tract_parameters

    state = env.reset(initial_state)
    if not end_state:
        raise ValueError("end state is None")
    states = []
    actions = []
    audios = []
    for i in range(num_steps_per_ep // 3):
        action = np.zeros(action_space)
        # action = (np.random.rand(action_space)) * 100.
        state, audio = env.step(action, True)
        states.append(state)
        actions.append(action)
        audios.append(audio)
        env.render()

    for i in range(num_steps_per_ep // 3):
        action = (np.asarray(end_state) - np.asarray(initial_state)) / (num_steps_per_ep // 3)
        state, audio = env.step(action, True)
        states.append(state)
        actions.append(action)
        audios.append(audio)
        env.render()

    for i in range(num_steps_per_ep // 3):
        action = np.zeros(action_space)
        # action = (np.random.rand(action_space)) * 100.
        state, audio = env.step(action, True)
        states.append(state)
        actions.append(action)
        audios.append(audio)
        env.render()

    actions = np.stack(actions)
    states = np.stack(states)
    audios = np.stack(audios)

    with open(os.path.join(directory, name+'.pkl'), 'wb') as f:
        pickle.dump({"audio": audios,
                     "tract_params": states[:, :24],
                     "glottis_params": states[:, 24:],
                     "action": actions}, f, protocol=0)
    return audios, states, actions


if __name__ == '__main__':
    speaker_fname = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'JD2.speaker')
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VocalTractLab2.dll')
    ep_duration = 3000
    timestep = 40
    num_steps_per_ep = ep_duration // timestep

    env = VTLEnv(lib_path, speaker_fname, timestep, max_episode_duration=ep_duration)

    name = 'a_i_2'
    initial_state = env.get_cf('a')
    end_state = env.get_cf('i')

    directory = 'references'

    audios, _, _ = create_reference(env, ep_duration, timestep, initial_state=initial_state, end_state=end_state, name=name, directory=directory)

    preproc_params = {
        "numcep": 12,
        "winlen": 0.04,
        "winstep": 0.04,
        "sample_rate": 22050
    }
    preproc = AudioPreprocessor(**preproc_params)

    res0 = preproc(audios.flatten(), preproc_params['sample_rate'])

    res1 = np.stack([preproc(audios[i], preproc_params['sample_rate']) for i in range(len(audios))]).squeeze()
    plt.figure()
    plt.imshow(res0.T)
    plt.figure()
    plt.imshow(res1.T)
    plt.figure()
    plt.plot(audios.flatten())
    plt.show()
    end = 0