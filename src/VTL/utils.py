import os
import time
import datetime
import numpy as np
import random
import pandas as pd
import pickle

from src.VTL.vtl_environment import VTLEnv


def random_rollout(env, ep_duration, timestep, initial_state=None):
    num_steps_per_ep = ep_duration // timestep
    action_space = env.number_glottis_parameters + env.number_vocal_tract_parameters

    state = env.reset(initial_state)
    states = [state]
    actions = []
    audios = []
    for step in range(num_steps_per_ep):
        action = (np.random.rand(action_space) - 0.5) * 0.2
        action[env.number_vocal_tract_parameters:] = 0.
        state, audio = env.step(action, True)
        states.append(state)
        actions.append(action)
        audios.append(audio)
        env.render()
    return audios, states, actions


def generate_random_rollout_dataset(env, num_rollouts, ep_duration, timestep, initial_states=None):
    base_dir = r'C:\Study\SpeechAcquisitionModel\data/raw/VTL_random_rollouts'
    dt = str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p_%S"))
    save_dir = os.path.join(base_dir, dt)
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except:
            pass
    fname = os.path.join(save_dir, dt) + '.pd'

    rollouts = pd.DataFrame(columns=['audio', 'states', 'actions'])
    print('Random rollout generation in directory"{}"'.format(save_dir))
    for i in range(num_rollouts):
        if initial_states is None:
            state = None
        else:
            idx = random.randint(0, len(initial_states)-1)
            state = initial_states[idx]
        audios, states, actions = random_rollout(env, ep_duration, timestep, state)
        ep_name = os.path.join(save_dir, str(i))
        env.dump_episode(ep_name)
        rollouts.loc[i] = [np.asarray(audios), np.asarray(states), np.asarray(actions)]

        if i % 500 == 0:
            rollouts.to_pickle(path=fname)
        print('\rProgress: {} out of {}'.format(i, num_rollouts), end='')
    rollouts.to_pickle(path=fname)


if __name__ == '__main__':
    speaker_fname = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'JD2.speaker')
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VocalTractLab2.dll')
    num_episodes = 10
    ep_duration = 1000
    timestep = 20
    num_steps_per_ep = ep_duration // timestep

    env = VTLEnv(lib_path, speaker_fname, timestep, max_episode_duration=ep_duration)

    soundnames = ['a', 'i', 'o', 'u']
    initial_states = [env.get_cf(s) for s in soundnames]

    generate_random_rollout_dataset(env, num_episodes, ep_duration, timestep, initial_states)