import os
import time
import datetime
import numpy as np
import random
import pandas as pd
import pickle

from src.VTL.vtl_environment import VTLEnv
from src.speech_classification.audio_processing import AudioPreprocessorFbank
from python_speech_features.sigproc import framesig
import librosa

import matplotlib.pyplot as plt


def create_reference(env,
                     ep_duration,
                     timestep,
                     initial_sound_name=None,
                     end_sound_name=None,
                     initial_state=None,
                     end_state=None,
                     state_sigma=0.01,
                     action_sigma=0.01,
                     time_shift_max=4,
                     directory=None,
                     name=None):
    num_steps_per_ep = ep_duration // timestep
    action_space = env.number_glottis_parameters + env.number_vocal_tract_parameters

    init_state_sampled = np.random.normal(env.normalize(initial_state, env.state_bound), state_sigma, len(initial_state))
    init_state_sampled = env.denormalize(init_state_sampled, env.state_bound)
    init_state_sampled[24:] = initial_state[24:]

    end_state_sampled = np.random.normal(env.normalize(end_state, env.state_bound), state_sigma, len(initial_state))
    end_state_sampled = env.denormalize(end_state_sampled, env.state_bound)
    end_state_sampled[24:] = end_state[24:]

    state = env.reset(init_state_sampled)
    if not end_state:
        raise ValueError("end state is None")
    states = []
    actions = []
    audios = []
    labels = []

    t0 = num_steps_per_ep // 3 + random.randint(0, 2 * time_shift_max) - time_shift_max

    t1 = 2 * num_steps_per_ep // 3 + random.randint(0, 2 * time_shift_max) - time_shift_max

    action_noise = lambda: np.random.normal(np.zeros(action_space), [action_sigma]*action_space, action_space)

    for i in range(t0):
        action = np.zeros(action_space) + action_noise()
        action[24:] = 0.
        # action = (np.random.rand(action_space)) * 100.
        state, audio = env.step(action, True)
        states.append(state)
        actions.append(action)
        audios.append(audio)
        labels.append(initial_sound_name)
        env.render()

    for i in range(t1 - t0):
        action = (np.asarray(end_state_sampled) - np.asarray(init_state_sampled)) / (t1 - t0) + action_noise()
        action[24:] = 0.
        state, audio = env.step(action, True)
        states.append(state)
        actions.append(action)
        audios.append(audio)
        if initial_sound_name == end_sound_name:
            labels.append(initial_sound_name)
        else:
            labels.append(f"{initial_sound_name}{end_sound_name}")
        env.render()

    for i in range(num_steps_per_ep - t1):
        action = np.zeros(action_space) + action_noise()
        action[24:] = 0.
        # action = (np.random.rand(action_space)) * 100.
        state, audio = env.step(action, True)
        states.append(state)
        actions.append(action)
        audios.append(audio)
        labels.append(end_sound_name)
        env.render()

    actions = np.stack(actions)
    states = np.stack(states)
    audios = np.stack(audios)

    with open(os.path.join(directory, name+'.pkl'), 'wb') as f:
        pickle.dump({"audio": audios,
                     "tract_params": states[:, :24],
                     "glottis_params": states[:, 24:],
                     "action": actions,
                     "label": labels}, f, protocol=0)
    env.dump_episode(os.path.join(directory, name))
    # with open(os.path.join(directory, name + '.wav'), 'wb') as f:
    return audios, states, actions, labels


def create_datatset(**kwargs):
    dir_name = kwargs['dir']
    sound_names = kwargs['sound_names']
    num_samples_per_sound = kwargs['num_samples_per_sound']
    num_rollouts = num_samples_per_sound * len(sound_names) * len(sound_names)

    speaker_fname = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'JD2.speaker')
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VocalTractLab2.dll')
    ep_duration = 1000
    timestep = 40
    num_steps_per_ep = ep_duration // timestep

    env = VTLEnv(lib_path, speaker_fname, timestep, max_episode_duration=ep_duration)


    dt = str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p_%S"))
    save_dir = os.path.join(dir_name, dt)
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except:
            pass
    fname = os.path.join(save_dir, dt) + '.pd'

    rollouts = pd.DataFrame(columns=['y', 'audio', 'states', 'actions', 'labels'])
    i_g = 0
    for s0 in sound_names:
        for s1 in sound_names:
            name = s0 + "_" + s1
            sound_dir = os.path.join(save_dir, name)
            if not os.path.exists(sound_dir):
                try:
                    os.makedirs(sound_dir)
                except:
                    pass
            for i in range(num_samples_per_sound):

                initial_state = env.get_cf(s0)
                end_state = env.get_cf(s1)
                audios, states, actions, labels = create_reference(env, ep_duration, timestep,
                                                           initial_sound_name=s0,
                                                           end_sound_name=s1,
                                                           initial_state=initial_state, end_state=end_state, name=name, directory=sound_dir)
                ep_name = os.path.join(sound_dir, str(i))
                env.dump_episode(ep_name)
                rollouts.loc[i_g] = [name, np.array(audios), np.array(states), np.array(actions), np.array(labels)]
                i_g += 1

                if i_g % 500 == 0:
                    rollouts.to_pickle(path=fname)
                print('\rProgress: {} out of {}'.format(i_g, num_rollouts), end='')
            rollouts.to_pickle(path=fname)


if __name__ == '__main__':
    kwargs = {
        "dir": "C:/Study/SpeechAcquisitionModel/data/raw/Simple_transitions_s2s",
        "sound_names": ['a', 'i', 'u', 'o'],
        "num_samples_per_sound": 300
    }
    create_datatset(**kwargs)




    speaker_fname = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'JD2.speaker')
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VocalTractLab2.dll')
    ep_duration = 1000
    timestep = 40
    num_steps_per_ep = ep_duration // timestep

    env = VTLEnv(lib_path, speaker_fname, timestep, max_episode_duration=ep_duration)

    name = 'i_O_2'
    initial_state = env.get_cf('i')
    end_state = env.get_cf('O')

    directory = 'references'

    audios, _, _ = create_reference(env, ep_duration, timestep, initial_state=initial_state, end_state=end_state, name=name, directory=directory)

    preproc_params = {
        "numcep": 12,
        "winlen": 0.04,
        "winstep": 0.04,
        "sample_rate": 22050
    }
    sr = preproc_params['sample_rate']
    winlen = preproc_params['winlen']
    preproc = AudioPreprocessorFbank(**preproc_params)

    audio_sin_wave = np.sin(np.arange(0, len(audios.flatten()))*np.pi / int(sr * 0.1*winlen))

    flat_audio = audios.flatten()
    audio_framed = framesig(flat_audio, int(preproc_params['winlen'] * sr), int(preproc_params['winstep'] / 2 * sr))
    audio_stacked = np.stack(audio_framed).flatten()
    res0 = preproc(audios.flatten(), preproc_params['sample_rate'])
    res2 = preproc(audio_stacked, preproc_params['sample_rate'])
    res_sin = preproc(audio_sin_wave, preproc_params['sample_rate'])
    res1 = np.stack([preproc(audios[i, :], preproc_params['sample_rate']) for i in range(audios.shape[0])]).squeeze()

    res_lib_sin = librosa.feature.mfcc(y=audio_sin_wave, sr=sr, dct_type=1, n_mfcc=12, n_fft=int(sr*winlen), hop_length=int(sr*winlen), norm=None)[:,1:]
    res_lib_0 = librosa.feature.mfcc(y=audios.flatten(), sr=sr, dct_type=1, n_mfcc=12, n_fft=int(sr*winlen-10), hop_length=int(sr*winlen+1), norm=None)[:,1:]
    res_lib_1 = [librosa.feature.mfcc(y=audios[i, :], sr=sr, dct_type=1, n_mfcc=12, n_fft=int(sr*winlen-10), hop_length=int(sr*winlen+1), norm=None)[1:] for i in range(audios.shape[0])]
    res_lib_1 = np.array(res_lib_1).squeeze().T



    plt.figure()
    plt.imshow(res0.T)
    plt.figure()
    plt.imshow(res_lib_0)
    plt.figure()
    plt.imshow(res1.T)
    plt.figure()
    plt.imshow(res_lib_1)
    plt.figure()
    plt.imshow(res2.T)
    plt.figure()
    plt.imshow(res_sin.T)
    plt.figure()
    plt.imshow(res_lib_sin)

    plt.figure()
    plt.plot(audios.flatten())
    plt.figure()
    plt.plot(audio_sin_wave)
    plt.show()
    end = 0