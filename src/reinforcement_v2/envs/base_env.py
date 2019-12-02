import torch
import numpy as np
import copy
import random
import pickle
import os
import datetime

from src.speech_classification.pytorch_conv_lstm import LstmNet, LstmNetEnsemble
from src.VTL.vtl_environment import VTLEnv, convert_to_gym
from src.speech_classification.audio_processing import AudioPreprocessorFbank, AudioPreprocessorMFCCDeltaDelta
from src.reinforcement_v2.utils.utils import str_to_class


class VTLEnvPreprocAudio(VTLEnv):
    def __init__(self, lib_path, speaker_fname, **kwargs):
        super(VTLEnvPreprocAudio, self).__init__(lib_path, speaker_fname, **kwargs)

        self.preproc_params = copy.deepcopy(kwargs['preprocessing_params'])
        preproc_name = self.preproc_params['name']
        self.preproc = str_to_class(__name__, preproc_name)(**self.preproc_params)

        self.sr = kwargs['preprocessing_params']['sample_rate']

        # self.audio_buffer = []
        if "preproc_net" in kwargs:
            self.device = kwargs["preproc_net"]['device']
            self.preproc_net = torch.load(kwargs["preproc_net"]['preproc_net_fname']).to(self.device)

            cols_per_timestep = kwargs['timestep'] / 1000 / self.preproc_params['winstep']  # preproc could return more than 1 column of features per timestep
            self.audio_dim = kwargs["preproc_net"]["output_dim"] * int(cols_per_timestep)
            self.audio_bound = [(-1., 1.)] * self.audio_dim
            self._hidden = None

        else:
            self.preproc_net = None
            cols_per_timestep = kwargs['timestep'] / 1000 / self.preproc_params['winstep'] # preproc could return more than 1 column of features per timestep
            self.audio_dim = self.preproc.get_dim() * int(cols_per_timestep)
            self.audio_bound = [(-10.01, 10.01)] * self.audio_dim  # should be changed (whats the bound of MFCC values?)

        self.state_dim += self.audio_dim
        self.state_bound.extend(self.audio_bound)

        self.observation_space = convert_to_gym(list(zip(*self.state_bound)))

    def reset(self, state_to_reset=None, **kwargs):
        state_out = super(VTLEnvPreprocAudio, self).reset(state_to_reset)
        if self.preproc_net:
            self._hidden = None
        return np.concatenate((state_out, np.zeros(self.audio_dim)))

    def _step(self, action, render=True):
        state_out, audio_out = super(VTLEnvPreprocAudio, self)._step(action, render)

        # self.audio_buffer.extend(audio_out)
        # pad audio with zeros to ensure even number of preprocessed columns per each timestep (boundary case for the first time step)
        zero_pad = np.zeros(int(self.audio_sampling_rate * (self.preproc_params['winlen'] - self.preproc_params['winstep'])))
        audio = np.array(self.audio_stream[:int(self.current_step * self.audio_sampling_rate * self.timestep / 1000)])
        audio = np.concatenate((zero_pad, audio))

        # preproc_audio = self.preproc(audio_out, self.sr)
        preproc_audio = self.preproc(audio, self.sr)
        # slice columns corresponding to the last timestep
        preproc_audio = preproc_audio[-int(self.timestep / 1000 / self.preproc_params['winstep']):]

        if self.preproc_net:
            self.preproc_net.eval()
            preproc_audio = torch.from_numpy(preproc_audio[np.newaxis]).float().to(self.device)
            _, self._hidden, new_goal_state = self.preproc_net(preproc_audio,
                                                               seq_lens=np.array([preproc_audio.shape[1]]),
                                                               hidden=self._hidden)
            new_goal_state = new_goal_state.detach().cpu().numpy().squeeze()
            state_out.extend(new_goal_state)
        else:
            state_out.extend(preproc_audio.flatten().squeeze())

        self.current_state = state_out

        # there is no reward and time limit constraint for this environment
        done = False
        if self.current_step > int(self.max_episode_duration / self.timestep) - 1:
            done = True
        reward = None
        info = {}

        self.current_state = state_out
        return state_out, reward, done, info

    def dump_episode(self, *args, fname=None, **kwargs ):
        if fname is None:
            directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            fname = directory + '/videos/episode_' + str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p_%S")) + str(self.id)
        else:
            fname += f'_{self.id}'
        super(VTLEnvPreprocAudio, self).dump_episode(*args, fname=fname, **kwargs)


