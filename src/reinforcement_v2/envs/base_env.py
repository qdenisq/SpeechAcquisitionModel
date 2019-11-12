import torch
import numpy as np
import random
import pickle

from src.speech_classification.pytorch_conv_lstm import LstmNet, LstmNetEnsemble
from src.VTL.vtl_environment import VTLEnv, convert_to_gym
from src.speech_classification.audio_processing import AudioPreprocessorFbank


class VTLEnvPreprocAudio(VTLEnv):
    def __init__(self, lib_path, speaker_fname, **kwargs):
        super(VTLEnvPreprocAudio, self).__init__(lib_path, speaker_fname, **kwargs)

        self.preproc = AudioPreprocessorFbank(**kwargs['preprocessing_params'])
        self.sr = kwargs['preprocessing_params']['sample_rate']

        if "preproc_net_fname" in kwargs:
            self.device = kwargs['device']
            self.preproc_net = torch.load(kwargs['preproc_net_fname']).to(self.device)

            self.audio_dim = kwargs["audio_dim"]
            self.audio_bound = [(-1., 1.)] * self.audio_dim
            self._hidden = None

        else:
            self.preproc_net = None
            self.audio_dim = self.preproc.get_dim()
            self.audio_bound = [(-0.01, 0.01)] * self.audio_dim  # should be changed (whats the bound of MFCC values?)

        self.state_dim += self.audio_dim
        self.state_bound.extend(self.audio_bound)

        self.observation_space = convert_to_gym(list(zip(*self.state_bound)))


    @staticmethod
    def normalize(data, bound):
        largest = np.array([max(abs(y[0]), abs(y[1])) for y in bound])
        normed_data = data / largest
        return normed_data

    @staticmethod
    def denormalize(normed_data, bound):
        largest = np.array([max(abs(y[0]), abs(y[1])) for y in bound])
        data = normed_data * largest
        return data

    def reset(self, state_to_reset=None, **kwargs):
        state_out = super(VTLEnvPreprocAudio, self).reset(state_to_reset)
        if self.preproc_net:
            self._hidden = None
        return np.concatenate((state_out, np.zeros(self.audio_dim)))

    def step(self, action, render=True):
        state_out, audio_out = super(VTLEnvPreprocAudio, self).step(action, render)
        preproc_audio = self.preproc(audio_out, self.sr)
        if self.preproc_net:
            self.preproc_net.eval()
            preproc_audio = torch.from_numpy(preproc_audio[np.newaxis]).float().to(self.device)
            _, self._hidden, new_goal_state = self.preproc_net(preproc_audio,
                                                               seq_lens=np.array([preproc_audio.shape[1]]),
                                                               hidden=self._hidden)
            new_goal_state = new_goal_state.detach().cpu().numpy().squeeze()
            state_out.extend(new_goal_state)
        else:
            state_out.extend(preproc_audio.squeeze())

        # there is no reward and time limit constraint for this environment
        done = None
        if self.current_step > 40:
            done = True
        reward = None
        info = {}
        return state_out, reward, done, info