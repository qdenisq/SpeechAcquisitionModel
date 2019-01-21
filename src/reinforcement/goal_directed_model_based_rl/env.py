import torch
import numpy as np

from src.VTL.vtl_environment import VTLEnv
from src.speech_classification.audio_processing import AudioPreprocessor


class VTLEnvPreprocAudio(VTLEnv):
    def __init__(self, lib_path, speaker_fname, **kwargs):
        super(VTLEnvPreprocAudio, self).__init__(lib_path, speaker_fname, **kwargs)

        self.preproc = AudioPreprocessor(**kwargs['preprocessing_params'])
        self.sr = kwargs['preprocessing_params']['sample_rate']

        if "preproc_net_fname" in kwargs:
            self.device = kwargs['device']
            self.preproc_net = torch.load(kwargs['preproc_net_fname']).to(self.device)

            self.goal_dim = kwargs["goal_dim"]
            self.goal_bound = [(-1.0, 1.0)]*self.goal_dim
            self.__hidden = None

        else:
            self.preproc_net = None
            self.goal_dim = kwargs["nump_cep"]
            self.goal_bound = [(-1.0, 1.0)] * self.goal_dim # should be changed (whats the bound of MFCC values?)

        self.state_dim += self.goal_dim
        self.state_bound.extend(self.goal_bound)

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

    def step(self, action, render=True):
        state_out, audio_out = super(VTLEnvPreprocAudio, self).step(action, render)
        preproc_audio = self.preproc(audio_out, self.sr)[np.newaxis]
        if self.preproc_net:
            preproc_audio = torch.from_numpy(preproc_audio).float().to(self.device)
            _, self.__hidden, new_goal_state = self.preproc_net(preproc_audio,
                                                         seq_lens=np.array([preproc_audio.shape[1]]),
                                                         hidden=self.__hidden)
            new_goal_state = new_goal_state.detach().cpu().numpy().squeeze()
            state_out.extend(new_goal_state)
        else:
            state_out.extend(preproc_audio)

        # there is no reward and time limit constraint for this environment
        reward = None
        done = None
        return state_out, reward, done


class VTLEnvPreprocAudioWithReference(VTLEnv):
    def __init__(self, lib_path, speaker_fname, **kwargs):
        super(VTLEnvPreprocAudioWithReference, self).__init__(lib_path, speaker_fname, **kwargs)

        self.reference_fnames = kwargs['reference_fnames']









