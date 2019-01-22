import torch
import numpy as np
import random

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
            self._hidden = None

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

    def reset(self, state_to_reset=None, **kwargs):
        state_out = super(VTLEnvPreprocAudio, self).reset(state_to_reset)
        return np.concatenate((state_out, np.zeros(self.goal_dim)))

    def step(self, action, render=True):
        state_out, audio_out = super(VTLEnvPreprocAudio, self).step(action, render)
        preproc_audio = self.preproc(audio_out, self.sr)[np.newaxis]
        if self.preproc_net:
            preproc_audio = torch.from_numpy(preproc_audio).float().to(self.device)
            _, self._hidden, new_goal_state = self.preproc_net(preproc_audio,
                                                         seq_lens=np.array([preproc_audio.shape[1]]),
                                                         hidden=self._hidden)
            new_goal_state = new_goal_state.detach().cpu().numpy().squeeze()
            state_out.extend(new_goal_state)
        else:
            state_out.extend(preproc_audio)

        # there is no reward and time limit constraint for this environment
        reward = None
        done = None
        info = None
        return state_out, reward, done, info


class VTLEnvPreprocAudioWithReference(VTLEnvPreprocAudio):
    def __init__(self, lib_path, speaker_fname, **kwargs):
        super(VTLEnvPreprocAudioWithReference, self).__init__(lib_path, speaker_fname, **kwargs)

        self.reference_fnames = kwargs['reference_fnames']
        self.references = []
        for fname in self.reference_fnames:
            preprocessed = self.preproc(fname)[np.newaxis]
            if self.preproc_net:
                preproc_audio = torch.from_numpy(preprocessed).float().to(self.device)
                _, self._hidden, reference = self.preproc_net(preproc_audio,
                                                                    seq_lens=np.array([preproc_audio.shape[1]]),
                                                                    hidden=self._hidden)
                reference = reference.detach().cpu().numpy().squeeze()
                self.references.append(reference)
            else:
                self.references.append(preprocessed)

        # add reference in the state space
        self.state_dim += self.goal_dim
        self.state_bound.extend(self.goal_bound)
        self.current_reference_idx = 0

    def reset(self, state_to_reset=None, **kwargs):
        state_out = super(VTLEnvPreprocAudioWithReference, self).reset(state_to_reset)
        self.current_reference_idx = random.randint(0, len(self.references) - 1)
        goal = self.references[self.current_reference_idx][self.current_step]
        return np.concatenate((state_out, goal))

    def step(self, action, render=True):
        action = action / 10
        action[24:] = 0.
        state_out, _, _, _ = super(VTLEnvPreprocAudioWithReference, self).step(action,render)

        goal = self.references[self.current_reference_idx][self.current_step]

        done = False
        if self.current_step >= self.references[self.current_reference_idx].shape[0] - 1:
            done = True
        reward = self._reward(state_out, goal)
        state_out = np.concatenate((state_out, goal))
        info = None
        return state_out, reward, done, info

    def _reward(self, state, goal):
        res = -np.sum((state[-self.goal_dim:] - goal)**2)
        return res











