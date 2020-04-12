import torch
import numpy as np
import copy
import random
import pickle
import dtwalign
import os
import datetime

from src.VTL.vtl_environment import VTLEnv, convert_to_gym
from src.soft_dtw_awe.audio_processing import AudioPreprocessorMFCCDeltaDelta
from src.reinforcement_v2.utils.utils import str_to_class
from src.reinforcement_v2.envs.base_env import VTLEnvPreprocAudio
from src.soft_dtw_awe.model import SiameseDeepLSTMNet
from src.soft_dtw_awe.soft_dtw import SoftDTW


class VTLDTWEnv(VTLEnvPreprocAudio):
    """
    This env implements reference handling, acoustic processing, calculation DTW between synthesized speech and the
    reference acouctics
    """
    def __init__(self, lib_path, speaker_fname, **kwargs):
        super(VTLDTWEnv, self).__init__(lib_path, speaker_fname, **kwargs)

        # load references
        self.references = []
        reference_fnames = kwargs['references']
        for fname in reference_fnames:
            ref_tem = {}
            with open(fname, 'rb') as f:
                ref = pickle.load(f)
            ref_item = copy.deepcopy(ref)
            ref_item['fname'] = fname
            audio = np.array(ref['audio']).flatten()
            sr = 22050
            zero_pad = np.zeros(int(self.audio_sampling_rate * (self.preproc_params['winlen'] - self.preproc_params['winstep'])))
            audio = np.concatenate((zero_pad, audio))
            preprocessed = self.preproc(audio, sr)[np.newaxis]
            ref_item['mfcc'] = preprocessed
            # preprocessed = preprocessed[:, 10:, :] # skip weird clicking in the begining
            # preprocessed = self.preproc(fname)[np.newaxis]
            if self.preproc_net:
                self.preproc_net.eval()
                preproc_audio = torch.from_numpy(preprocessed).float().to(self.device)
                reference, self._hidden = self.preproc_net.single_forward(preproc_audio,
                                                                          hidden=self._hidden)
                reference = reference.detach().cpu().numpy().squeeze()
                ref_item['acoustics'] = reference
            else:
                ref_item['acoustics'] = preprocessed.squeeze()

            self.references.append(ref_item)

        self.cur_reference = self.references[np.random.randint(0, len(self.references))]

        self.dist_params = copy.deepcopy(kwargs['distance'])

    def reset(self, state_to_reset=None, **kwargs):
        self.cur_reference = self.references[np.random.randint(0, len(self.references))]

        if state_to_reset is None:
            state_to_reset = np.concatenate((self.cur_reference['tract_params'][0, :],
                                             self.cur_reference['glottis_params'][0, :]))

        res = super().reset(state_to_reset, **kwargs)
        return res

    def _step(self, action, render=True):
        state_out, audio_out = super(VTLEnvPreprocAudio, self)._step(action, render)

        # self.audio_buffer.extend(audio_out)
        # pad audio with zeros to ensure even number of preprocessed columns per each timestep (boundary case for the first time step)
        zero_pad = np.zeros(int(self.audio_sampling_rate * (self.preproc_params['winlen'] - self.preproc_params['winstep'])))
        audio = np.array(self.audio_stream[:int(self.current_step * self.audio_sampling_rate * self.timestep / 1000)])
        audio = np.concatenate((zero_pad, audio))

        # preproc_audio = self.preproc(audio_out, self.sr)
        preproc_audio = self.preproc(audio, self.sr)

        if self.preproc_net:
            self.preproc_net.eval()
            preproc_audio_tensor = torch.from_numpy(preproc_audio[np.newaxis]).float().to(self.device)
            embeddings, self._hidden = self.preproc_net.single_forward(preproc_audio_tensor,
                                                                       hidden=None)

            embeddings = embeddings.detach().cpu()
            embeddings = embeddings.numpy().squeeze()

        else:
            embeddings = preproc_audio.squeeze()

        state_out.extend(embeddings[-int(self.timestep / 1000 / self.preproc_params['winstep']):].flatten())

        # calc open_end dtw distance between embeddings and current reference
        dist = self.calc_distance(embeddings, self.cur_reference['acoustics'])
        l1_dist = self.cur_reference['mfcc'].squeeze()[:preproc_audio.shape[0], :] - preproc_audio

        # there is no reward and time limit constraint for this environment
        done = False
        if self.current_step >= int(self.max_episode_duration / self.timestep) - 1:
            done = True
            # self.episode_states = []
            # self.episode_states = embeddings
            # self.dump_episode()

        reward = dist
        if not np.isnan(reward):
            reward = np.exp(0. - reward)
        else:
            reward = 0.
        info = {'dtw_dist': dist,
                'l1_dist': l1_dist}
        
        self.current_state = state_out

        return state_out, reward, done, info

    def dump_reference(self, fname=None):
        if fname is None:
            directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            fname = directory + '/videos/reference_episode_' + str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p_%S")) + '_' + str(self.id)
        else:
            fname += f'_{self.id}'

        # save states to pickle
        with open(f'{fname}.pkl', 'wb') as f:
            pickle.dump(self.cur_reference, f)

    def calc_distance(self, s, reference):
        if self.dist_params['name'] == 'soft-DTW':
            dist_func = SoftDTW(open_end=self.dist_params['open_end'], dist=self.dist_params['dist'])
            return dist_func(torch.from_numpy(s).to(self.device),
                             torch.from_numpy(reference).to(self.device)).detach().cpu().item()
        elif self.dist_params['name'] == 'dtwalign':
            return dtwalign.dtw(s, reference, dist=self.dist_params['dist'], step_pattern=self.dist_params['step_pattern'],
                         open_end=self.dist_params['open_end'], dist_only=True).normalized_distance
        else:
            raise KeyError(f"unknown name of the dist:  {self.dist_params['name']}")
