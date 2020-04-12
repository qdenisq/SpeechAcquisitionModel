import torch
import numpy as np
import copy
import random
import pickle
import dtwalign
import os
import datetime

from src.reinforcement_v2.envs.masked_dtw_we_env import VTLMaskedActionDTWEnv
from src.VTL.vtl_environment import VTLEnv, convert_to_gym
from src.reinforcement_v2.utils.utils import str_to_class
from src.reinforcement_v2.envs.base_env import VTLEnvPreprocAudio
from src.soft_dtw_awe.audio_processing import AudioPreprocessorMFCCDeltaDelta
from src.soft_dtw_awe.model import SiameseDeepLSTMNet
from src.soft_dtw_awe.soft_dtw import SoftDTW



class VTLRefMaskedActionDTWEnv(VTLMaskedActionDTWEnv):
    """
    This env includes reference in the observation space
    """
    def __init__(self, lib_path, speaker_fname, **kwargs):
        super(VTLRefMaskedActionDTWEnv, self).__init__(lib_path, speaker_fname, **kwargs)

        self.selected_ref_params = kwargs['selected_reference_state']
        vtl_names = self.tract_param_names + self.glottis_param_names
        self.ref_tract_param_name_to_idx = dict(zip(vtl_names, range(len(vtl_names))))
        self.ref_tract_param_idx_to_name = dict(zip(range(len(vtl_names)), vtl_names))
        self.selected_ref_param_idx = [self.ref_tract_param_name_to_idx[name] for name in self.selected_ref_params if name in self.ref_tract_param_name_to_idx]

        # change state space
        self.agent_state_dim = self.state_dim

        ref_tract_state_bound = [self.state_bound[i] for i in self.selected_ref_param_idx]
        self.state_dim += len(ref_tract_state_bound)
        self.state_bound.extend(ref_tract_state_bound)

        if "ACOUSTICS" in self.selected_ref_params:
            self.state_dim += self.audio_dim
            self.state_bound.extend(self.audio_bound)
        self.observation_space = convert_to_gym(list(zip(*self.state_bound)))

        if "ACOUSTICS" in self.selected_ref_params:
            self.reference_mask = self.selected_ref_param_idx + [i + len(vtl_names) for i in range(self.audio_dim)]
        else:
            self.reference_mask = self.selected_ref_param_idx

    def get_current_ref_obs(self):
        ref_full_vtl_state = np.concatenate((self.cur_reference['tract_params'][self.current_step + 1, :],
                                              self.cur_reference['glottis_params'][self.current_step + 1, :]))
        ref_obs = ref_full_vtl_state[self.selected_ref_param_idx]

        if "ACOUSTICS" in self.selected_ref_params:
            cols_per_step = int(self.timestep / 1000 / self.preproc_params['winstep'])

            ref_ac_obs = self.cur_reference['acoustics'][self.current_step*cols_per_step: (self.current_step + 1)*cols_per_step, :].flatten().squeeze()
            ref_obs = np.concatenate((ref_obs, ref_ac_obs))
        return ref_obs

    def reset(self, state_to_reset=None, **kwargs):
        obs = super(self).reset(state_to_reset, **kwargs)
        ref_obs = self.get_current_ref_obs()
        res = np.concatenate((obs, ref_obs))
        return res

    def _step(self, action, render=True):

        state_out, reward, done, info = super(VTLRefMaskedActionDTWEnv, self)._step(action, render)
        if self.current_step >= int(self.max_episode_duration / self.timestep) - 1:
            ref_obs = np.zeros(self.state_dim - len(state_out))
        else:
            ref_obs = self.get_current_ref_obs()

        state_out = np.concatenate((state_out, ref_obs))
        return state_out, reward, done, info