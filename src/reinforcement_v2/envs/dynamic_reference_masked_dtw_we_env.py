import torch
import numpy as np
import copy
import random
import pickle
import dtwalign
import os
import datetime

from src.speech_classification.pytorch_conv_lstm import LstmNet, LstmNetEnsemble
from src.VTL.vtl_environment import VTLEnv, convert_to_gym
from src.speech_classification.audio_processing import AudioPreprocessorFbank, AudioPreprocessorMFCCDeltaDelta
from src.reinforcement_v2.utils.utils import str_to_class
from src.reinforcement_v2.envs.base_env import VTLEnvPreprocAudio
from src.reinforcement_v2.envs.masked_dtw_we_env import VTLMaskedActionDTWEnv
from src.reinforcement_v2.envs.reference_masked_dtw_we_env import VTLRefMaskedActionDTWEnv

from src.siamese_net_sound_similarity.train_v2 import SiameseDeepLSTMNet
from src.siamese_net_sound_similarity.soft_dtw import SoftDTW


class VTLDynamicRefMaskedActionDTWEnv(VTLRefMaskedActionDTWEnv):
    """
    This env includes reference in the observation space
    """
    def __init__(self, lib_path, speaker_fname, **kwargs):
        super(VTLDynamicRefMaskedActionDTWEnv, self).__init__(lib_path, speaker_fname, **kwargs)

    # dynamic reference frame based on dtw dist
    def get_current_ref_obs(self):
        # print(self.current_step)
        if self.current_step <= 2:
            ref_full_vtl_state = np.concatenate((self.cur_reference['tract_params'][self.current_step + 1, :],
                                                  self.cur_reference['glottis_params'][self.current_step + 1, :]))
            ref_obs = ref_full_vtl_state[self.selected_ref_param_idx]

            if "ACOUSTICS" in self.selected_ref_params:
                cols_per_step = int(self.timestep / 1000 / self.preproc_params['winstep'])

                ref_ac_obs = self.cur_reference['acoustics'][self.current_step*cols_per_step: (self.current_step + 1)*cols_per_step, :].flatten().squeeze()
                ref_obs = np.concatenate((ref_obs, ref_ac_obs))
        else:
            # dtw

            # ep_states = np.array(self.episode_states)[:, -self.audio_dim:]
            # acoustic dtw

            ref_ac = self.cur_reference['acoustics']
            ep_ac = np.array(self.episode_states)[:, -self.audio_dim:].reshape(-1, ref_ac.shape[-1])
            dtw_res_ac = dtwalign.dtw(ep_ac, ref_ac, open_end=True, step_pattern="symmetricP2")
            last_ref_matched_elem = dtw_res_ac.path[-1, 1]

            #artic dtw
            # TODO: do smth with art dtw as well
            ref_full_vtl_state = np.concatenate((self.cur_reference['tract_params'][:, :],
                                                 self.cur_reference['glottis_params'][:, :]), axis=-1)
            ep_ar = np.array(self.episode_states)[:,  self.selected_ref_param_idx]
            ref_ar = ref_full_vtl_state[:, self.selected_ref_param_idx]
            dtw_res_ar = dtwalign.dtw(ep_ar, ref_ar, open_end=True, step_pattern="symmetricP2")
            print(dtw_res_ar.path)

            last_ref_matched_elem_ar = dtw_res_ar.path[-1, 1]


            if "ACOUSTICS" in self.selected_ref_params:
                cols_per_step = int(self.timestep / 1000 / self.preproc_params['winstep'])

                last_matched_step = (last_ref_matched_elem + 1) // cols_per_step
                last_matched_step = last_ref_matched_elem_ar

                last_matched_step = min(self.cur_reference['tract_params'].shape[0] - 2, last_matched_step)
                print(last_matched_step, self.current_step)

                ref_ac_obs = self.cur_reference['acoustics'][
                             (last_matched_step ) * cols_per_step: (last_matched_step + 1) * cols_per_step,
                             :].flatten().squeeze()

                ref_full_vtl_state = np.concatenate((self.cur_reference['tract_params'][last_matched_step + 1, :],
                                                     self.cur_reference['glottis_params'][last_matched_step + 1, :]))
                ref_obs = ref_full_vtl_state[self.selected_ref_param_idx]

                ref_obs = np.concatenate((ref_obs, ref_ac_obs))

        return ref_obs
    #
    # def reset(self, state_to_reset=None, **kwargs):
    #     obs = super().reset(state_to_reset, **kwargs)
    #     ref_obs = self.get_current_ref_obs()
    #     res = np.concatenate((obs, ref_obs))
    #     return res

    def _step(self, action, render=True):

        state_out, reward, done, info = super(VTLRefMaskedActionDTWEnv, self)._step(action, render)
        if self.current_step >= int(self.max_episode_duration / self.timestep) - 1:
            ref_obs = np.zeros(self.state_dim - len(state_out))
        else:
            ref_obs = self.get_current_ref_obs()

        ref_full_vtl_state = np.concatenate((self.cur_reference['tract_params'][:, :],
                                             self.cur_reference['glottis_params'][:, :]), axis=-1)
        ep_ar = np.array(self.episode_states)[:, self.selected_ref_param_idx]
        ref_ar = ref_full_vtl_state[:, self.selected_ref_param_idx]
        dtw_res_ar = dtwalign.dtw(ep_ar, ref_ar, open_end=True, step_pattern="symmetricP2")

        # if max(abs(ref_ar[ep_ar.shape[0], :] - ep_ar[-1, :])) > 0.8:
        #     done = True
        if dtw_res_ar.distance > 5 and self.current_step > 2:
            done = True

        state_out = np.concatenate((state_out, ref_obs))
        return state_out, reward, done, info