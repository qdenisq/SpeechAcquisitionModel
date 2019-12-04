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
from src.reinforcement_v2.envs.dtw_we_env import VTLDTWEnv
from src.siamese_net_sound_similarity.train_v2 import SiameseDeepLSTMNet
from src.siamese_net_sound_similarity.soft_dtw import SoftDTW


class VTLMaskedActionDTWEnv(VTLDTWEnv):
    """
    This env allows agent to articulate only selected actions (the rest is substituted with ground-truth actions from the reference)
    """
    def __init__(self, lib_path, speaker_fname, **kwargs):
        super(VTLMaskedActionDTWEnv, self).__init__(lib_path, speaker_fname, **kwargs)

        self.selected_actions = kwargs['selected_actions']
        vtl_names = self.tract_param_names + self.glottis_param_names
        self.action_name_to_idx = dict(zip(vtl_names, range(len(vtl_names))))
        self.action_idx_to_name = dict(zip(range(len(vtl_names)), vtl_names))
        self.selected_actions_idx = [self.action_name_to_idx[name] for name in self.selected_actions]

        # change action space
        self.action_bound = [self.action_bound[i] for i in self.selected_actions_idx]
        self.action_space = convert_to_gym(list(zip(*self.action_bound)))

    def _step(self, action, render=True):

        # unmask action

        full_action = self.cur_reference['action'][self.current_step]
        full_action[self.selected_actions_idx] = action

        res = super(VTLMaskedActionDTWEnv, self)._step(full_action, render)
        return res