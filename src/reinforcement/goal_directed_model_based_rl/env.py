import torch
import numpy as np
import random
import pickle

from src.speech_classification.pytorch_conv_lstm import LstmNet, LstmNetEnsemble
from src.VTL.vtl_environment import VTLEnv
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
            self.audio_bound = [(-0.015, 0.015)] * self.audio_dim  # should be changed (whats the bound of MFCC values?)

        self.state_dim += self.audio_dim
        self.state_bound.extend(self.audio_bound)

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
                self.preproc_net.eval()
                preproc_audio = torch.from_numpy(preprocessed).float().to(self.device)
                _, self._hidden, reference = self.preproc_net(preproc_audio,
                                                              seq_lens=np.array([preproc_audio.shape[1]]),
                                                              hidden=self._hidden)
                reference = reference.detach().cpu().numpy().squeeze()
                self.references.append(reference)
            else:
                self.references.append(preprocessed)

        # add reference in the state space
        self.state_dim += self.audio_dim
        self.state_bound.extend(self.audio_bound)
        self.current_reference_idx = 0

    def reset(self, state_to_reset=None, **kwargs):
        self.current_reference_idx = random.randint(0, len(self.references) - 1)
        goal = self.references[self.current_reference_idx][0]
        state_to_reset = state_to_reset if state_to_reset else goal
        state_out = super(VTLEnvPreprocAudioWithReference, self).reset(state_to_reset)
        return np.concatenate((state_out, goal))

    def step(self, action, render=True):
        action = action / 10
        action[24:] = 0.
        state_out, _, _, _ = super(VTLEnvPreprocAudioWithReference, self).step(action, render)

        goal = self.references[self.current_reference_idx][self.current_step]

        done = False
        if self.current_step >= self.references[self.current_reference_idx].shape[0] - 1:
            done = True
        reward = self._reward(state_out, goal)
        state_out = np.concatenate((state_out, goal))
        info = None
        return state_out, reward, done, info

    def _reward(self, state, goal):
        res = -np.sum((state[-self.audio_dim:] - goal) ** 2)
        return res


class VTLEnvWithReferenceTransition(VTLEnvPreprocAudio):
    def __init__(self, lib_path, speaker_fname, **kwargs):
        super(VTLEnvWithReferenceTransition, self).__init__(lib_path, speaker_fname, **kwargs)

        self.reference_fnames = kwargs['reference_fnames']
        self.references = []
        self.reference_actions = []
        for fname in self.reference_fnames:
            with open(fname, 'rb') as f:
                ref = pickle.load(f)
            # audio = np.array(ref['audio'])
            audio = np.int16(np.array(ref['audio'])* (2 ** 15 - 1))

            sr = 22050
            preprocessed = np.stack([self.preproc(audio[i, :], sr) for i in range(audio.shape[0])]).squeeze()
            if self.preproc_net:
                self.preproc_net.eval()
                preproc_audio = torch.from_numpy(preprocessed[np.newaxis]).float().to(self.device)
                self._hidden = None
                _, self._hidden, reference = self.preproc_net(preproc_audio,
                                                              seq_lens=np.array([preproc_audio.shape[1]]),
                                                              hidden=self._hidden)
                reference = reference.detach().cpu().numpy().squeeze()
                # ref['goal_audio'] = np.concatenate((np.zeros((1, reference.shape[1])), reference))
                ref['goal_audio'] = reference
            else:
                # ref['goal_audio'] = np.concatenate((np.zeros((1, preprocessed.shape[1])), preprocessed))
                ref['goal_audio'] = preprocessed
            # stack tract, glottis and goal audio into goal and then slice in subclasses if needed
            goal_ref = np.concatenate((np.asarray(ref['tract_params']),
                                       np.asarray(ref['glottis_params']),
                                       ref['goal_audio'][:, :]), axis=-1)

            # audio_max = ref['goal_audio'][3:,:].max(axis=0)
            # audio_bound = list(zip(-2.*audio_max, 2*audio_max))
            # self.audio_bound = audio_bound
            # vtl_names = self.tract_param_names + self.glottis_param_names
            # self.state_bound[len(vtl_names): len(vtl_names) + self.audio_dim] = self.audio_bound
            self.references.append(goal_ref)
            self.reference_actions.append(ref['action'][1:])

        # add reference both tract_glottis and goal_audio in the state space
        self.goal_dim = self.state_dim
        self.goal_bound = self.state_bound.copy()
        self.state_dim += self.goal_dim
        self.state_bound.extend(self.goal_bound)
        self.current_reference_idx = 0

    def reset(self, state_to_reset=None, **kwargs):
        self.current_reference_idx = random.randint(0, len(self.references) - 1)
        goal = self.references[self.current_reference_idx][0]
        state_to_reset = state_to_reset if state_to_reset else goal
        state_out = super(VTLEnvWithReferenceTransition, self).reset(state_to_reset)
        goal = self.references[self.current_reference_idx][self.current_step + 1]
        return np.concatenate((state_out, goal))

    def step(self, action, render=True):
        # goal idx is cur_idx + 1

        state_out, _, _, _ = super(VTLEnvWithReferenceTransition, self).step(action, render)
        goal = self.references[self.current_reference_idx][self.current_step + 1]

        done = False
        if self.current_step >= self.references[self.current_reference_idx].shape[0] - 1:
            done = True
        reward = self._reward(state_out, goal)
        state_out = np.concatenate((state_out, goal))
        info = None
        return state_out, reward, done, info

    def _reward(self, state, goal, action):
        res = -np.sum((state[-self.state_dim:] - goal) ** 2)
        return res


class VTLEnvWithReferenceTransitionMasked(VTLEnvWithReferenceTransition):
    def __init__(self, lib_path, speaker_fname, **kwargs):
        super(VTLEnvWithReferenceTransitionMasked, self).__init__(lib_path, speaker_fname, **kwargs)

        self.original_state_dim = self.state_dim
        self.original_state_bound = self.state_bound
        self.original_action_dim = self.action_dim
        self.original_action_bound = self.action_bound
        self.original_goal_dim = self.goal_dim
        self.original_goal_bound = self.goal_bound

        vtl_names = self.tract_param_names + self.glottis_param_names
        self.state_mask = np.zeros(self.original_state_dim, dtype=bool)

        self.state_mask_names = kwargs['state_parameters_selected']
        for name in self.state_mask_names:
            if name == 'audio':
                self.state_mask[len(vtl_names): len(vtl_names) + self.audio_dim] = 1
            else:
                try:
                    idx = vtl_names.index(name)
                    self.state_mask[idx] = 1
                except ValueError:
                    raise ValueError("unrecognised parameter name in VTL: {}".format(name))

        self.state_goal_mask = np.zeros(int(sum(self.state_mask)), dtype=bool)
        self.goal_mask_names = kwargs['goal_parameters_selected']
        self.goal_mask = np.zeros(self.goal_dim, dtype=bool)
        shift = self.original_state_dim - self.goal_dim
        for name in self.goal_mask_names:
            if name == 'audio':
                self.state_mask[len(vtl_names) + shift: len(vtl_names) + self.audio_dim + shift] = 1
                self.goal_mask[len(vtl_names): len(vtl_names) + self.audio_dim] = 1
            else:
                try:
                    idx = vtl_names.index(name)
                    self.state_mask[idx + shift] = 1
                    self.goal_mask[idx] = 1
                except ValueError:
                    raise ValueError("unrecognised parameter name in VTL: {}".format(name))

        j = 0
        for i in range(len(self.goal_mask)):
            if self.state_mask[i]:
                self.state_goal_mask[j] = self.goal_mask[i]
                j += 1

        self.action_mask_names = kwargs['action_parameters_selected']
        self.action_mask = np.zeros(self.original_action_dim, dtype=bool)
        for name in self.action_mask_names:
            try:
                idx = vtl_names.index(name)
                self.action_mask[idx] = 1
            except ValueError:
                raise ValueError("unrecognised parameter name in VTL: {}".format(name))

        self.state_dim = int(sum(self.state_mask))
        self.goal_dim = int(sum(self.goal_mask))
        self.action_dim = int(sum(self.action_mask))

        self.state_bound = np.array(self.state_bound)[np.array(self.state_mask)]
        self.action_bound = np.array(self.action_bound)[np.array(self.action_mask)]
        self.goal_bound = np.array(self.goal_bound)[np.array(self.goal_mask)]

    def reset(self, state_to_reset=None, **kwargs):
        state_out = super(VTLEnvWithReferenceTransitionMasked, self).reset(state_to_reset)


        # skip first 3 steps due to weird clicking sound in the beginning
        for i in range(3):
            # self._hidden = None
            reference_action = self.reference_actions[self.current_reference_idx][self.current_step ]
            state_out, reward, done, info = self.step(reference_action)
            # dif = self.references[self.current_reference_idx][i, 30:] - state_out[15:15+64]
            self.render()
        # constistency = state_out[15: 64 + 15] - self.references[self.current_reference_idx][self.current_step - 1][30:]

        return state_out

    def step(self, action, render=True):
        # cur_state = np.array(list(self.tract_params_out) + list(self.glottis_params_out))
        # goal = self.references[self.current_reference_idx][self.current_step + 1]
        # action_desired = goal - cur_state
        reference_action = self.reference_actions[self.current_reference_idx][self.current_step ]
        j = 0
        unmasked_action = np.zeros(len(reference_action))
        for i in range(len(reference_action)):
            if self.action_mask[i]:
                unmasked_action[i] = action[j]
                j += 1
            else:
                unmasked_action[i] = reference_action[i]

        cur_goal = self.references[self.current_reference_idx][self.current_step + 1]

        state_out, _, _, _ = super(VTLEnvWithReferenceTransition, self).step(unmasked_action, render)

        diff = state_out - cur_goal
        diff_1 = state_out - self.references[self.current_reference_idx][self.current_step + 1]

        # update goal only after step, beacuse it changes current step
        goal = self.references[self.current_reference_idx][self.current_step + 1]
        done = False
        if self.current_step >= self.references[self.current_reference_idx].shape[0] - 2:
            done = True
        s_m_normed = self.normalize(np.array(state_out), np.array(self.original_state_bound[:self.original_goal_dim]))[
            self.state_mask[:self.original_goal_dim]]
        g_m_normed = self.normalize(np.array(goal)[self.goal_mask],
                                    np.array(self.original_state_bound[:self.original_goal_dim])[self.goal_mask])
        reward = self._reward(s_m_normed, g_m_normed, action)
        state_out = np.concatenate((state_out, goal))
        state_out = state_out[self.state_mask]
        info = None
        return state_out, reward, done, info

    def _reward(self, state, goal, action):
        res = np.exp(-1. * 50.0 * np.mean((state[self.state_goal_mask] - goal) ** 2))
        return res

    def get_current_reference(self):
        return self.references[self.current_reference_idx][:, self.goal_mask]


class VTLEnvWithReferenceTransitionMaskedEntropyScore(VTLEnvWithReferenceTransitionMasked):
    def __init__(self, lib_path, speaker_fname, **kwargs):
        super(VTLEnvWithReferenceTransitionMaskedEntropyScore, self).__init__(lib_path, speaker_fname, **kwargs)
        self.device = kwargs['device']
        self.ensemble_classifier = torch.load(kwargs['ensemble_speech_classifier']).to(self.device)
        self.ensemble_hidden = None
        self.ref_class = None

    def reset(self, state_to_reset=None, **kwargs):
        state_out = super(VTLEnvWithReferenceTransitionMaskedEntropyScore, self).reset(state_to_reset)

        ref = self.references[self.current_reference_idx]

        self.ensemble_classifier.eval()
        self.ensemble_hidden = None
        preproc_audio = torch.from_numpy(np.array(ref[3:])[:, -self.audio_dim:]).float().to(self.device).view(1, len(ref)-3, -1)
        x, self.ensemble_hidden, lstm_out, pred_full = self.ensemble_classifier(preproc_audio,
                                                                                seq_lens=np.array(
                                                                                    [preproc_audio.shape[1]]),
                                                                                hidden=self.ensemble_hidden)
        pred_full = torch.stack(pred_full).squeeze()[:, :, :]
        softmax = (torch.nn.Softmax(dim=-1)(pred_full)).mean(dim=0).squeeze()
        _, self.ref_class = softmax[-1, :].max(0)
        self.ensemble_hidden = None
        return state_out

    def step(self, action, render=True):
        # cur_state = np.array(list(self.tract_params_out) + list(self.glottis_params_out))
        # goal = self.references[self.current_reference_idx][self.current_step + 1]
        # action_desired = goal - cur_state
        reference_action = self.reference_actions[self.current_reference_idx][self.current_step ]
        j = 0
        unmasked_action = np.zeros(len(reference_action))
        for i in range(len(reference_action)):
            if self.action_mask[i]:
                unmasked_action[i] = action[j]
                j += 1
            else:
                unmasked_action[i] = reference_action[i]

        cur_goal = self.references[self.current_reference_idx][self.current_step + 1]

        state_out, _, _, _ = super(VTLEnvWithReferenceTransition, self).step(unmasked_action, render)

        #sound
        self.ensemble_classifier.eval()
        preproc_audio = torch.from_numpy(np.array(state_out[-self.audio_dim:])).float().to(self.device).view(1, 1, -1)
        x, self.ensemble_hidden, lstm_out, pred_full = self.ensemble_classifier(preproc_audio,
                                                                                seq_lens=np.array([preproc_audio.shape[1]]),
                                                                                hidden=self.ensemble_hidden)
        pred_full = torch.stack(pred_full)
        log_softmax = (torch.nn.LogSoftmax(dim=-1)(pred_full)).mean(dim=0).squeeze()
        softmax = (torch.nn.Softmax(dim=-1)(pred_full)).mean(dim=0).squeeze()
        entropy = -1 * (log_softmax * softmax).sum(dim=0)
        prob = softmax

        # if self.ref_class is not None:
        #     prob = softmax[self.ref_class]
        # else:
        #     prob = torch.Tensor([0])

        diff = state_out - cur_goal
        diff_1 = state_out - self.references[self.current_reference_idx][self.current_step + 1]

        # update goal only after step, beacuse it changes current step
        goal = self.references[self.current_reference_idx][self.current_step + 1]
        done = False
        if self.current_step >= self.references[self.current_reference_idx].shape[0] - 2:
            done = True
            self.ensemble_hidden = None
        s_m_normed = self.normalize(np.array(state_out), np.array(self.original_state_bound[:self.original_goal_dim]))[
            self.state_mask[:self.original_goal_dim]]
        g_m_normed = self.normalize(np.array(goal)[self.goal_mask],
                                    np.array(self.original_state_bound[:self.original_goal_dim])[self.goal_mask])
        reward = self._reward(s_m_normed, g_m_normed, action)

        reward = (prob, entropy)

        state_out = np.concatenate((state_out, goal))
        state_out = state_out[self.state_mask]
        info = None
        return state_out, reward, done, info

    def _reward(self, state, goal, action):

        # extract



        res = np.exp(-1. * 50.0 * np.mean((state[self.state_goal_mask] - goal) ** 2))
        return res

    def get_current_reference(self):
        return self.references[self.current_reference_idx][:, self.goal_mask]
