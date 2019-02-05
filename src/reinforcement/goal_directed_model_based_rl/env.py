import torch
import numpy as np
import random
import pickle

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

            self.audio_dim = kwargs["audio_dim"]
            self.audio_bound = [(-1.0, 1.0)]*self.audio_dim
            self._hidden = None

        else:
            self.preproc_net = None
            self.audio_dim = kwargs["nump_cep"]
            self.audio_bound = [(-1.0, 1.0)] * self.audio_dim # should be changed (whats the bound of MFCC values?)

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
        return np.concatenate((state_out, np.zeros(self.audio_dim)))

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
        self.state_dim += self.audio_dim
        self.state_bound.extend(self.audio_bound)
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
        res = -np.sum((state[-self.audio_dim:] - goal)**2)
        return res


class VTLEnvWithReferenceTransition(VTLEnvPreprocAudio):
    def __init__(self, lib_path, speaker_fname, **kwargs):
        super(VTLEnvWithReferenceTransition, self).__init__(lib_path, speaker_fname, **kwargs)

        self.reference_fnames = kwargs['reference_fnames']
        self.references = []
        for fname in self.reference_fnames:
            with open(fname, 'rb') as f:
                ref = pickle.load(f)
            audio = ref['audio']
            sr = 22050
            preprocessed = self.preproc(audio, sr)[np.newaxis]
            if self.preproc_net:
                preproc_audio = torch.from_numpy(preprocessed).float().to(self.device)
                _, self._hidden, reference = self.preproc_net(preproc_audio,
                                                              seq_lens=np.array([preproc_audio.shape[1]]),
                                                              hidden=self._hidden)
                reference = reference.detach().cpu().numpy().squeeze()
                ref['goal_audio'] = reference
            else:
                ref['goal_audio'] = preprocessed
            # stack tract, glottis and goal audio into goal and then slice in subclasses if needed
            goal_ref = np.concatenate((np.asarray(ref['tract_params']),
                                       np.asarray(ref['glottis_params']),
                                       ref['goal_audio']), axis=-1)
            self.references.append(goal_ref)

        # add reference both tract_glottis and goal_audio in the state space
        self.goal_dim = self.state_dim
        self.goal_bound = self.state_bound.copy()
        self.state_dim += self.goal_dim
        self.state_bound.extend(self.goal_bound)
        self.current_reference_idx = 0

    def reset(self, state_to_reset=None, **kwargs):
        state_out = super(VTLEnvWithReferenceTransition, self).reset(state_to_reset)
        self.current_reference_idx = random.randint(0, len(self.references) - 1)
        goal = self.references[self.current_reference_idx][self.current_step]
        return np.concatenate((state_out, goal))

    def step(self, action, render=True):
        # get goal before the step bcs it will increase cur step
        goal = self.references[self.current_reference_idx][self.current_step]

        state_out, _, _, _ = super(VTLEnvWithReferenceTransition, self).step(action, render)
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
        state_out = super(VTLEnvWithReferenceTransition, self).reset(state_to_reset)
        self.current_reference_idx = random.randint(0, len(self.references) - 1)
        goal = self.references[self.current_reference_idx][self.current_step]
        return np.concatenate((state_out, goal))[self.state_mask]

    def step(self, action, render=True):
        cur_state = np.array(list(self.tract_params_out) + list(self.glottis_params_out))
        reference_action = self.references[self.current_reference_idx][self.current_step][:len(cur_state)] - cur_state
        unmasked_action = [action[i] if self.action_mask[i] else reference_action[i] for i in range(len(reference_action))]

        goal = self.references[self.current_reference_idx][self.current_step]

        state_out, _, _, _ = super(VTLEnvWithReferenceTransition, self).step(unmasked_action, render)
        done = False
        if self.current_step >= self.references[self.current_reference_idx].shape[0] - 1:
            done = True
        s_m_normed = self.normalize(np.array(state_out), np.array(self.original_state_bound[:self.original_goal_dim]))[self.state_mask[:self.original_goal_dim]]
        g_m_normed = self.normalize(np.array(goal)[self.goal_mask], np.array(self.original_state_bound[:self.original_goal_dim])[self.goal_mask])
        reward = self._reward(s_m_normed, g_m_normed, action)
        state_out = np.concatenate((state_out, goal))
        state_out = state_out[self.state_mask]
        info = None
        return state_out, reward, done, info

    def _reward(self, state, goal, action):
        res = np.exp(-10.*np.sum((state[self.state_goal_mask] - goal) ** 2))
        return res

    def get_current_reference(self):
        return self.references[self.current_reference_idx][:, self.goal_mask]















