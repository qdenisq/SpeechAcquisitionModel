import gym
import numpy as np
import torch

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.vec_env.subproc_vec_env import _flatten_obs


class NormalizedActionWrapper(gym.ActionWrapper):
    def action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        # action = np.clip(action, low, high)

        return action

    def action_torch(self, action):
        low = torch.from_numpy(self.action_space.low).to(action.device)
        high = torch.from_numpy(self.action_space.high).to(action.device)

        action = low + (action + 1.0) * 0.5 * (high - low)
        # action = torch.clamp(action, low, high)

        return action


class NormalizedObservationWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        low = self.observation_space.low
        high = self.observation_space.high
        observation = 2 * (observation - low) / (high - low) - 1.

        return observation

    def reverse_observation(self, observation):
        low = self.observation_space.low
        high = self.observation_space.high

        observation = low + np.add(observation, 1.0) * 0.5 * (high - low)
        return observation


class VectorizedWrapper(SubprocVecEnv):
    def __init__(self, env_fns, start_method=None):
        super(VectorizedWrapper, self).__init__(env_fns, start_method)

    def render(self, mode='human', *args, **kwargs):
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(('render', (args, {'mode': 'human', **kwargs})))
        imgs = [pipe.recv() for pipe in self.remotes]
        return
        # imgs = [pipe.recv() for pipe in self.remotes]
        # # Create a big image by tiling images from subprocesses
        # bigimg = tile_images(imgs)
        # if mode == 'human':
        #     import cv2
        #     cv2.imshow('vecenv', bigimg[:, :, ::-1])
        #     cv2.waitKey(1)
        # elif mode == 'rgb_array':
        #     return bigimg
        # else:
        #     raise NotImplementedError

    def dump_episode(self, *args, **kwargs):
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(('env_method', ('dump_episode', args, {**kwargs})))
        res = [pipe.recv() for pipe in self.remotes]
        return

    def reset(self, remotes=None):
        """Reset specified environments. If remotes is None, then all environments will be reseted"""
        if remotes is None:
            remotes = np.arange(len(self.remotes))
        for i in remotes:
            self.remotes[i].send(('reset', None))
        obs = [self.remotes[i].recv() for i in remotes]
        return _flatten_obs(obs, self.observation_space)
