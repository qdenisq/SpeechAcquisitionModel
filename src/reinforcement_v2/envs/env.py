import gym
import copy

from src.reinforcement_v2.utils.utils import str_to_class
from src.reinforcement_v2.common.env_wrappers import NormalizedActionWrapper, NormalizedObservationWrapper, VectorizedWrapper
from src.reinforcement_v2.envs.base_env import *
from src.reinforcement_v2.envs import *


class EnvironmentManager:
    def __init__(self):
        self.env = None
        self.env_id_to_class_name = {"vtl_base": "VTLEnvPreprocAudio",
                                     "dtw_we_vtl": "VTLDTWEnv"
                                     }

    def make(self, env_id, *args, **kwargs):
        num_workers = kwargs.pop('num_workers')
        seed = kwargs.pop('seed')
        self.env = VectorizedWrapper(
            [lambda: self._make_single_env(env_id, *args, seed=seed + i, **copy.deepcopy(kwargs)) for i in range(num_workers)])
        return self.env

    # def close(self):
    #     self.env.close()
    #     if self.osim_like:
    #
    #         os_type = platform.system()
    #         print("Reopen opensim")
    #         self.env.vtgt.vis['plt'].close()
    #         print("killing proc")
    #         for proc in psutil.process_iter():
    #             if os_type == 'Windows':
    #                 if proc.name() == "simbody-visualizer.exe":
    #                     proc.kill()
    #             if os_type == 'Linux' or 'Darwin':
    #                 if proc.name() == "simbody-visualizer":
    #                     proc.kill()
    #         print("proc killed")
    #     self.env = None
    #     self.osim_like = False
    #     pass

    def _make_single_env(self, env_id, *args, **kwargs):

        # create base env and then wrap it according to config
        norm_obs = kwargs.pop('norm_observation')
        norm_act = kwargs.pop('norm_action')

        if 'vtl' in env_id:
            env = self._make_vtl_like(env_id, **kwargs)
        else:
            env = gym.make(env_id, **kwargs)

        if norm_obs:
            env = NormalizedObservationWrapper(env)

        if norm_act:
            env = NormalizedActionWrapper(env)

        return env

    def _make_vtl_like(self, env_id, *args, **kwargs):
        class_name = self.env_id_to_class_name[env_id]
        env = str_to_class(__name__, class_name)(**kwargs)
        return env
