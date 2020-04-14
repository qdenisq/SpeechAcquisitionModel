import os
import ctypes
import sys
import datetime
import time
import subprocess
import av
from scipy.io import wavfile
import numpy as np
from gym import spaces
import gym
import pickle
from collections import defaultdict
import copy

def convert_to_gym(space):
    return spaces.Box(np.array(space[0]), np.array(space[1]))


class VTLEnv(gym.Env):
    def __init__(self, lib_path,
                 speaker_fname,
                 timestep=10,
                 max_episode_duration=1000,
                 img_width=400,
                 img_height=400,
                 showControlPoints = True,
                 showCenterLine = False,
                 showCutVectors = True,
                 renderBothSides = False,
                 renderMode = 2,
                 **kwargs):
        # load vocaltractlab binary
        # Use 'VocalTractLabApi32.dll' if you use a 32-bit python version.
        if sys.platform == 'win32':
            self.VTL = ctypes.cdll.LoadLibrary(lib_path)
        else:
            self.VTL = ctypes.cdll.LoadLibrary(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VocalTractLabApi64.so'))

        # get version / compile date
        version = ctypes.c_char_p(b'                                ')
        self.VTL.vtlGetVersion(version)
        print('Compile date of the library: "%s"' % version.value.decode())

        # initialize vtl
        speaker_file_name = ctypes.c_char_p(speaker_fname.encode())

        failure = self.VTL.vtlInitialize(speaker_file_name,
                                         True, # render always true
                                         showControlPoints,
                                         showCenterLine,
                                         showCutVectors,
                                         renderBothSides,
                                         renderMode)
        if failure != 0:
            raise ValueError('Error in vtlInitialize! Errorcode: %i' % failure)

        # get some constants
        audio_sampling_rate = ctypes.c_int(0)
        number_tube_sections = ctypes.c_int(0)
        number_vocal_tract_parameters = ctypes.c_int(0)
        number_glottis_parameters = ctypes.c_int(0)

        self.VTL.vtlGetConstants(ctypes.byref(audio_sampling_rate),
                            ctypes.byref(number_tube_sections),
                            ctypes.byref(number_vocal_tract_parameters),
                            ctypes.byref(number_glottis_parameters))

        self.audio_sampling_rate = audio_sampling_rate.value
        self.number_tube_sections = number_tube_sections.value
        self.number_vocal_tract_parameters = number_vocal_tract_parameters.value
        self.number_glottis_parameters = number_glottis_parameters.value

        # get information about the parameters of the vocal tract model
        # Hint: Reserve 32 chars for each parameter.
        TRACT_PARAM_TYPE = ctypes.c_double * number_vocal_tract_parameters.value
        tract_param_names = ctypes.c_char_p((' ' * 32 * number_vocal_tract_parameters.value).encode())
        tract_param_min = TRACT_PARAM_TYPE()
        tract_param_max = TRACT_PARAM_TYPE()
        tract_param_neutral = TRACT_PARAM_TYPE()

        self.VTL.vtlGetTractParamInfo(tract_param_names,
                                 ctypes.byref(tract_param_min),
                                 ctypes.byref(tract_param_max),
                                 ctypes.byref(tract_param_neutral))

        self.tract_param_names = [v.decode() for v in tract_param_names.value.split()]
        self.tract_param_max = list(tract_param_max)
        self.tract_param_min = list(tract_param_min)
        self.tract_param_neutral = list(tract_param_neutral)

        # get information about the parameters of glottis model
        # Hint: Reserve 32 chars for each parameter.
        GLOTTIS_PARAM_TYPE = ctypes.c_double * number_glottis_parameters.value
        glottis_param_names = ctypes.c_char_p((' ' * 32 * number_glottis_parameters.value).encode())
        glottis_param_min = GLOTTIS_PARAM_TYPE()
        glottis_param_max = GLOTTIS_PARAM_TYPE()
        glottis_param_neutral = GLOTTIS_PARAM_TYPE()

        self.VTL.vtlGetGlottisParamInfo(glottis_param_names,
                                   ctypes.byref(glottis_param_min),
                                   ctypes.byref(glottis_param_max),
                                   ctypes.byref(glottis_param_neutral))

        self.glottis_param_names = [v.decode() for v in glottis_param_names.value.split()]
        self.glottis_param_max = list(glottis_param_max)
        self.glottis_param_min = list(glottis_param_min)
        self.glottis_param_neutral = list(glottis_param_neutral)

        self.tract_params_acts = (ctypes.c_double * (number_vocal_tract_parameters.value))()
        self.glottis_params_acts = (ctypes.c_double * (number_glottis_parameters.value))()

        self.tract_params_out = (ctypes.c_double * (number_vocal_tract_parameters.value))()
        self.glottis_params_out = (ctypes.c_double * (number_glottis_parameters.value))()

        self.timestep = timestep
        self.max_episode_duration = max_episode_duration
        self.max_number_of_frames = self.max_episode_duration // self.timestep
        self.img_width = img_width
        self.img_height = img_height

        self.audio_buffer = (ctypes.c_double * int(self.max_episode_duration * audio_sampling_rate.value + 2000))()
        self.img_buffer = (ctypes.c_ubyte * int(self.img_width * self.img_height * 3))()

        self.video_stream = np.zeros(shape=(self.max_number_of_frames, int(self.img_width * self.img_height * 3)), dtype=np.uint8)
        self.audio_stream = np.zeros(int(self.max_episode_duration * audio_sampling_rate.value + 2000))

        self.action_dim = len(self.glottis_param_neutral) + len(self.tract_param_neutral)
        action_frac = 0.1
        self.action_bound = list(zip(-abs(np.subtract(self.tract_param_min, self.tract_param_max)) * action_frac,
                                     abs(np.subtract(self.tract_param_max, self.tract_param_min)) * action_frac))
        self.action_bound.extend(zip(-abs(np.subtract(self.glottis_param_min, self.glottis_param_max)) * action_frac,
                                     abs(np.subtract(self.glottis_param_max, self.glottis_param_min)) * action_frac))

        self.state_dim = len(self.glottis_param_neutral) + len(self.tract_param_neutral)
        self.state_bound = list(zip(self.tract_param_min, self.tract_param_max))
        self.state_bound.extend(zip(self.glottis_param_min, self.glottis_param_max))

        self.action_space = convert_to_gym(list(zip(*self.action_bound)))
        self.observation_space = convert_to_gym(list(zip(*self.state_bound)))
        self.current_step = 0
        self.episode_states = []

        self.id = np.random.randint(0, 10000)

        self.episode_history = defaultdict(list)
        return

    def _before_step(self):
        pass


    def _after_step(self):
        pass

    def _step(self, action, render=True):
        for i in range(self.number_vocal_tract_parameters):
            self.tract_params_acts[i] = action[i]
        for i in range(self.number_glottis_parameters):
            self.glottis_params_acts[i] = action[i + self.number_vocal_tract_parameters]

        self.VTL.vtlStep(ctypes.c_int(self.timestep),
                    ctypes.byref(self.tract_params_acts),
                    ctypes.byref(self.glottis_params_acts),
                    ctypes.byref(self.tract_params_out),
                    ctypes.byref(self.glottis_params_out),
                    ctypes.byref(self.audio_buffer),
                    ctypes.byref(self.img_buffer),
                    ctypes.c_bool(render)
                         )

        self.video_stream[self.current_step, :] = self.img_buffer

        # (int)(interval_ms * SAMPLING_RATE / 1000.0)
        idx = int(self.current_step * self.audio_sampling_rate * self.timestep / 1000)
        idx_1 = int((self.current_step + 1) * self.audio_sampling_rate * self.timestep / 1000)
        self.audio_stream[idx:idx_1] = self.audio_buffer[:int(self.audio_sampling_rate * self.timestep / 1000)]

        state_out = list(self.tract_params_out) + list(self.glottis_params_out)
        audio_out = self.audio_stream[idx:idx_1]
        self.current_step += 1

        self.current_state = np.concatenate((state_out, audio_out))
        self.episode_states.append(self.current_state)
        self.episode_history['vt'].append(copy.deepcopy(state_out))
        return state_out, audio_out

    def step(self, action, render=True):
        self._before_step()
        res = self._step(action, render)
        self._after_step()
        return res

    def reset(self, state_to_reset=None):
        self.episode_history = defaultdict(list)
        self.episode_states = []
        if state_to_reset is not None:
            tract_params_to_reset = (ctypes.c_double * (self.number_vocal_tract_parameters))()
            glottis_params_to_reset = (ctypes.c_double * (self.number_glottis_parameters))()
            for i in range(self.number_vocal_tract_parameters):
                tract_params_to_reset[i] = state_to_reset[i]
            for i in range(self.number_glottis_parameters):
                glottis_params_to_reset[i] = state_to_reset[i + self.number_vocal_tract_parameters]

            self.VTL.vtlReset(ctypes.byref(self.tract_params_out),
                              ctypes.byref(self.glottis_params_out),
                              ctypes.byref(tract_params_to_reset),
                              ctypes.byref(glottis_params_to_reset)
                              )
        else:
            self.VTL.vtlReset(ctypes.byref(self.tract_params_out),
                              ctypes.byref(self.glottis_params_out),
                              0,
                              0
                              )

        state_out = list(self.tract_params_out) + list(self.glottis_params_out)
        self.episode_history['vt'].append(state_out)
        self.current_step = 0
        # self.episode_states.append(state_out)
        return state_out

    def render(self, *args, **kwargs):
        self.VTL.vtlRender()
        return

    def close(self):
        self.VTL.vtlClose()
        return

    def dump_episode(self, *args, fname=None, **kwargs ):
        if fname is None:
            directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            fname = directory + '/videos/episode_' + str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p_%S")) + '_' + str(self.id)

        # save states to pickle
        with open(f'{fname}.pkl', 'wb') as f:
            pickle.dump(self.episode_states, f)

        codec = 'mpeg4'
        bitrate = 8000000
        format = 'yuv420p'
        rate = str(1000.0 / self.timestep)
        width = self.img_width
        height = self.img_height

        wav = np.array(self.audio_stream[:int(self.audio_sampling_rate * self.current_step * self.timestep / 1000)])
        wav_int = np.int16(wav * (2 ** 15 - 1))
        wavfile.write(fname + '.wav', self.audio_sampling_rate, wav_int)

        output = av.open(fname + ".mp4", 'w')
        stream = output.add_stream(codec, rate)
        stream.bit_rate = bitrate
        stream.pix_fmt = format

        for i in range(self.current_step):

            img = self.video_stream[i].reshape(self.img_width, self.img_height, 3)
            img = np.flip(img, axis=0)

            if not i:
                stream.height = img.shape[0]
                stream.width = img.shape[1]

            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format='rgb24')
            packet = stream.encode(frame)
            output.mux(packet)

        output.close()
        cmd = 'ffmpeg -y -i {}.wav  -r 30 -i {}.mp4  -filter:a aresample=async=1 -c:v copy {}_v.mp4'.format(
            fname, fname, fname)
        with open(os.devnull, 'w') as devnull:
            # subprocess.call(cmd, shell=False, stdout=devnull, stderr=devnull)  # "Muxing Done
            p = subprocess.Popen(cmd, shell=False, stdout=devnull, stderr=devnull)  # "Muxing Done
            p.communicate()
            p.wait()
        return fname+"_v"

    def get_episode_history(self, *args, **kwargs):
        self.episode_history['vt'] = np.array(self.episode_history['vt'])
        return self.episode_history

    def get_cf(self, sound_name):
        shape_name = ctypes.c_char_p(sound_name.encode())
        cf =(ctypes.c_double * (self.number_vocal_tract_parameters))()
        failure = self.VTL.vtlGetTractParams(shape_name, ctypes.byref(cf))
        if failure != 0:
            raise ValueError('Error in vtlGetTractParams! Errorcode: %i' % failure)
        cf = list(cf)
        cf.extend(self.glottis_param_neutral)
        return cf

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


def run_test():
    speaker_fname = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'JD2.speaker')
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VocalTractLab2.dll')
    num_episodes = 5
    ep_duration = 5000
    timestep = 20
    num_steps_per_ep = ep_duration // timestep

    env = VTLEnv(lib_path, speaker_fname, timestep, max_episode_duration=ep_duration)

    action_space = env.number_glottis_parameters + env.number_vocal_tract_parameters

    for i in range(num_episodes):
        time_start = time.time()

        for step in range(num_steps_per_ep):
            action = (np.random.rand(action_space) - 0.5) * 0.1
            action[env.number_glottis_parameters:] = 0.
            env.step(action, True)
            if (step % 10 == 0):
                env.render()

        time_elapsed = time.time() - time_start
        print("iterations: {}; time simulated: {:2f}sec; time elapsed: {:2f}sec".format(step, step * timestep/1000, time_elapsed))
        env.dump_episode()
        env.reset()


if __name__ == '__main__':
    run_test()