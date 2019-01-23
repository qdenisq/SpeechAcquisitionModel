#!/usr/bin/env python3

"""
vtl api
"""
import os
import ctypes
import sys
import pickle
import time
import datetime
import subprocess
import numpy as np

# try to load some non-essential packages
try:
    import numpy as np
except ImportError:
    np = None
try:
    from scipy.io import wavfile
except ImportError:
    wavefile = None
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import av
except ImportError:
    av = None


# load vocaltractlab binary
# Use 'VocalTractLabApi32.dll' if you use a 32-bit python version.
if sys.platform == 'win32':
    VTL = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VocalTractLab2.dll'))
else:
    VTL = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(os.path.abspath(__file__)),'VocalTractLabApi64.so'))


# get version / compile date
version = ctypes.c_char_p(b'                                ')
VTL.vtlGetVersion(version)

# initialize vtl
speaker_file_name = ctypes.c_char_p(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'JD2.speaker').encode())

failure = VTL.vtlInitialize(speaker_file_name, False)
if failure != 0:
    raise ValueError('Error in vtlInitialize! Errorcode: %i' % failure)


# get some constants
audio_sampling_rate = ctypes.c_int(0)
number_tube_sections = ctypes.c_int(0)
number_vocal_tract_parameters = ctypes.c_int(0)
number_glottis_parameters = ctypes.c_int(0)

VTL.vtlGetConstants(ctypes.byref(audio_sampling_rate),
                    ctypes.byref(number_tube_sections),
                    ctypes.byref(number_vocal_tract_parameters),
                    ctypes.byref(number_glottis_parameters))

# get information about the parameters of the vocal tract model
# Hint: Reserve 32 chars for each parameter.
TRACT_PARAM_TYPE = ctypes.c_double * number_vocal_tract_parameters.value
tract_param_names = ctypes.c_char_p((' ' * 32 * number_vocal_tract_parameters.value).encode())
tract_param_min = TRACT_PARAM_TYPE()
tract_param_max = TRACT_PARAM_TYPE()
tract_param_neutral = TRACT_PARAM_TYPE()

VTL.vtlGetTractParamInfo(tract_param_names,
                         ctypes.byref(tract_param_min),
                         ctypes.byref(tract_param_max),
                         ctypes.byref(tract_param_neutral))

tract_param_max = list(tract_param_max)
tract_param_min = list(tract_param_min)
tract_param_neutral = list(tract_param_neutral)

# get information about the parameters of glottis model
# Hint: Reserve 32 chars for each parameter.
GLOTTIS_PARAM_TYPE = ctypes.c_double * number_glottis_parameters.value
glottis_param_names = ctypes.c_char_p((' ' * 32 * number_glottis_parameters.value).encode())
glottis_param_min = GLOTTIS_PARAM_TYPE()
glottis_param_max = GLOTTIS_PARAM_TYPE()
glottis_param_neutral = GLOTTIS_PARAM_TYPE()

VTL.vtlGetGlottisParamInfo(glottis_param_names,
                           ctypes.byref(glottis_param_min),
                           ctypes.byref(glottis_param_max),
                           ctypes.byref(glottis_param_neutral))

glottis_param_max = list(glottis_param_max)
glottis_param_min = list(glottis_param_min)
glottis_param_neutral = list(glottis_param_neutral)


def cf_transition(cf_1, cf_2, num_frames):
    tract_params = []
    glottis_params = []
    for i in range(int(num_frames / 3)):
        tract_params.append([cf_1[p] for p in range(24)])
        glottis_params.append([cf_1[p] for p in range(24, 30)])
    for i in range(int(num_frames / 3), int(2. * num_frames / 3)):
        tract_params.append([cf_1[p] + (cf_2[p] - cf_1[p]) * (3. * i - num_frames) / num_frames for p in range(24)])
        glottis_params.append(
            [cf_1[p] + (cf_2[p] - cf_1[p]) * (3. * i - num_frames) / num_frames for p in range(24, 30)])
    for i in range(int(2 * num_frames / 3), num_frames):
        tract_params.append([cf_2[p] for p in range(24)])
        glottis_params.append([cf_2[p] for p in range(24, 30)])
    return tract_params, glottis_params


def get_cf(sound_name):
    shape_name = ctypes.c_char_p(sound_name.encode())
    cf = TRACT_PARAM_TYPE()
    failure = VTL.vtlGetTractParams(shape_name, ctypes.byref(cf))
    if failure != 0:
        raise ValueError('Error in vtlGetTractParams! Errorcode: %i' % failure)
    cf = list(cf)
    cf.extend(glottis_param_neutral)
    return cf


def synth_dynamic_sound(s1, s2, duration, frame_rate):
    cf_1 = get_cf(s1)
    cf_2 = get_cf(s2)
    num_frames = int(duration * frame_rate)
    tract_params, glottis_params = cf_transition(cf_1, cf_2, num_frames)

    audio = (ctypes.c_double * int(duration * audio_sampling_rate.value + 2000))()
    number_audio_samples = ctypes.c_int(int(duration * audio_sampling_rate.value + 2000))

    tract_params_transition = (ctypes.c_double * (number_vocal_tract_parameters.value * num_frames))()
    glottis_params_transition = (ctypes.c_double * (number_glottis_parameters.value * num_frames))()

    tract_params_flatten = [i for sl in tract_params for i in sl]
    for i in range(len(tract_params_transition)):
        tract_params_transition[i] = tract_params_flatten[i]

    glottis_params_flatten = [i for sl in glottis_params for i in sl]
    for i in range(len(glottis_params_transition)):
        glottis_params_transition[i] = glottis_params_flatten[i]
    # init the arrays
    tube_areas = (ctypes.c_double * (num_frames * number_tube_sections.value))()
    tube_articulators = ctypes.c_char_p(b' ' * num_frames * number_tube_sections.value)

    # Call the synthesis function. It may calculate a few seconds.
    failure = VTL.vtlSynthBlock(ctypes.byref(tract_params_transition),  # input
                                ctypes.byref(glottis_params_transition),  # input
                                ctypes.byref(tube_areas),  # output
                                tube_articulators,  # output
                                num_frames,  # input
                                ctypes.c_double(frame_rate),  # input
                                ctypes.byref(audio),  # output
                                ctypes.byref(number_audio_samples))  # output

    if failure != 0:
        raise ValueError('Error in vtlSynthBlock! Errorcode: %i' % failure)
    wav = np.array(audio[:-2000])
    wav_audio = np.int16(wav * (2 ** 15 - 1))
    return wav_audio, tract_params, glottis_params


def create_reference(s1, s2, duration=1., frame_rate=50, name='', directory=''):
    if name is '':
        name = '{}_{}'.format(s1, s2)
    audio, tract_params, glottis_params = synth_dynamic_sound(s1, s2, duration, frame_rate)
    wavfile.write(os.path.join(directory, name+'.wav'), audio_sampling_rate.value, audio)
    with open(os.path.join(directory, name+'.pkl'), 'wb') as f:
        pickle.dump({"audio": audio,
                     "tract_params": tract_params,
                     "glottis_params": glottis_params}, f, protocol=0)
    return audio, tract_params, glottis_params


def test_create_reference():
    s1 = 'a'
    s2 = 'i'
    name = s1 + '_' + s2
    directory = 'references'
    create_reference(s1, s2, name=name, directory=directory)
    with open(os.path.join(directory, name + '.pkl'), 'rb') as f:
        ref = pickle.load(f)
    print(ref)


def tes_dynamic_sound_synthesis():
    s1 = 'a'
    s2 = 'o'
    audio, tract_params, glottis_params = synth_dynamic_sound(s1, s2)
    wavfile.write('ai_test.wav', audio_sampling_rate.value, audio)

if __name__ == '__main__':
    test_create_reference()
