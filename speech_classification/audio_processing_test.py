from audio_processing import AudioProcessor
from utils import get_default_model_settings
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from time import time

def test_1_sample():
    model_settings = get_default_model_settings()
    data_dir = model_settings['data_dir']
    proc = AudioProcessor(data_dir, model_settings)
    fname = r"C:\tmp\speech_dataset\dog\00b01445_nohash_0.wav"
    with tf.Session() as sess:
        chunks, mfcc = proc.get_sequential_data_sample(fname)

    print(chunks.shape)
    for i in range(chunks.shape[0]):
        plt.figure()
        plt.imshow(np.transpose(chunks[i]), origin='lower')
    plt.figure()
    plt.imshow(np.transpose(mfcc), origin='lower')
    plt.show()
    return

def test_training_data_gathering():
    np.random.seed(0)
    model_settings = get_default_model_settings()
    data_dir = model_settings['data_dir']
    wanted_words = model_settings['wanted_words']
    validation_percentage = model_settings['validation_percentage']
    testing_percentage = model_settings['testing_percentage']
    model_settings['strip_window_size_ms'] = 100.0
    model_settings['strip_window_stride_ms'] = 50.0
    model_settings['window_size_ms'] = 30.0
    model_settings['window_stride_ms'] = 10.0

    with tf.Session() as sess:
        proc = AudioProcessor(model_settings,
                              data_dir,
                              wanted_words=wanted_words.split(','),
                              validation_percentage=validation_percentage,
                              testing_percentage=testing_percentage)
        print('spectrogram_length', proc._model_settings['spectrogram_length'])
        print('strip_window_size_samples', proc._model_settings['strip_window_size_samples'])
        print('strip_array_length', proc._model_settings['strip_array_length'])
        print('strip_window_stride_samples', proc._model_settings['strip_window_stride_samples'])
        tic = time()
        data, labels = proc.get_data(10, 0, 'training')
    toc = time()
    print("time", toc -tic)
    print('data_shape', data.shape)
    print(labels[0])

    strip_array_length = data.shape[1]
    fig, axes = plt.subplots(1, strip_array_length, sharey=True)
    sample_idx = 9
    norm = plt.Normalize(np.min(data[sample_idx]), np.max(data[sample_idx]))
    print(np.min(data[sample_idx]), np.max(data[sample_idx]))
    for i in range(strip_array_length):
        axes[i].imshow(np.transpose(data[sample_idx, i]), origin='low', interpolation='none', norm=norm)
        axes[i].set_title(labels[sample_idx])
    plt.show()
    return


test_training_data_gathering()