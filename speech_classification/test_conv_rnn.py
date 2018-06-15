import utils
from audio_processing import AudioProcessor
from convRNN import ConvRNN, Conv2dNN, MultiConvRNN
import tensorflow as tf

import numpy as np
from time import time
from matplotlib import pyplot as plt
from conv.input_data import AudioProcessor as A_proc

def qwetest_conv_rnn():
    # We want to see all the logging messages for this tutorial.
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Session() as sess:
        model_settings = utils.get_default_model_settings()
        model_settings['wanted_words'] = 'yes,no'

        data_dir = model_settings['data_dir']
        wanted_words = model_settings['wanted_words']
        validation_percentage = model_settings['validation_percentage']
        testing_percentage = model_settings['testing_percentage']
        model_settings['strip_window_size_ms'] = 30.0
        model_settings['strip_window_stride_ms'] = 30.0
        model_settings['window_size_ms'] = 30.0
        model_settings['window_stride_ms'] = 10.0


        proc = AudioProcessor(model_settings,
                                      data_dir,
                                      wanted_words=wanted_words.split(','),
                                      validation_percentage=validation_percentage,
                                      testing_percentage=testing_percentage)
        model_settings = proc._model_settings
        conv_rnn = Conv2dNN("conv2d", model_settings)
        conv_rnn.train(proc)

def test_restore_multi_conv_rnn():
    name = 'multi_conv_rnn_3'
    meta_path = r'C:\tmp\speech_commands_train\multi_conv_rnn_3.ckpt-6000.meta'
    ckpnt_path = r'C:\tmp\speech_commands_train\multi_conv_rnn_3.ckpt-6000'
    with tf.Session() as sess:
        rnn = MultiConvRNN.restore(name, meta_path, ckpnt_path)


def test_restore_and_predict():
    name = 'multi_conv_rnn_3'
    meta_path = r'C:\tmp\speech_commands_train\multi_conv_rnn_4.ckpt-6000.meta'
    ckpnt_path = r'C:\tmp\speech_commands_train\multi_conv_rnn_4.ckpt-6000'
    with tf.Session() as sess:
        # restore rnn
        rnn = MultiConvRNN.restore(name, meta_path, ckpnt_path)
        # import model settings
        model_settings = utils.get_default_model_settings()
        data_dir = model_settings['data_dir']
        wanted_words = model_settings['wanted_words']
        validation_percentage = model_settings['validation_percentage']
        testing_percentage = model_settings['testing_percentage']
        model_settings['strip_window_size_ms'] = 30.0
        model_settings['strip_window_stride_ms'] = 10.0
        model_settings['window_size_ms'] = 30.0
        model_settings['window_stride_ms'] = 10.0
        model_settings['how_many_training_steps'] = '3000,3000'

        # init audio preprocessing object
        proc = AudioProcessor(model_settings,
                              data_dir,
                              wanted_words=wanted_words.split(','),
                              validation_percentage=validation_percentage,
                              testing_percentage=testing_percentage)

        # create 1 sample
        batch_size = 1
        batchX, batchY = proc.get_data(batch_size, 0,'training')
        pred = rnn.predict(batchX)
        print("correct y:{} ; predicted y:{}".format(batchY, pred))


def test_restore_and_get_hidden_state():
    name = 'multi_conv_rnn_4'
    meta_path = r'C:\tmp\speech_commands_train\multi_conv_rnn_4.ckpt-6000.meta'
    ckpnt_path = r'C:\tmp\speech_commands_train\multi_conv_rnn_4.ckpt-6000'
    with tf.Session() as sess:
        # restore rnn
        rnn = MultiConvRNN.restore(name, meta_path, ckpnt_path)
        # import model settings
        model_settings = utils.get_default_model_settings()
        data_dir = model_settings['data_dir']
        wanted_words = model_settings['wanted_words']
        validation_percentage = model_settings['validation_percentage']
        testing_percentage = model_settings['testing_percentage']
        model_settings['strip_window_size_ms'] = 30.0
        model_settings['strip_window_stride_ms'] = 10.0
        model_settings['window_size_ms'] = 30.0
        model_settings['window_stride_ms'] = 10.0
        model_settings['how_many_training_steps'] = '3000,3000'

        # init audio preprocessing object
        proc = AudioProcessor(model_settings,
                              data_dir,
                              wanted_words=wanted_words.split(','),
                              validation_percentage=validation_percentage,
                              testing_percentage=testing_percentage)

        # create 1 sample
        batch_size = 1
        batchX, batchY = proc.get_data(batch_size, 0, 'training')
        outputs, state = rnn.get_hidden_state(batchX)
        print(outputs[0].shape)

# qwetest_conv_rnn()
# test_restore_multi_conv_rnn()
# test_restore_and_predict()
test_restore_and_get_hidden_state()