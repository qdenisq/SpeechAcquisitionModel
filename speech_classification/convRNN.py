from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import math
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
from numpy.lib.stride_tricks import as_strided


class BaseNN(object):
    def __init__(self, name, model_settings):
        self._model_settings = model_settings
        self.name = name
        self._logits = None
        self._batchX = None
        self._batchY = None
        self._learning_rate_input = None
        self._cross_entropy_mean = None
        self._train_step = None
        self._confusion_matrix = None
        self._evaluation_step = None
        self._increment_global_step = None
        self._dropout_prob = None
        # self.build_forward_pass_graph()
        # self.build_train_graph()
        return

    def build_train_graph(self):
        assert(tf.get_default_session() is not None)
        # Create the back propagation and training evaluation machinery in the graph.
        num_classes = self._model_settings['label_count']
        self._batchY = tf.placeholder(tf.int64, [None], name='batchY')

        # control_dependencies = []
        # checks = tf.add_check_numerics_ops()
        # control_dependencies = [checks]

        with tf.name_scope('cross_entropy'):
            cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
                labels=self._batchY, logits=self._logits)
            self._cross_entropy_mean = tf.identity(cross_entropy_mean, name='cross_entropy_mean')
        tf.summary.scalar('cross_entropy', self._cross_entropy_mean)
        with tf.name_scope('train'):
            self._learning_rate_input = tf.placeholder(
                tf.float32, [], name='learning_rate_input')
            self._train_step = tf.train.AdamOptimizer(
                self._learning_rate_input).minimize(self._cross_entropy_mean, name='train_step')
        predicted_indices = tf.argmax(self._logits, 1)
        correct_prediction = tf.equal(predicted_indices, self._batchY)
        self._confusion_matrix = tf.confusion_matrix(
            self._batchY, predicted_indices, num_classes=num_classes, name='confusion_matrix')
        self._evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='evaluation_step')
        tf.summary.scalar('accuracy', self._evaluation_step)

        global_step = tf.train.get_or_create_global_step()
        self._increment_global_step = tf.assign(global_step, global_step + 1, name='increment_global_step')

    def train(self, audio_processor):
        sess = tf.get_default_session()
        assert(sess is not None)

        training_steps_list = list(map(int, self._model_settings['how_many_training_steps'].split(',')))
        learning_rates_list = list(map(float, self._model_settings['learning_rate'].split(',')))
        batch_size = self._model_settings['batch_size']
        if len(training_steps_list) != len(learning_rates_list):
            raise Exception(
                '--how_many_training_steps and --learning_rate must be equal length '
                'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                           len(learning_rates_list)))

        saver = tf.train.Saver(tf.global_variables())

        # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
        merged_summaries = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self._model_settings['summaries_dir'] + '/{}_train'.format(self.name),
                                             sess.graph)
        validation_writer = tf.summary.FileWriter(self._model_settings['summaries_dir'] + '/{}_validation'.format(self.name))

        tf.global_variables_initializer().run()

        start_step = 1

        # if FLAGS.start_checkpoint:
        #     models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
        #     start_step = global_step.eval(session=sess)

        tf.logging.info('Training from step: %d ', start_step)

        # Save graph.pbtxt.
        tf.train.write_graph(sess.graph_def, self._model_settings['train_dir'],
                             self.name + '.pbtxt')

        # Save list of words.
        with gfile.GFile(
                os.path.join(self._model_settings['train_dir'], self.name + '_labels.txt'),
                'w') as f:
            f.write('\n'.join(self._model_settings['wanted_words']))

        # Training loop.
        training_steps_max = np.sum(training_steps_list)
        for training_step in xrange(start_step, training_steps_max + 1):
            # Figure out what the current learning rate is.
            training_steps_sum = 0
            for i in range(len(training_steps_list)):
                training_steps_sum += training_steps_list[i]
                if training_step <= training_steps_sum:
                    learning_rate_value = learning_rates_list[i]
                    break

            batchX, batchY = audio_processor.get_data(batch_size, 0,'training')
            # Run the graph with this batch of training data.
            train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
                [
                    merged_summaries, self._evaluation_step, self._cross_entropy_mean, self._train_step,
                    self._increment_global_step
                ],
                feed_dict={
                    self._batchX: batchX,
                    self._batchY: batchY,
                    self._learning_rate_input: learning_rate_value,
                    self._dropout_prob: 0.5
                })
            train_writer.add_summary(train_summary, training_step)
            tf.logging.info('Step #%d: rate %f, accuracy %.1f%%, cross entropy %f' %
                            (training_step, learning_rate_value, train_accuracy * 100,
                             cross_entropy_value))
            is_last_step = (training_step == training_steps_max)
            if (training_step % self._model_settings['eval_step_interval']) == 0 or is_last_step:
                set_size = audio_processor.set_size('validation')
                total_accuracy = 0
                total_conf_matrix = None
                for i in xrange(0, set_size, batch_size):
                    validation_batchX, validation_batchY = (
                        audio_processor.get_data(batch_size, i, 'validation'))
                    # Run a validation step and capture training summaries for TensorBoard
                    # with the `merged` op.
                    validation_summary, validation_accuracy, conf_matrix = sess.run(
                        [merged_summaries, self._evaluation_step, self._confusion_matrix],
                        feed_dict={
                            self._batchX: validation_batchX,
                            self._batchY: validation_batchY,
                            self._dropout_prob: 1.0
                        })
                    validation_writer.add_summary(validation_summary, training_step)
                    validation_batch_size = min(batch_size, set_size - i)
                    total_accuracy += (validation_accuracy * validation_batch_size) / set_size
                    if total_conf_matrix is None:
                        total_conf_matrix = conf_matrix
                    else:
                        total_conf_matrix += conf_matrix
                tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
                tf.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' %
                                (training_step, total_accuracy * 100, set_size))

            # Save the model checkpoint periodically.
            if training_step % self._model_settings['save_step_interval'] == 0 or training_step == training_steps_max:
                checkpoint_path = os.path.join(self._model_settings['train_dir'],
                                               self.name + '.ckpt')
                tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
                saver.save(sess, checkpoint_path, global_step=training_step)

        set_size = audio_processor.set_size('testing')
        tf.logging.info('set_size=%d', set_size)
        total_accuracy = 0
        total_conf_matrix = None
        for i in xrange(0, set_size, batch_size):
            test_batchX, test_batchY = audio_processor.get_data(
                batch_size, i, 'testing')
            test_accuracy, conf_matrix = sess.run(
                [self._evaluation_step, self._confusion_matrix],
                feed_dict={
                    self._batchX: test_batchX,
                    self._batchY: test_batchY,
                    self._dropout_prob: 1.0
                })
            test_batch_size = min(batch_size, set_size - i)
            total_accuracy += (test_accuracy * test_batch_size) / set_size
            if total_conf_matrix is None:
                total_conf_matrix = conf_matrix
            else:
                total_conf_matrix += conf_matrix
        tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
        tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (total_accuracy * 100,
                                                                 set_size))

    def predict(self, x):
        sess = tf.get_default_session()
        assert(sess is not None)
        logits = sess.run(self._logits, feed_dict={self._batchX: x,
                                                   self._dropout_prob: 1.0})
        predicted = np.argmax(logits)
        return predicted


class ConvRNN(BaseNN):
    def __init__(self, name, model_settings):
        BaseNN.__init__(self, name, model_settings)
        self._model_settings = model_settings
        self.name = name
        self._logits = None
        self._batchX = None
        self._batchY = None
        self._learning_rate_input = None
        self._cross_entropy_mean = None
        self._train_step = None
        self._confusion_matrix = None
        self._evaluation_step = None
        self._increment_global_step = None
        self._dropout_prob = None
        self.build_forward_pass_graph()
        self.build_train_graph()
        return

    def build_forward_pass_graph(self):
        assert(tf.get_default_session() is not None)
        seq_length = self._model_settings['strip_array_length']
        with tf.name_scope('forward_prop'):
            self._batchX = tf.placeholder(tf.float32, [None, seq_length,
                                                 self._model_settings['strip_window_size_samples'],
                                                 self._model_settings['dct_coefficient_count']], name='batchX')

            # self._batchX = tf.placeholder(tf.float32, [None, self._model_settings['fingerprint_size']], name=self.name + '_batchX')
            num_classes = self._model_settings['label_count']

            num_hidden = self._model_settings['hidden_reccurent_cells_count']

            input_5d = tf.reshape(self._batchX,
                                        [-1, seq_length, self._model_settings['strip_window_size_samples'],
                                         self._model_settings['dct_coefficient_count'], 1])
            first_filter_time_depth = 1
            first_filter_width = 4
            first_filter_height = 20
            first_filter_count = 64
            first_weights = tf.Variable(
                tf.truncated_normal(
                    [first_filter_time_depth, first_filter_height, first_filter_width, 1, first_filter_count],
                    stddev=0.01))
            first_bias = tf.Variable(tf.zeros([first_filter_count]))
            first_conv = tf.nn.conv3d(input_5d, first_weights, [1, 1, 1, 1, 1],
                                      'SAME') + first_bias
            first_relu = tf.nn.relu(first_conv)
            self._dropout_prob = tf.placeholder(dtype=tf.float32, shape=[], name='dropout')
            first_dropout = tf.nn.dropout(first_relu, self._dropout_prob)

            max_pool = tf.nn.max_pool3d(first_dropout, [1, 1, 2, 2, 1], [1, 1, 2, 2, 1], 'SAME')
            max_pool_reshaped = tf.reshape(max_pool, [-1,  max_pool.shape[1],  max_pool.shape[-3] * max_pool.shape[-2] * max_pool.shape[-1]])
            rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=num_hidden, state_is_tuple=True)
            val, state = tf.nn.dynamic_rnn(rnn_cell, max_pool_reshaped, dtype=tf.float32)
            val = tf.transpose(val, [1, 0, 2])
            last = tf.gather(val, int(val.get_shape()[0]) - 1)
            weight = tf.Variable(tf.truncated_normal([num_hidden, num_classes]))
            bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
            self._logits = tf.add(tf.matmul(last, weight), bias, name='logits')


class MultiConvRNN(BaseNN):
    def __init__(self, name, model_settings, restored_model=None):
        BaseNN.__init__(self, name, model_settings)
        self._model_settings = model_settings
        self.name = name
        if restored_model is None:
            self._logits = None
            self._batchX = None
            self._batchY = None
            self._learning_rate_input = None
            self._cross_entropy_mean = None
            self._train_step = None
            self._confusion_matrix = None
            self._evaluation_step = None
            self._increment_global_step = None
            self._dropout_prob = None
            self._state = None
            self._rnn_outputs = None
            self.build_forward_pass_graph()
            self.build_train_graph()
        else:
            self._logits = restored_model['forward_prop/logits']
            self._batchX = restored_model['forward_prop/batchX']
            self._batchY = restored_model['batchY']
            self._learning_rate_input = restored_model['train/learning_rate_input']
            self._cross_entropy_mean = restored_model['cross_entropy/cross_entropy_mean']
            self._train_step = restored_model['train/train_step']
            self._confusion_matrix = restored_model['confusion_matrix/SparseTensorDenseAdd']
            self._evaluation_step = restored_model['evaluation_step']
            self._increment_global_step = restored_model['increment_global_step']
            self._dropout_prob = restored_model['forward_prop/dropout']
            self._state = restored_model['forward_prop/state']
            self._rnn_outputs = restored_model['forward_prop/rnn_outputs']
        #     no need to build graph as its already restored
        return

    @classmethod
    def restore(cls, model_name, meta_path, ckpnt_path):
        sess = tf.get_default_session()
        assert(sess is not None)
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, ckpnt_path)
        graph = tf.get_default_graph()

        list_tensors = [(i.name, i.values()) for i in graph.get_operations()]
        tensors = dict(list_tensors)
        restored_model = tensors
        return cls(model_name, None, restored_model)

    def build_forward_pass_graph(self):
        assert(tf.get_default_session() is not None)
        seq_length = self._model_settings['strip_array_length']
        with tf.name_scope('forward_prop'):
            self._batchX = tf.placeholder(tf.float32, [None, seq_length,
                                                 self._model_settings['strip_window_size_samples'],
                                                 self._model_settings['dct_coefficient_count']], name='batchX')

            # self._batchX = tf.placeholder(tf.float32, [None, self._model_settings['fingerprint_size']], name=self.name + '_batchX')
            num_classes = self._model_settings['label_count']

            num_hidden = self._model_settings['hidden_reccurent_cells_count']

            input_5d = tf.reshape(self._batchX,
                                        [-1, seq_length, self._model_settings['strip_window_size_samples'],
                                         self._model_settings['dct_coefficient_count'], 1])
            first_filter_time_depth = 1
            first_filter_width = 8
            first_filter_height = 10
            first_filter_count = 64
            first_weights = tf.Variable(
                tf.truncated_normal(
                    [first_filter_time_depth, first_filter_height, first_filter_width, 1, first_filter_count],
                    stddev=0.01))
            first_bias = tf.Variable(tf.zeros([first_filter_count]))
            first_conv = tf.nn.conv3d(input_5d, first_weights, [1, 1, 1, 1, 1],
                                      'SAME') + first_bias
            first_relu = tf.nn.relu(first_conv)
            self._dropout_prob = tf.placeholder(dtype=tf.float32, shape=[], name='dropout')
            first_dropout = tf.nn.dropout(first_relu, self._dropout_prob)

            first_max_pool = tf.nn.max_pool3d(first_dropout, [1, 1, 1, 3, 1], [1, 1, 1, 3, 1], 'SAME')

            second_filter_time_depth = 1
            second_filter_width = 4
            second_filter_height = 5
            second_filter_count = 64
            second_weights = tf.Variable(
                tf.truncated_normal(
                    [second_filter_time_depth, second_filter_height, second_filter_width, first_filter_count, second_filter_count],
                    stddev=0.01))
            second_bias = tf.Variable(tf.zeros([second_filter_count]))
            second_conv = tf.nn.conv3d(first_max_pool, second_weights, [1, 1, 1, 1, 1],
                                      'SAME') + second_bias
            second_relu = tf.nn.relu(second_conv)
            second_dropout = tf.nn.dropout(second_relu, self._dropout_prob)

            # second_max_pool = tf.nn.max_pool3d(second_dropout, [1, 1, 2, 2, 1], [1, 1, 2, 2, 1], 'SAME')
            second_max_pool = second_dropout

            max_pool_reshaped = tf.reshape(second_max_pool, [-1,
                                                             second_max_pool.shape[1],
                                                             second_max_pool.shape[-3] * second_max_pool.shape[-2] * second_max_pool.shape[-1]])
            rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=num_hidden, state_is_tuple=True, name='rnn_cell')
            val, state = tf.nn.dynamic_rnn(rnn_cell, max_pool_reshaped, dtype=tf.float32)
            self._state = tf.identity(state, name='state')
            self._rnn_outputs = tf.identity(val, name='rnn_outputs')
            val = tf.transpose(val, [1, 0, 2])
            last = tf.gather(val, int(val.get_shape()[0]) - 1)
            weight = tf.Variable(tf.truncated_normal([num_hidden, num_classes]))
            bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
            self._logits = tf.add(tf.matmul(last, weight), bias, name='logits')

    def get_hidden_state(self, x):
        sess = tf.get_default_session()
        assert (sess is not None)
        outputs, state = sess.run([self._rnn_outputs, self._state], feed_dict={self._batchX: x,
                                                   self._dropout_prob: 1.0})
        return outputs, state


class Conv2dNN(BaseNN):
    def __init__(self, name, model_settings):
        BaseNN.__init__(self, name, model_settings)
        # self._logits = None
        # self._batchX = None
        # self._batchY = None
        # self._learning_rate_input = None
        # self._cross_entropy_mean = None
        # self._train_step = None
        # self._confusion_matrix = None
        # self._evaluation_step = None
        # self._increment_global_step = None
        # self._dropout_prob = None
        self.build_forward_pass_graph()
        BaseNN.build_train_graph(self)
        return

    def build_forward_pass_graph(self):
        assert(tf.get_default_session() is not None)
        seq_length = self._model_settings['strip_array_length']
        with tf.name_scope('forward_prop'):
            self._batchX = tf.placeholder(tf.float32, [None, seq_length,
                                                 self._model_settings['strip_window_size_samples'],
                                                 self._model_settings['dct_coefficient_count']], name=self.name + '_batchX')

            # self._batchX = tf.placeholder(tf.float32, [None, self._model_settings['fingerprint_size']],
            #                               name=self.name + '_batchX')

            num_classes = self._model_settings['label_count']

            input_4d = tf.reshape(self._batchX,
                                        [-1, seq_length * self._model_settings['strip_window_size_samples'],
                                         self._model_settings['dct_coefficient_count'], 1])

            # input_4d = tf.reshape(self._batchX,
            #                       [-1, self._model_settings['spectrogram_length'],
            #                        self._model_settings['dct_coefficient_count'], 1])

            first_filter_width = 8
            first_filter_height = 20
            first_filter_count = 64
            first_weights = tf.Variable(
                tf.truncated_normal(
                    [first_filter_height, first_filter_width, 1, first_filter_count],
                    stddev=0.01))
            first_bias = tf.Variable(tf.zeros([first_filter_count]))
            first_conv = tf.nn.conv2d(input_4d, first_weights, [1, 1, 1, 1],
                                      'SAME') + first_bias
            first_relu = tf.nn.relu(first_conv)
            self._dropout_prob = tf.placeholder(dtype=tf.float32, shape=[])
            first_dropout = tf.nn.dropout(first_relu, self._dropout_prob)
            max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
            second_filter_width = 4
            second_filter_height = 10
            second_filter_count = 64
            second_weights = tf.Variable(
                tf.truncated_normal(
                    [
                        second_filter_height, second_filter_width, first_filter_count,
                        second_filter_count
                    ],
                    stddev=0.01))
            second_bias = tf.Variable(tf.zeros([second_filter_count]))
            second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1],
                                       'SAME') + second_bias
            second_relu = tf.nn.relu(second_conv)
            second_dropout = tf.nn.dropout(second_relu, self._dropout_prob)
            second_conv_shape = second_dropout.get_shape()
            second_conv_output_width = second_conv_shape[2]
            second_conv_output_height = second_conv_shape[1]
            second_conv_element_count = int(
                second_conv_output_width * second_conv_output_height *
                second_filter_count)
            flattened_second_conv = tf.reshape(second_dropout,
                                               [-1, second_conv_element_count])

            first_fc_num = 200
            first_fc_weights = tf.Variable(
                tf.truncated_normal(
                    [second_conv_element_count, first_fc_num], stddev=0.01))
            first_fc_bias = tf.Variable(tf.zeros([first_fc_num]))
            first_fc = tf.matmul(flattened_second_conv, first_fc_weights) + first_fc_bias

            final_fc_weights = tf.Variable(
                tf.truncated_normal(
                    [first_fc_num, num_classes], stddev=0.01))
            final_fc_bias = tf.Variable(tf.zeros([num_classes]))
            final_fc = tf.matmul(first_fc, final_fc_weights) + final_fc_bias
            self._logits = final_fc


