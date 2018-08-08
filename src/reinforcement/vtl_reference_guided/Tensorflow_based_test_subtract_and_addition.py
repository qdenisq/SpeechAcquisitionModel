from random import seed
from random import randint
from random import random as rnd
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
import keras
from keras import backend as K
from keras.layers import LSTM
from keras.utils import plot_model
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import numpy as np
import tensorflow as tf

ds_norm_largest = 1000.
ds_largest = 1000.
largest = 1000.


def generate_model_dynamics_training_data(n_examples):
    s0 = np.array([[(rnd() - 0.5) * 1. * largest for n in range(n_dim)] for _ in range(n_examples)])
    ds = np.array([[(rnd() - 0.5) * 1. * ds_largest for n in range(n_dim)] for _ in range(n_examples)])
    t = s0 + ds
    a = np.array([[(rnd() - 0.5) * 1. * ds_largest for n in range(n_dim)] for _ in range(n_examples)])

    # real model dynamics
    s1 = s0 + a

    # normalise staff
    s0 = normalize(s0, largest)
    t = normalize(t, largest)
    a = normalize(a, ds_norm_largest)
    s1 = normalize(s1, largest)

    X_md = np.concatenate((s0, a), axis=1)
    Y_md = s1

    return X_md, Y_md


def generate_policy_training_data(n_examples):
    s0 = np.array([[(rnd() - 0.5) * 1. * largest for n in range(n_dim)] for _ in range(n_examples)])
    ds = np.array([[(rnd() - 0.5) * 1. * ds_largest for n in range(n_dim)] for _ in range(n_examples)])
    t = s0 + ds
    # normalise staff
    s0 = normalize(s0, largest)
    t = normalize(t, largest)

    X = np.concatenate((s0, t), axis=1)

    a = ds
    a = normalize(a, ds_norm_largest)

    return X, s0, t, a


# invert normalization
def denormalize(value, largest):
    return value * float(largest)


# apply normalization
def normalize(value, largest):
    return value / largest

# define data parameters
seed(1)
n_examples = 10000
n_dim = 20
# define training options
n_batch = 200
n_epoch = 50


################################################################
# Model Dynamics
################################################################

# create model dynamics

# specify input placeholders
md_input = tf.placeholder(tf.float32, [None, n_dim * 2])

# specify network structure
md_initializer = tf.glorot_uniform_initializer()
md_dense1 = tf.layers.dense(inputs=md_input, units=4 * n_dim, kernel_initializer=md_initializer)
md_dense2 = tf.layers.dense(inputs=md_dense1, units=2 * n_dim, kernel_initializer=md_initializer)

# specify output
md_output = tf.layers.dense(inputs=md_dense2, units=1 * n_dim, kernel_initializer=md_initializer)

# network target for training
y = tf.placeholder(tf.float32, [None, n_dim * 1])

# loss
loss = tf.losses.mean_squared_error(y, md_output)

# train step
global_step = tf.train.create_global_step()
train_step = tf.train.AdadeltaOptimizer(learning_rate=1.0).minimize(loss=loss, global_step=global_step)

# Open the session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# train loop
for _ in range(n_epoch):
    # generate new dataset for each epoch
    X, Y = generate_model_dynamics_training_data(n_examples)

    for i in range(n_examples // n_batch):
        # get next batch
        x_batch = X[i * n_batch : (i + 1) * n_batch]
        y_batch = Y[i * n_batch: (i + 1) * n_batch]

        # train step
        feed_dict = {md_input: x_batch, y: y_batch}
        train_step.run(feed_dict=feed_dict)

    # calculate loss on the whole dataset
    y_pred = sess.run(md_output, feed_dict={md_input:X})
    total_loss = np.mean(np.sum(np.square(Y - y_pred), axis=1), axis=0)
    y_pred_denormed = denormalize(y_pred, largest)
    y_denormed = denormalize(Y, largest)
    total_loss_denormed = np.mean(np.sum(np.square(y_denormed - y_pred_denormed), axis=1), axis=0)
    print('|epoch : {}| model dynamics loss: {:.10f}| unnormalized loss: {:.7f}'.format(_, total_loss, total_loss_denormed))

# Final evaluation on some new patterns
X, Y = generate_model_dynamics_training_data(n_examples)
y_pred = sess.run(md_output, feed_dict={md_input:X})
# calculate error
expected = denormalize(Y, largest)
predicted = denormalize(y_pred, largest)
rmse = sqrt(mean_squared_error(expected, predicted))
print('RMSE: %f' % rmse)
# show some examples
for i in range(5):
    error = expected[i] - predicted[i]
    print('Expected={}, Predicted={} (err={})'.format(expected[i], predicted[i], error))