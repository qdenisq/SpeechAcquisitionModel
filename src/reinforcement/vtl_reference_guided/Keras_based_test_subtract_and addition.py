from random import seed
from random import randint
from random import random as rnd
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
import keras
from keras import backend as K
from keras.layers import LSTM
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import numpy as np
import tensorflow as tf

ds_norm_largest = 1000.
ds_largest = 1000.
largest = 1000.


def generate_model_dynamics_training_data(n_examples, policy):
    s0 = np.array([[(rnd() - 0.5) * 1. * largest for n in range(n_dim)] for _ in range(n_examples)])
    ds = np.array([[(rnd() - 0.5) * 1. * ds_largest for n in range(n_dim)] for _ in range(n_examples)])
    t = s0 + ds
    a = np.array([[(rnd() - 0.5) * 1. * ds_largest for n in range(n_dim)] for _ in range(n_examples)])
    #
    # X_policy_unnormed = np.concatenate((s0, t), axis=1)
    # X_policy = normalize(X_policy_unnormed, largest)
    # a_normed = policy.predict(X_policy) * 0.1
    # a = denormalize(a_normed, largest)


    # real model dynamics
    s1 = s0 + a
    # s1 = np.clip(s1, -largest, largest)

    # normalise staff
    s0 = normalize(s0, largest)
    t = normalize(t, largest)
    a = normalize(a, ds_largest)
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


sess = tf.Session()

# generate training data
seed(1)
n_examples = 10000
n_dim = 20
# define LSTM configuration
n_batch = 200
n_epoch = 50
# create model dynamics
model_dynamics = Sequential()
model_dynamics.add(Dense(4 * n_dim, input_dim=2 * n_dim))
model_dynamics.add(Dense(2 * n_dim))
model_dynamics.add(Dense(1 * n_dim))
model_dynamics.compile(loss='mean_squared_error', optimizer='adadelta')


# Create policy network
policy_initializer = keras.initializers.RandomUniform(minval=-largest/ds_largest, maxval=largest/ds_largest)
policy_initializer = keras.initializers.glorot_uniform()

policy = Sequential()
policy.add(Dense(4 * n_dim, input_dim=2 * n_dim, kernel_initializer=policy_initializer))
policy.add(Dense(2 * n_dim, kernel_initializer=policy_initializer))
policy.add(Dense(1 * n_dim, kernel_initializer=policy_initializer))
policy.compile(loss='mean_squared_error', optimizer='adadelta')


# action gradients
target = tf.placeholder(tf.float32, [None, n_dim])
a_grads = tf.gradients(tf.reduce_mean(tf.square(target - model_dynamics.output)), model_dynamics.input)
# train model dynamics
for _ in range(n_epoch):
    X, y = generate_model_dynamics_training_data(n_examples, policy)
    X_flat = np.reshape(X, [n_examples, n_dim * 2])
    model_dynamics.fit(X_flat, y, epochs=1, batch_size=n_batch, verbose=2)
# evaluate on some new patterns
X, y = generate_model_dynamics_training_data(n_examples, policy)
X_flat = np.reshape(X, [n_examples, n_dim * 2])
result = model_dynamics.predict(X_flat, batch_size=n_batch, verbose=0)
# calculate error
expected = [denormalize(x, largest) for x in y]
predicted = [denormalize(x, largest) for x in result]
rmse = sqrt(mean_squared_error(expected, predicted))
print('RMSE: %f' % rmse)
# show some examples
for i in range(20):
    error = expected[i] - predicted[i]
    print('Expected={}, Predicted={} (err={})'.format(expected[i], predicted[i], error))



# train policy

output_l = model_dynamics.output
target = tf.placeholder(tf.float32, [None, n_dim])

loss = K.mean(K.square(output_l - target), axis=-1)
grads = K.gradients(loss, model_dynamics.input)
a_grads_op = grads[0][:, n_dim:]
opt = tf.train.AdadeltaOptimizer()
# opt = keras.optimizers.get('adadelta')
# policy_grads = tf.gradients(policy.output, policy.trainable_weights, grad_ys=a_grads_op)
policy_grads = opt.compute_gradients(policy.output, policy.trainable_weights, grad_loss=a_grads_op)
# grads_and_vars = zip(policy_grads, policy.trainable_weights)
grads_and_vars = policy_grads
optimize_op = opt.apply_gradients(grads_and_vars)

# test loss and grads
test_expected_a = tf.placeholder(tf.float32, [None, n_dim])
policy_output = policy.output
test_loss = K.mean(K.square(policy_output - test_expected_a), axis=-1)
test_grads = K.gradients(test_loss, policy_output)
test_a_grads_op = test_grads[0][:, :]
test_policy_grads = opt.compute_gradients(policy_output, policy.trainable_weights, grad_loss=test_a_grads_op)
test_optimize_op = opt.apply_gradients(test_policy_grads)


for _ in range(n_epoch*100):
    X, s0, t, expected_a = generate_policy_training_data(n_examples)

    # policy.fit(X, expected_a, epochs=1, batch_size=n_batch, verbose=2)
    #
    # custom optimization of policy
    y_pred = policy.predict(X)
    X_md = np.concatenate((s0, y_pred), axis=1)


    sess = K.get_session()
    for k in range(n_batch, n_examples, n_batch):
        optimize, grads_out, test_grads_out, policy_grads_out, test_policy_grads_out \
            = sess.run([optimize_op, a_grads_op, test_a_grads_op, policy_grads, test_policy_grads], feed_dict={model_dynamics.input: X_md[k - n_batch: k,:],
                                                                             policy.input: X[k - n_batch: k,:],
                                                                             target: t[k - n_batch: k,:],
                                                                            test_expected_a: expected_a[k - n_batch: k,:]
                                                    })

        # test_optimize = sess.run([test_optimize_op, a_grads_op, test_a_grads_op, policy_grads, test_policy_grads], feed_dict={model_dynamics.input: X_md[k - n_batch: k,:],
        #                                                                      policy.input: X[k - n_batch: k,:],
        #                                                                      target: t[k - n_batch: k,:],
        #                                                                     test_expected_a: expected_a[k - n_batch: k,:]
        #                                             })

    # policy loss
    expected = denormalize(expected_a, ds_norm_largest)
    predicted = denormalize(y_pred, ds_norm_largest)
    policy_loss = sqrt(mean_squared_error(expected, predicted))
    # policy_loss = np.mean(np.sum(np.square(true_actions - y_pred_unnormed), axis=1), axis=0)
    print('|epoch : {}| policy loss: {}|'.format(_, policy_loss))

# evaluate on some new patterns
X, s0, t, a = generate_policy_training_data(n_examples)
y_pred = policy.predict(X, batch_size=n_batch, verbose=0)

y_pred_unnormed = denormalize(y_pred, ds_norm_largest)
expected = denormalize(a, ds_norm_largest)
predicted = y_pred_unnormed

# expected = a
# predicted = y_pred


policy_loss = sqrt(mean_squared_error(expected, predicted))
print('RMSE: {}', policy_loss)
# show some examples
for i in range(20):
    error = expected[i] - predicted[i]
    print('Expected={}, Predicted={} (err={})'.format(expected[i], predicted[i], error))

