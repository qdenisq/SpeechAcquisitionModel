import datetime
import os
import pickle
import random
from collections import deque
from random import randrange

import math
import numpy as np
from scipy import spatial
import tensorflow as tf
from tensorflow.python.keras.initializers import RandomUniform
from tensorflow.python.keras.layers import Dense, Input, BatchNormalization, Activation, Concatenate
import keras


from src.VTL.vtl_environment import VTLEnv


class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s0, g0, a, s1, g1, target):
        if self.count < self.buffer_size:
            self.buffer.append((s0, g0, a, s1, g1, target))
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append((s0, g0, a, s1, g1, target))

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s0_batch = np.array([_[0] for _ in batch])
        g0_batch = np.array([_[1] for _ in batch])
        a_batch = np.array([_[2] for _ in batch])
        s1_batch = np.array([_[3] for _ in batch])
        g1_batch = np.array([_[4] for _ in batch])
        target_batch = np.array([_[5] for _ in batch])

        return s0_batch, g0_batch, a_batch, s1_batch, g1_batch, target_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0
# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.003, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class Policy(object):
    """
    Input to the network is the current state and the goal state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, name, model_settings):
        self.sess = tf.get_default_session()
        assert (self.sess is not None)
        self.name = name
        self.s_dim = model_settings['state_dim']
        self.state_bound = model_settings['state_bound']
        self.g_dim = model_settings['goal_dim']
        self.goal_bound = model_settings['goal_bound']
        self.a_dim = model_settings['action_dim']
        self.action_bound = model_settings['action_bound']
        self.learning_rate = model_settings['actor_learning_rate']
        self.tau = model_settings['actor_tau']
        self.batch_size = model_settings['minibatch_size']

        y_max = [y[1] for y in self.action_bound]
        y_min = [y[0] for y in self.action_bound]
        self._k = 2. / (np.subtract(y_max, y_min))
        self._b = 1. - np.array(y_max) * self._k

        y_max = [y[1] for y in self.state_bound]
        y_min = [y[0] for y in self.state_bound]
        self._k_s = 2. / (np.subtract(y_max, y_min))
        self._b_s = -0.5 * np.add(y_max, y_min) * self._k_s

        # Actor Network
        with tf.variable_scope(self.name + '_policy'):
            self.inputs_state,\
            self.inputs_goal,\
            self.inputs_target,\
            self.out,\
            self.scaled_out\
                = self.create_policy_network()
            self.network_params = tf.trainable_variables(scope=self.name + '_policy')

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, self.action_gradient)
        self.actor_gradients = list(
            map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))
        #     # Combine the gradients here
        # self.actor_gradients = \
        #     tf.train.AdamOptimizer(self.learning_rate).compute_gradients(self.scaled_out,
        #                                                                             self.network_params,
        #                                                                             grad_loss=self.action_gradient)
        #
        # # Optimization Op
        # self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
        #     apply_gradients(self.actor_gradients)



        self.ground_truth_actions = tf.placeholder(tf.float32, [None, self.a_dim])
        self.se_loss = tf.losses.mean_squared_error(self.ground_truth_actions, self.scaled_out)
        self.se_loss1 = tf.reduce_mean(tf.squared_difference(self.ground_truth_actions ,self.scaled_out))
        self.grads1 = tf.gradients(self.se_loss1, self.scaled_out)
        self.actor_gradients1 = \
            tf.train.AdamOptimizer(self.learning_rate).compute_gradients(self.scaled_out,
                                                                                    self.network_params,
                                                                                    grad_loss=self.grads1[0])
        # Optimization Op

        self.optimize1 = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(self.actor_gradients1)

        self.optimize_1 = tf.train.AdamOptimizer(self.learning_rate).minimize(self.se_loss1)

        self.ground_truth_goal_out = tf.placeholder(tf.float32, [None, self.s_dim])
        self.next_state = tf.add(self.inputs_state, self.scaled_out)
        y_max = np.tile([y[1] for y in self.state_bound], (self.batch_size, 1))
        y_min = np.tile([y[0] for y in self.state_bound], (self.batch_size, 1))
        self.next_state = tf.clip_by_value(self.next_state, y_min, y_max)
        self.goal_loss = tf.losses.mean_squared_error(self.ground_truth_goal_out, self.next_state)
        self.optimize_2 = tf.train.AdamOptimizer(0.00001).minimize(self.goal_loss)


        self.num_trainable_vars = len(self.network_params)

    def create_policy_network(self):
        state_x = tf.placeholder(tf.float32, [None, self.s_dim])
        target_x = tf.placeholder(tf.float32, [None, self.g_dim])
        goal_x = tf.placeholder(tf.float32, [None, self.g_dim])

        net = tf.concat([state_x, target_x, goal_x], axis=1)
        self.conc = net


        #
        # net = Dense(4 * self.s_dim, input_dim=3 * self.s_dim)(net)
        # net = tf.nn.relu(net)
        # net = Dense(2 * self.s_dim)(net)
        # net = tf.nn.relu(net)
        # net = Dense(1 * self.s_dim, kernel_initializer=keras.initializers.RandomUniform(minval=-0.005, maxval=0.005))(net)
        # net = tf.nn.relu(net)
        # net = Dense(1 * self.s_dim, kernel_initializer=keras.initializers.RandomUniform(minval=-0.005, maxval=0.005))(net)

        n_hidden_0 = 128
        w0 = tf.Variable(tf.random_normal([net.get_shape().as_list()[1], n_hidden_0], stddev=0.001))
        b0 = tf.Variable(tf.random_normal([n_hidden_0]))
        net = tf.add(tf.matmul(net, w0), b0)
        net = tf.nn.tanh(net)

        n_hidden_1 = self.a_dim
        w1 = tf.Variable(tf.random_normal([net.get_shape().as_list()[1], n_hidden_1], stddev=0.001))
        b1 = tf.Variable(tf.random_normal([n_hidden_1], stddev=0.001))
        net = tf.add(tf.matmul(net, w1), b1)
        net = tf.nn.tanh(net)
        #
        # n_out = self.a_dim
        # w2 = tf.Variable(tf.random_normal([net.get_shape().as_list()[1], n_out], stddev=0.001))
        # b2 = tf.Variable(tf.random_normal([n_out], stddev=0.001))
        # net = tf.add(tf.matmul(net, w2), b2)

        action_y = net
        action_y_scaled_out = action_y
        return state_x, goal_x, target_x, action_y, action_y_scaled_out

    def train(self, inputs_state, inputs_goal, inputs_target, a_gradient):
        return self.sess.run([self.optimize], feed_dict={
                self.inputs_state: inputs_state,
                self.inputs_goal: inputs_goal,
                self.inputs_target: inputs_target,
                self.action_gradient: a_gradient,
            })

    def train_1(self, inputs_state, inputs_goal, inputs_target, ground_truth_actions):
        return self.sess.run([self.optimize1, self.se_loss1], feed_dict={
            self.inputs_state: inputs_state,
            self.inputs_goal: inputs_goal,
            self.inputs_target: inputs_target,
            self.ground_truth_actions: ground_truth_actions,
        })

    def train_2(self, inputs_state, inputs_goal, inputs_target):
        return self.sess.run([self.goal_loss, self.optimize_2], feed_dict={
            self.inputs_state: inputs_state,
            self.inputs_goal: inputs_goal,
            self.inputs_target: inputs_target,
            self.ground_truth_goal_out: inputs_target,
        })

    def predict(self, inputs_state, inputs_goal, inputs_target):
        return self.sess.run([self.scaled_out], feed_dict={
            self.inputs_state: inputs_state,
            self.inputs_goal: inputs_goal,
            self.inputs_target: inputs_target
        })

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class ModelDynamics(object):
    """
    Input to the network is the current state and the goal state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, name, model_settings):
        self.sess = tf.get_default_session()
        assert (self.sess is not None)
        self.name = name
        self.s_dim = model_settings['state_dim']
        self.state_bound = model_settings['state_bound']
        self.g_dim = model_settings['goal_dim']
        self.goal_bound = model_settings['goal_bound']
        self.a_dim = model_settings['action_dim']
        self.action_bound = model_settings['action_bound']
        self.learning_rate = model_settings['model_dynamics_learning_rate']
        self.tau = model_settings['actor_tau']
        self.batch_size = model_settings['minibatch_size']
        self.state_goal_gamma = 0.5
        self.se_cos_gamma = 1.0

        y_max = [y[1] for y in self.state_bound]
        y_min = [y[0] for y in self.state_bound]
        self._k_state = 2. / (np.subtract(y_max, y_min))
        self._b_state = 1. - np.array(y_max) * self._k_state

        y_max = [y[1] for y in self.goal_bound]
        y_min = [y[0] for y in self.goal_bound]
        self._k_goal = 2. / (np.subtract(y_max, y_min))
        self._b_goal = 1. - np.array(y_max) * self._k_goal
        # Model Dynamics Network
        with tf.variable_scope(self.name + '_model_dynamics'):
            self.inputs_state,\
            self.inputs_goal,\
            self.inputs_action,\
            self.state_out,\
            self.scaled_state_out,\
            self.goal_out,\
            self.scaled_goal_out\
                = self.create_model_dynamics_network()
            self.network_params = tf.trainable_variables(scope=self.name + '_model_dynamics')

        self.ground_truth_state_out = tf.placeholder(tf.float32, [None, self.s_dim])
        self.ground_truth_goal_out = tf.placeholder(tf.float32, [None, self.g_dim])
        self.ground_truth_out = Concatenate()([self.ground_truth_state_out, self.ground_truth_goal_out])

        self.scaled_out = Concatenate()([self.scaled_state_out, self.scaled_goal_out])

        # Loss Ops
        self.state_mse_loss = tf.losses.mean_squared_error(self.ground_truth_state_out, self.scaled_state_out)
        self.state_cos_loss = tf.losses.cosine_distance(tf.nn.l2_normalize(self.ground_truth_state_out - self.inputs_state, axis=1),
                                                  tf.nn.l2_normalize(self.scaled_state_out - self.inputs_state, axis=1),
                                                  axis=1)
        self.state_loss = tf.add(self.state_mse_loss * self.se_cos_gamma, self.state_cos_loss * (1. - self.se_cos_gamma))

        self.goal_mse_loss = tf.losses.mean_squared_error(self.ground_truth_goal_out, self.scaled_goal_out)
        self.goal_cos_loss = tf.losses.cosine_distance(
            tf.nn.l2_normalize(self.ground_truth_goal_out - self.inputs_goal, axis=1),
            tf.nn.l2_normalize(self.scaled_goal_out - self.inputs_goal, axis=1),
            axis=1)
        self.goal_loss = tf.add(self.goal_mse_loss * self.se_cos_gamma, self.goal_cos_loss * (1. - self.se_cos_gamma))

        # self.loss = tf.add(self.state_loss * self.state_goal_gamma, self.goal_loss * (1. - self.state_goal_gamma))
        self.loss = self.state_loss + self.goal_loss

        # Optimize op
        self.optimize = tf.train.GradientDescentOptimizer(
            self.learning_rate).minimize(self.loss)

        # Acion gradients extraction
        self.policy_goal_loss = tf.reduce_mean(tf.squared_difference(self.ground_truth_goal_out, self.scaled_goal_out))
        self.action_grads = tf.gradients(self.policy_goal_loss, self.inputs_action)
        # Real action gradients
        self.ground_truth_actions = tf.placeholder(tf.float32, [None, self.a_dim])
        self.policy_action_loss = tf.reduce_mean(tf.squared_difference(self.ground_truth_actions, self.inputs_action))
        self.ground_truth_action_grads = tf.gradients(self.policy_action_loss, self.inputs_action)


        self.num_trainable_vars = len(self.network_params)

    def create_model_dynamics_network(self):
        state_x = tf.placeholder(tf.float32, [None, self.s_dim])
        action_x = tf.placeholder(tf.float32, [None, self.a_dim])
        goal_x = tf.placeholder(tf.float32, [None, self.g_dim])

        net = tf.concat([state_x, action_x, goal_x], axis=1)
        self.conc = net


        s_net = Dense(4 * self.s_dim, input_dim=3 * self.s_dim)(net)
        s_net = Dense(2 * self.s_dim)(s_net)
        s_net = Dense(1 * self.s_dim)(s_net)

        #
        # n_hidden_0 = 128
        # w0 = tf.Variable(tf.random_normal([net.get_shape().as_list()[1], n_hidden_0]))
        # b0 = tf.Variable(tf.random_normal([n_hidden_0]))
        # net = tf.add(tf.matmul(net, w0), b0)
        # net = tf.nn.tanh(net)
        #
        # n_hidden_1 = 64
        # w1 = tf.Variable(tf.random_normal([net.get_shape().as_list()[1], n_hidden_1]))
        # b1 = tf.Variable(tf.random_normal([n_hidden_1]))
        # s_net = tf.add(tf.matmul(net, w1), b1)
        # s_net = tf.nn.tanh(s_net)
        #
        # n_out = self.s_dim
        # w2 = tf.Variable(tf.random_normal([s_net.get_shape().as_list()[1], n_out]))
        # b2 = tf.Variable(tf.random_normal([n_out]))
        # s_net = tf.add(tf.matmul(s_net, w2), b2)

        state_y = s_net
        state_y_scaled = state_y

        # n_hidden_1 = 64
        # w3 = tf.Variable(tf.random_normal([net.get_shape().as_list()[1], n_hidden_1]))
        # b3 = tf.Variable(tf.random_normal([n_hidden_1]))
        # g_net = tf.add(tf.matmul(net, w3), b3)
        # g_net = tf.nn.tanh(g_net)
        #
        # n_out = self.s_dim
        # w4 = tf.Variable(tf.random_normal([g_net.get_shape().as_list()[1], n_out]))
        # b4 = tf.Variable(tf.random_normal([n_out]))
        # g_net = tf.add(tf.matmul(g_net, w4), b4)
        #

        g_net = Dense(4 * self.s_dim, input_dim=3 * self.s_dim)(net)
        g_net = Dense(2 * self.s_dim)(g_net)
        g_net = Dense(1 * self.s_dim)(g_net)

        goal_y = g_net
        goal_y_scaled = goal_y
        return state_x, goal_x, action_x, state_y, state_y_scaled, goal_y, goal_y_scaled

    def calc_loss(self, inputs_state, inputs_goal, inputs_action, ground_truth_state_out, ground_truth_goal_out):
        return self.sess.run([self.loss, self.goal_loss, self.state_mse_loss, self.state_cos_loss, self.scaled_state_out], feed_dict={
                self.inputs_state: inputs_state,
                self.inputs_action: inputs_action,
                self.inputs_goal: inputs_goal,
                self.ground_truth_state_out: ground_truth_state_out,
                self.ground_truth_goal_out: ground_truth_goal_out,
        })

    def train(self, inputs_state, inputs_goal, inputs_action, ground_truth_state_out, ground_truth_goal_out):
        return self.sess.run([self.optimize, self.loss, self.goal_loss, self.state_mse_loss, self.state_cos_loss, self.scaled_state_out], feed_dict={
                self.inputs_state: inputs_state,
                self.inputs_action: inputs_action,
                self.inputs_goal: inputs_goal,
                self.ground_truth_state_out: ground_truth_state_out,
                self.ground_truth_goal_out: ground_truth_goal_out,
        })

    def action_gradients(self, inputs_state, inputs_goal, inputs_action, target_goal_out, ground_truth_actions):
        return self.sess.run([self.action_grads, self.ground_truth_action_grads], feed_dict={
            self.inputs_state: inputs_state,
            self.inputs_action: inputs_action,
            self.inputs_goal: inputs_goal,
            self.ground_truth_goal_out: target_goal_out,
            self.ground_truth_actions: ground_truth_actions
        })

    def predict(self, inputs_state, inputs_goal, inputs_action):
        return self.sess.run([self.scaled_state_out, self.scaled_goal_out], feed_dict={
            self.inputs_state: inputs_state,
            self.inputs_goal: inputs_goal,
            self.inputs_action: inputs_action
        })

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


def build_summaries():
    step_loss = tf.Variable(0.)
    tf.summary.scalar("Per step actor loss", step_loss)

    # episode_ave_max_q = tf.Variable(0.)
    # tf.summary.scalar("Qmax Value", episode_ave_max_q)
    #
    # actor_ep_loss = tf.Variable(0.)
    # tf.summary.scalar("Actor loss", actor_ep_loss)

    model_dynamics_loss = tf.Variable(0.)
    tf.summary.scalar("Model dynamics loss", model_dynamics_loss)


    model_dynamics_goal_loss = tf.Variable(0.)
    tf.summary.scalar("Model dynamics goal loss", model_dynamics_goal_loss)


    # act_grads = tf.placeholder(dtype=tf.float32, shape=None)
    # tf.summary.histogram('action_gradients', act_grads)
    #
    actor_grads = tf.placeholder(dtype=tf.float32, shape=None)
    tf.summary.histogram('actor_gradients', actor_grads)
    #
    # actor_activations = tf.placeholder(dtype=tf.float32, shape=None)
    # tf.summary.histogram('actor_activations', actor_activations)


    variables = [v for v in tf.trainable_variables()]
    [tf.summary.histogram(v.name, v) for v in variables]

    summary_vars = [step_loss, model_dynamics_loss, model_dynamics_goal_loss ,actor_grads]
    # episode_reward, episode_ave_max_q, actor_ep_loss, critic_loss, avg_action, act_grads, actor_grads, actor_activations]

    summary_vars.extend(variables)
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


class real_dynamics(object):
    def __init__(self, settings):
        with tf.variable_scope('real_model_dynamics'):
            self.action_x = Input(batch_shape=[None, settings['action_dim']])
            self.state_x = Input(batch_shape=[None, settings['state_dim']])
            self.ground_truth_goal_out = tf.placeholder(tf.float32, [None, settings['state_dim']])
            self.next_state = tf.add(self.state_x, self.action_x)
            y_max = np.tile([y[1] for y in settings['state_bound']], (settings['minibatch_size'], 1))
            y_min = np.tile([y[0] for y in settings['state_bound']], (settings['minibatch_size'], 1))
            self.next_state = tf.clip_by_value(self.next_state, y_min, y_max)
            self.goal_loss = tf.losses.mean_squared_error(self.ground_truth_goal_out, self.next_state)

            # self.actor_obj = tf.abs(tf.subtract(self.ground_truth_goal_out, self.scaled_goal_out))
            # self.action_grads = tf.gradients(self.goal_loss, self.action_x)
            self.action_grads = \
                tf.gradients(self.goal_loss, self.action_x)

    def action_gradients(self, action, state, ground_truth):
        sess = tf.get_default_session()
        return sess.run([self.action_grads, self.goal_loss], feed_dict={
            self.action_x: action,
            self.state_x: state,
            self.ground_truth_goal_out: ground_truth
        })


def normalize(data, bound):
    y_max = [y[1] for y in bound]
    y_min = [y[0] for y in bound]
    k = np.subtract(y_max, y_min) / 2.
    b = np.add(y_max, y_min) / 2.
    normed_data = (data - b) / k
    normed_data = data / abs(np.subtract(y_max, y_min))
    return normed_data


def denormalize(normed_data, bound):
    y_max = [y[1] for y in bound]
    y_min = [y[0] for y in bound]
    k = np.subtract(y_max, y_min) / 2.
    b = np.add(y_max, y_min) / 2.
    data = normed_data * k + b
    data = normed_data * abs(np.subtract(y_max, y_min))
    return data


def train(settings, policy, model_dynamics, env, replay_buffer, reference_trajectory):
    sess = tf.get_default_session()
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    dt = str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p"))
    writer = tf.summary.FileWriter(settings['summary_dir'] + '/summary_md_' + dt, sess.graph)
    video_dir = settings['videos_dir'] + '/video_md_' + dt
    try:
        os.makedirs(video_dir)
    except:
        print("directory '{}' already exists")

    num_episodes = 50000
    action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(env.action_dim), sigma=0.001)
    s_dim = settings['state_dim']
    g_dim = settings['goal_dim']
    a_dim = settings['action_dim']
    s_bound = settings['state_bound']
    a_bound = settings['action_bound']
    g_bound = settings['goal_bound']

    for i in range(num_episodes):
        # pick random initial state from the reference trajectory
        s0_index = randrange(0, reference_trajectory.shape[0] - 1)
        if i % 200 == 0:
            s0_index = 0

        s0 = reference_trajectory[s0_index]
        g0 = s0
        env.reset(s0)

        r = []
        # rollout episode
        for j in range(s0_index, len(reference_trajectory) - 1):
            target = reference_trajectory[j + 1]
            # normalize data before feeding it to policy
            s0_normed = normalize(np.reshape(s0, (1, s_dim)), s_bound)
            g0_normed = normalize(np.reshape(g0, (1, g_dim)), g_bound)
            target_normed = normalize(np.reshape(target, (1, g_dim)), g_bound)
            action_normed = policy.predict(s0_normed, g0_normed, target_normed)
            # add noise
            a_noise = action_noise()
            if i % 200 == 0:
                action_normed += a_noise
            else:
                action_normed += a_noise

            # denormalize action
            action = denormalize(action_normed, settings['action_bound'])
            action = np.reshape(action, (a_dim))
            # make a step
            s1, _ = env.step(action)
            g1 = s1

            env.render()

            # calc reward
            g1_normed = normalize(g1, g_bound)
            tar_normed = normalize(target, g_bound)
            miss = abs(tar_normed - g1_normed)

            last_loss = np.linalg.norm(target - g1)

            r.append( -1. * np.linalg.norm(target - g1))

            if i % 200 != 0:
                replay_buffer.add(s0, g0, action, s1, g1, target)
            s0 = s1
            g0 = g1

            if max(miss) > 0.2 and i % 200 != 0:
                break
        if i % 200 == 0:
            fname = video_dir + '/episode_' + str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p_%S"))
            env.dump_episode(fname)
        # train model_dynamics and policy
        minibatch_size = settings['minibatch_size']
        if replay_buffer.size() > minibatch_size:
            s0_batch, g0_batch, a_batch, s1_batch, g1_batch, target_batch = \
                replay_buffer.sample_batch(minibatch_size)

            # train model_dynamics
            # normalize data before train
            s0_batch_normed = normalize(s0_batch, s_bound)
            g0_batch_normed = normalize(g0_batch, g_bound)
            a_batch_normed = normalize(a_batch, a_bound)
            s1_batch_normed = normalize(s1_batch, s_bound)
            g1_batch_normed = normalize(g1_batch, g_bound)
            target_batch_normed = normalize(target_batch, g_bound)

            _, md_loss, md_goal_loss, md_abs_loss, md_cos_loss, s1_pred1 = \
                model_dynamics.train(s0_batch_normed, g0_batch_normed, a_batch_normed, s1_batch_normed, g1_batch_normed)
            if i % 200 == 0:
                s1_pred, g1_pred = model_dynamics.predict(s0_batch, g0_batch, a_batch)
                ds_pred = s1_pred - s0_batch
                ds = s1_batch - s0_batch
                cos_dist = np.mean([spatial.distance.cosine(ds_pred[e], ds[e]) for e in range(minibatch_size)])
                print("cosine distance: ", cos_dist)
            # train policy
            actions_normed = policy.predict(s0_batch_normed, g0_batch_normed, target_batch_normed)
            actions_normed = np.squeeze(actions_normed)

            desired_actions = target_batch_normed - g0_batch_normed
            policy_se_loss = np.sum(((actions_normed - desired_actions) ** 2), axis=1).mean(axis=None)
            cos_dists = [spatial.distance.cosine(np.linalg.norm(desired_actions[l]),
                                                               np.linalg.norm(actions_normed[l])) for l in range(minibatch_size)]
            cos_dists = [0. if math.isnan(e) else e for e in cos_dists]
            policy_cos_loss = np.mean(cos_dists)
            #     # np.mean(np.linalg.norm(actions - desired_actions, axis=1))

            action_gradients, ground_truth_action_gradients = model_dynamics.action_gradients(s0_batch_normed, g0_batch_normed, actions_normed,
                                                               target_batch_normed, desired_actions)
            action_gradients = action_gradients[0]
            ground_truth_action_gradients = ground_truth_action_gradients[0]
            gradient_loss = np.mean(np.linalg.norm(action_gradients - ground_truth_action_gradients))

            if md_loss < 0.002:
                # _ = policy.train_1(s0_batch_normed, g0_batch_normed, target_batch_normed, desired_actions)
                _ = policy.train(s0_batch_normed, g0_batch_normed, target_batch_normed, action_gradients)

            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: policy_se_loss,
                summary_vars[1]: md_loss,
                summary_vars[2]: md_goal_loss,
                summary_vars[3]: 0
            })

            writer.add_summary(summary_str, i)
            writer.flush()

            print(' Episode: {:d} |'
                  ' Policy loss: {:.4f}'
                  ' Policy cosine loss: {:.4f}'
                  ' Policy squared error loss: {:.4f}'
                  ' Model dynamics loss: {:.4f}|'
                  ' MD goal loss: {:.4f}'.format(i,
                                                 policy_se_loss,
                                                 policy_cos_loss,
                                                 policy_se_loss,
                                                  md_loss,
                                                  md_goal_loss))


def test_policy(settings, policy, model_dynamics, env, replay_buffer, reference_trajectory, render=True):

    # temp
    dm = real_dynamics(settings)

    sess = tf.get_default_session()
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    dt = str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p"))
    writer = tf.summary.FileWriter(settings['summary_dir'] + '/summary_md_' + dt, sess.graph)

    video_dir = settings['video_dir'] + '/video_md_' + dt
    os.makedirs(video_dir)

    num_episodes = 10000
    action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(env.action_dim))
    s_dim = settings['state_dim']
    g_dim = settings['goal_dim']
    a_dim = settings['action_dim']
    for i in range(num_episodes):
        # pick random initial state from the reference trajectory
        s0_index = randrange(0, reference_trajectory.shape[0] - 1)
        if i % 200 == 0:
            s0_index = 0
        s0 = reference_trajectory[s0_index]
        g0 = s0
        s_out = env.reset(s0)
        if render:
            env.render()
        r = []
        # rollout episode
        for j in range(s0_index, len(reference_trajectory) - 1):
            target = reference_trajectory[j + 1]
            action = policy.predict(np.reshape(s0, (1, s_dim)),
                                    np.reshape(g0, (1, g_dim)),
                                    np.reshape(target, (1, g_dim)))
            # add noise
            # action = action_noise()
            # make a step
            action += action_noise()
            # fix glottis temporary
            action = np.reshape(action, (a_dim))
            s1 = env.step(action)
            if render:
                env.render()
            g1 = s1
            # calc reward
            last_loss = np.linalg.norm(target - g1)

            r.append(-1. * np.linalg.norm(target - g1))
            replay_buffer.add(s0, g0, action, s1, g1, target)
            s0 = s1
            g0 = g1

            if last_loss > 4. and i % 200 != 0:
                break
        if i % 200 == 0:
            fname = video_dir + '/episode_' + str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p_%S"))
            env.dump_episode(fname)
        # train model_dynamics and policy
        minibatch_size = settings['minibatch_size']
        if replay_buffer.size() > minibatch_size:
            s0_batch, g0_batch, a_batch, s1_batch, g1_batch, target_batch = \
                replay_buffer.sample_batch(minibatch_size)

            # # # train policy
            # actions = policy.predict(s0_batch, g0_batch, target_batch)
            # actions = np.squeeze(actions)
            # action_gradients = model_dynamics.action_gradients(s0_batch, g0_batch, actions, target_batch)[0]
            #
            # # temp
            # action_gradients_1 = dm.action_gradients(actions, s0_batch, target_batch)[0]
            # #
            # # a_temp = np.reshape(np.append([0.]*10, [1.]*20), (1, 30))
            # # s0_temp = np.reshape([1.]*30, (1, 30))
            # # target_temp = np.reshape([3.]*30, (1, 30))
            # # action_gradients = dm.action_gradients(a_temp, s0_temp, target_temp)

            # loss, _ = policy.train_1(s0_batch, g0_batch, target_batch, target_batch-g0_batch)

            # policy.train(s0_batch, g0_batch, target_batch, action_gradients_1)

            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: np.mean(r),
                summary_vars[1]: loss,
                summary_vars[2]: 0,
                summary_vars[3]: 0
            })

            writer.add_summary(summary_str, i)
            writer.flush()

            print('| Reward: {:.4f} |'
                  ' Episode: {:d} |'
                  ' Model dynamics loss: {:.4f}|'
                  ' MD goal loss: {:.4f}'.format(loss,
                                                  i,
                                                  0,
                                                  0))


def main():
    speaker_fname = os.path.join(r'C:\Study\SpeechAcquisitionModel\src\VTL', 'JD2.speaker')
    lib_path = os.path.join(r'C:\Study\SpeechAcquisitionModel\src\VTL', 'VocalTractLab2.dll')
    ep_duration = 5000
    timestep = 20
    env = VTLEnv(lib_path, speaker_fname, timestep, max_episode_duration=ep_duration)

    settings = {
            'state_dim': env.state_dim,
            'action_dim': env.action_dim,
            'state_bound': env.state_bound,
            'action_bound': env.action_bound,
            'goal_dim': env.state_dim,
            'goal_bound': env.state_bound,
            'episode_length': 40,
            'minibatch_size': 512,

            'actor_tau': 0.01,
            'actor_learning_rate': 0.000001,

            'model_dynamics_learning_rate': 0.05,

            'summary_dir': r'C:\Study\SpeechAcquisitionModel\reports\summaries',
            'videos_dir': r'C:\Study\SpeechAcquisitionModel\reports\videos'
        }
    with tf.Session() as sess:

        policy = Policy('P1', settings)
        md = ModelDynamics('MD1', settings)
        replay_buffer = ReplayBuffer(100000)

        sess.run(tf.global_variables_initializer())
        reference_fname = r'C:\Study\SpeechAcquisitionModel\src\VTL\references\a_i.pkl'
        with open(reference_fname, 'rb') as f:
            (tract_params, glottis_params) = pickle.load(f)
            target_trajectory = np.hstack((np.array(tract_params), np.array(glottis_params)))
        train(settings, policy, md, env, replay_buffer, target_trajectory)
    return


if __name__ == '__main__':
    np.random.seed(1234)
    tf.set_random_seed(1234)
    main()


