""" 
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task 
and developed with tflearn + Tensorflow

Author: Patrick Emami
"""
import tensorflow as tf
from collections import deque
import numpy as np
import gym
from gym import wrappers
import tflearn
from tensorflow.python.keras.layers import Dense, Input, TimeDistributed, LSTMCell, LSTM, BatchNormalization, Activation
from tensorflow.python.keras.initializers import RandomUniform

import argparse
import pprint as pp
import datetime
import time
import timeit

from sequence_replay_buffer import ReplayBuffer

# ===========================
#   Actor and Critic DNNs
# ===========================


class StateToStateNetwork(object):
    """
    Input to the network is the current state and the goal state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.state_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_state_to_state_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_state_to_state_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.goal_state_gradient = tf.placeholder(tf.float32, [None, self.s_dim])

        # Combine the gradients here
        self.unnormalized_sts_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.goal_state_gradient)
        self.goal_state_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_sts_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.goal_state_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_state_to_state_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.s_dim, activation='tanh', weights_init=w_init)
        scaled_out = tf.multiply(out, self.state_bound)
        return inputs, out, scaled_out

    def train(self, inputs, gs_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.goal_state_gradient: gs_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class LSTMCriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, name, model_settings):
        self.sess = tf.get_default_session()
        assert (self.sess is not None)
        self.name = name
        self.s_dim = model_settings['state_dim']
        self.state_bound = model_settings['state_bound']
        self.learning_rate = model_settings['critic_learning_rate']
        self.tau = model_settings['critic_tau']
        self.episode_length = model_settings['critic_episode_length']
        self.lstm_num_cells = model_settings['critic_lstm_num_cells']

        # Create the critic network
        with tf.variable_scope(self.name + '_critic'):
            self.state, self.next_state, self.out = self.create_critic_network()
            self.network_params = tf.trainable_variables(scope=self.name + '_critic')

        # Target Network
        with tf.variable_scope(self.name + '_target_critic'):
            self.target_state, self.target_next_state, self.target_out = self.create_critic_network()
            self.target_network_params = tf.trainable_variables(scope=self.name + '_target_critic')

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, self.episode_length, 1])

        # Define loss and optimization Op
        self.loss = tf.losses.mean_squared_error(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.next_state_grads = tf.gradients(self.out[:, -1, :], self.next_state)

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    def create_critic_network(self):
        ep_length = self.episode_length
        batch_state_x = Input(batch_shape=[None, ep_length, self.s_dim])
        batch_next_state_x = Input(batch_shape=[None, ep_length, self.s_dim])

        #  state branch
        state_net = TimeDistributed(Dense(400, activation='relu'))(batch_state_x)

        # action branch
        next_state_net = TimeDistributed(Dense(400, activation='relu'))(batch_next_state_x)

        # merge branches
        t1_layer = TimeDistributed(Dense(400))
        t1_layer_out = t1_layer(state_net)
        t2_layer = TimeDistributed(Dense(400))
        t2_layer_out = t2_layer(next_state_net)

        state_net_reshaped = tf.reshape(state_net, shape=[-1, 400])
        action_net_reshaped = tf.reshape(next_state_net, shape=[-1, 400])
        merged_net = tf.matmul(state_net_reshaped, t1_layer.get_weights()[0]) + tf.matmul(action_net_reshaped,
                                                                                          t2_layer.get_weights()[0]) \
                     + t1_layer.get_weights()[1] + t2_layer.get_weights()[1]
        merged_net = Activation('relu')(merged_net)
        merged_net = tf.reshape(merged_net, shape=[-1, ep_length, 400])

        # lstm cell
        rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.lstm_num_cells, state_is_tuple=True)
        val, state = tf.nn.dynamic_rnn(rnn_cell, merged_net, dtype=tf.float32)
        lstm_outputs = val

        # final dense layer
        w_init = RandomUniform(minval=-0.005, maxval=0.005)
        last_layer = Dense(1)
        batch_y = last_layer(lstm_outputs)
        return batch_state_x, batch_next_state_x, batch_y

    def train(self, states, next_states, predicted_q_values):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.state: states,
            self.next_state: next_states,
            self.predicted_q_value: predicted_q_values
        })

    def predict(self, states, next_states):
        return self.sess.run(self.out, feed_dict={
            self.state: states,
            self.next_state: next_states
        })

    def predict_target(self, states, next_states):
        return self.sess.run(self.target_out, feed_dict={
            self.target_state: states,
            self.target_next_state: next_states
        })

    def next_state_gradients(self, states, next_states):
        return self.sess.run(self.next_state_grads, feed_dict={
            self.state: states,
            self.next_state: next_states
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)


class LSTMActorNetwork(object):
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
        self.a_dim = model_settings['action_dim']
        self.action_bound = model_settings['action_bound']
        self.learning_rate = model_settings['actor_learning_rate']
        self.tau = model_settings['actor_tau']
        self.episode_length = model_settings['actor_episode_length']
        self.lstm_num_cells = model_settings['actor_lstm_num_cells']
        self.batch_size = model_settings['minibatch_size']

        # Actor Network
        with tf.variable_scope(self.name + '_actor'):
            self.inputs, self.out, self.scaled_out = self.create_actor_network()
            self.network_params = tf.trainable_variables(scope=self.name + '_actor')

        # Target Network
        with tf.variable_scope(self.name + '_target_actor'):
            self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()
            self.target_network_params = tf.trainable_variables(scope=self.name + '_target_actor')

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.episode_length, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        ep_length = self.episode_length
        batch_state_x = Input(batch_shape=[None, ep_length, self.s_dim])

        #  state branch
        state_net = TimeDistributed(Dense(400, activation='relu'))(batch_state_x)

        # lstm cell
        rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.lstm_num_cells, state_is_tuple=True)
        val, state = tf.nn.dynamic_rnn(rnn_cell, state_net, dtype=tf.float32)
        lstm_outputs = val

        # final dense layer
        w_init = RandomUniform(minval=-0.005, maxval=0.005)
        last_layer = Dense(self.a_dim)
        batch_action_y = last_layer(lstm_outputs)
        batch_action_y_scaled_out = tf.multiply(batch_action_y, self.action_bound)
        return batch_state_x, batch_action_y, batch_action_y_scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class LSTMWorldModelerNetwork(object):
    """
    Input to the network is the sequence of (state, action) couples, output is the the sequence of expected states after
     applying action.

    The output layer activation is a tanh to keep the action
    between -state_bound and state_bound
    """

    def __init__(self, name, model_settings):
        self.sess = tf.get_default_session()
        assert(self.sess is not None)
        self.name = name
        self.s_dim = model_settings['state_dim']
        self.a_dim = model_settings['action_dim']
        self.state_bound = model_settings['state_bound']
        self.learning_rate = model_settings['world_modeler_learning_rate']
        self.tau = model_settings['world_modeler_tau']
        self.episode_length = model_settings['world_modeler_episode_length']
        self.lstm_num_cells = model_settings['world_modeler_lstm_num_cells']
        self.batch_size = model_settings['minibatch_size']

        # build_froward_pass for the network
        with tf.variable_scope(name + '_world_modeler'):
            self.inputs, self.action, self.out, self.scaled_out = self.create_world_modeler_network()
        self.network_params = tf.trainable_variables(scope=name + '_world_modeler')

        with tf.variable_scope(name + '_target_world_modeler'):
            # build_froward_pass for the Target Network
            self.target_inputs, self.target_action, self.target_out, self.target_scaled_out = self.create_world_modeler_network()
        self.target_network_params = tf.trainable_variables(scope=name + '_target_world_modeler')
        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_state = tf.placeholder(tf.float32, [None, self.episode_length, self.s_dim])

        # Define loss and optimization Op
        self.loss = tf.losses.mean_squared_error(self.predicted_state, self.scaled_out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.

        # This gradient will be provided by the critic network
        self.next_state_gradient = tf.placeholder(tf.float32, [None, self.episode_length, self.s_dim])

        # Combine the gradients here
        self.unnormalized_action_gradients = tf.gradients(
            self.scaled_out, self.action, self.next_state_gradient)
        self.normed_action_gradients = tf.div(self.unnormalized_action_gradients, self.batch_size)

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_world_modeler_network(self):
        ep_length = self.episode_length
        batchStateX = Input(batch_shape=[None, ep_length, self.s_dim])
        batchActionX = Input(batch_shape=[None, ep_length, self.a_dim])

        #  state branch
        state_net = TimeDistributed(Dense(400, activation='relu'))(batchStateX)

        # action branch
        action_net = TimeDistributed(Dense(400, activation='relu'))(batchActionX)

        # merge branches
        t1_layer = TimeDistributed(Dense(400))
        t1_layer_out = t1_layer(state_net)
        t2_layer = TimeDistributed(Dense(400))
        t2_layer_out = t2_layer(action_net)

        state_net_reshaped = tf.reshape(state_net, shape=[-1, 400])
        action_net_reshaped = tf.reshape(action_net, shape=[-1, 400])
        merged_net = tf.matmul(state_net_reshaped, t1_layer.get_weights()[0]) + tf.matmul(action_net_reshaped, t2_layer.get_weights()[0])\
                     + t1_layer.get_weights()[1] + t2_layer.get_weights()[1]
        merged_net = Activation('relu')(merged_net)
        merged_net = tf.reshape(merged_net, shape=[-1, ep_length, 400])

        # lstm cell
        rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.lstm_num_cells, state_is_tuple=True)
        val, state = tf.nn.dynamic_rnn(rnn_cell, merged_net, dtype=tf.float32)
        lstm_outputs = val

        # final dense layer
        w_init = RandomUniform(minval=-0.005, maxval=0.005)
        last_layer = Dense(self.s_dim)
        batchStateY = last_layer(lstm_outputs)
        batchStateY_scaled_out = tf.multiply(batchStateY, self.state_bound)

        return batchStateX, batchActionX, batchStateY, batchStateY_scaled_out

    def train(self, inputs, action, predicted_state):
        return self.sess.run([self.loss, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_state: predicted_state
        })

    def predict(self, inputs, action):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_test(self, inputs, action):
        return self.sess.run([self.scaled_out], feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions, next_state_grads):
        return self.sess.run(self.normed_action_gradients, feed_dict={
            self.inputs: inputs,
            self.action: actions,
            self.next_state_gradient: next_state_grads
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)


# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
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


def get_default_env_settings(env_name):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high
    state_bound = env.observation_space.high
    settings = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'state_bound': state_bound,
        'action_bound': action_bound,
        'episode_length': 10,
        'minibatch_size': 128
    }
    return settings


def get_default_world_modeler_settings():
    settings = {
        'world_modeler_learning_rate': 0.0001,
        'world_modeler_tau': 0.001,
        'world_modeler_lstm_num_cells': 200
    }
    return settings


def get_default_actor_settings():
    settings = {
        'actor_learning_rate': 0.0001,
        'actor_tau': 0.001,
        'actor_lstm_num_cells': 200,
        'actor_batch_size': 128
    }
    return settings


def get_default_critic_settings():
    settings = {
        'critic_learning_rate': 0.001,
        'critic_tau': 0.001,
        'critic_lstm_num_cells': 200,
        'critic_gamma': 0.99
    }
    return settings


def get_default_model_settings(env_name):
    actor_settings = get_default_actor_settings()
    world_modeler_settings = get_default_world_modeler_settings()
    critic_settings = get_default_critic_settings()
    env_settings = get_default_env_settings(env_name)
    settings = {**actor_settings,
                **world_modeler_settings,
                **critic_settings,
                **env_settings}

    settings['critic_episode_length'] = settings['episode_length']
    settings['actor_episode_length'] = settings['episode_length']
    settings['world_modeler_episode_length'] = settings['episode_length']
    return settings

# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    actor_ep_loss = tf.Variable(0.)
    tf.summary.scalar("Actor loss", actor_ep_loss)

    summary_vars = [episode_reward, episode_ave_max_q, actor_ep_loss]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================


def train(sess, env, settings, args, actor, critic, actor_noise, world_modeler):

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    dt = str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p"))
    writer = tf.summary.FileWriter(args['summary_dir'] + '_' + dt, sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()
    world_modeler.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    # Needed to enable BatchNorm. 
    # This hurts the performance on Pendulum but could be useful
    # in other environments.
    # tflearn.is_training(True)

    a_dim = settings['action_dim']
    s_dim = settings['state_dim']
    episode_length = settings['episode_length']
    minibatch_size = settings['minibatch_size']


    for i in range(int(args['max_episodes'])):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0
        loss = 0

        zero_episode = (np.reshape(s, (s_dim,)),
                        np.reshape(np.zeros(a_dim), (a_dim,)),
                        np.array(0.),
                        np.array(False),
                        np.reshape(s, (s_dim,)))
        history = deque([zero_episode] * episode_length)

        for j in range(int(args['max_episode_len'])):

            if args['render_env']:
                env.render()

            timer = {
                'world_modeler_train_time': 0.,
                'world_modeler_predict_time': 0.,
                'target_q_predict_time': 0.,
                'critic_train_time': 0.,
                'actor_train_time': 0.,
                'actor_predict_time': 0.,
                'actor_gradients': 0.,
                'actor_predict_bacth_time': 0.,
                'world_modeler_predict_batch_time': 0.,
            }



            history_states = np.array([sample_ep[0] for sample_ep in history])

            tic = time.time()
            for test_i in range(1):
                a = actor.predict(np.reshape(history_states, (1, -1, s_dim))) + actor_noise()
            timer['actor_predict_time'] += time.time() - tic
            # # Added exploration noise
            # #a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
            # a = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()

            s2, r, terminal, info = env.step(a[0])

            history.popleft()
            history.append((np.reshape(s, (actor.s_dim,)), np.reshape(a[:, -1, :], (actor.a_dim,)), np.array(r),
                              np.array(terminal), np.reshape(s2, (actor.s_dim,))))

            replay_buffer.add(np.array(history))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > minibatch_size:
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(minibatch_size)


                tic = time.time()
                loss, _ = world_modeler.train(s_batch, a_batch, s2_batch)
                timer['world_modeler_train_time'] += time.time() - tic

                tic = time.time()
                target_next_states = world_modeler.predict_target(s_batch, a_batch)
                timer['world_modeler_predict_time'] += time.time() - tic

                tic = time.time()
                # Calculate targets
                target_q = critic.predict_target(s_batch, target_next_states)
                timer['target_q_predict_time'] += time.time() - tic

                # Bellman equation for target calculation
                gamma = settings['critic_gamma']
                t_q = np.multiply(-1 * np.array(t_batch).astype(int) + 1., np.squeeze(target_q))
                y_i = np.squeeze(r_batch) + gamma * t_q
                # for k in range(minibatch_size):
                #     if t_batch[k]:
                #         y_i.append(r_batch[k])
                #     else:
                #         y_i.append(r_batch[k] + critic.gamma * target_q[k])

                tic = time.time()
                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, s2_batch, np.expand_dims(y_i, axis=2))
                timer['critic_train_time'] += time.time() - tic

                ep_ave_max_q += np.amax(predicted_q_value)

                tic = time.time()
                # Update the actor policy using the sampled gradient
                action_outs = actor.predict(s_batch)
                timer['actor_predict_bacth_time'] += time.time()- tic
                tic = time.time()
                next_states_outs = world_modeler.predict(s_batch, action_outs)
                timer['world_modeler_predict_batch_time'] += time.time() - tic
                tic = time.time()
                next_state_grads = critic.next_state_gradients(s_batch, next_states_outs)[0]
                action_grads = world_modeler.action_gradients(s_batch, action_outs, next_state_grads)
                timer['actor_gradients'] += time.time() - tic
                tic = time.time()
                actor.train(s_batch, action_grads[0])

                timer['actor_train_time'] += time.time() - tic
                # Update target networks
                actor.update_target_network()
                critic.update_target_network()
                world_modeler.update_target_network()

            s = s2
            ep_reward += r[0]
            # # # Update the actor policy using the sampled gradient
            #     # gs_outs = state_to_state_net.predict(s_batch)
            #     # grads = critic.goal_state_gradients(s_batch, gs_outs)
            #     # state_to_state_net.train(s_batch, grads[0])
            #
            #
            #     # Calculate goal_state_targets
            #     # Calculate action_target
            #     # target_goal_state = world_modeler.predict_target(
            #     #     s_batch, actor.predict_target(s_batch, s2_batch))
            #
            #     # Update world modeler
            #     # out, _ = world_modeler.train(s_batch, a_batch, s2_batch)
            #
            #     # Update actor
            #     loss, _ = actor.train(s_batch, s2_batch, a_batch)
            #
            #     # action_outs = actor.predict(s_batch, s2_batch)
            #     # grads = world_modeler.action_gradients(s_batch, action_outs)
            #     # actor.train(s_batch, s2_batch, grads[0])
            #
            #     # Update target networks
            #     critic.update_target_network()
            #     actor.update_target_network()
            #     world_modeler.update_target_network()
            #
            # s = s2
            # ep_reward += r

            if terminal:

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j),
                    summary_vars[2]: loss
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f} | Loss: {:.4f}'.format(int(ep_reward),
                                                                                            i, (ep_ave_max_q / float(j)), loss))

                for k, v in timer.items():
                    print("{:<15} {:0.3f}".format(k, v))
                print(timer)

                break


def main(args):

    with tf.Session() as sess:

        env = gym.make(args['env'])
        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        env.seed(int(args['random_seed']))

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        state_bound = env.observation_space.high

        # Ensure action bound is symmetric
        assert (env.action_space.high == -env.action_space.low)
        assert (sum(abs(env.observation_space.high + env.observation_space.low)) == 0)

        actor_settings = get_default_actor_settings()
        world_modeler_settings = get_default_world_modeler_settings()
        critic_settings = get_default_critic_settings()
        env_settings = get_default_env_settings(args['env'])
        settings = {**actor_settings,
                    **world_modeler_settings,
                    **critic_settings,
                    **env_settings}
        settings = get_default_model_settings(args['env'])

        critic = LSTMCriticNetwork('c_0', settings)

        actor = LSTMActorNetwork('a_0', settings)

        world_modeler = LSTMWorldModelerNetwork('wm_0', settings)

        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        if args['use_gym_monitor']:
            if not args['render_env']:
                env = wrappers.Monitor(
                    env, args['monitor_dir'], video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, args['monitor_dir'], force=True)

        train(sess, env, settings, args, actor, critic, actor_noise, world_modeler)

        if args['use_gym_monitor']:
            env.monitor.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=100000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=12345)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=50000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_ddpg')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/tf_ddpg')

    parser.set_defaults(render_env=True)
    parser.set_defaults(use_gym_monitor=True)
    
    args = vars(parser.parse_args())
    
    pp.pprint(args)

    main(args)
