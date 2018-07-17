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
from tensorflow.python.keras.layers import Dense, Input, TimeDistributed, LSTMCell, LSTM, BatchNormalization, Activation, Add
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
        self.a_dim = model_settings['action_dim']
        self.state_bound = model_settings['state_bound']
        self.learning_rate = model_settings['critic_learning_rate']
        self.tau = model_settings['critic_tau']
        self.episode_length = model_settings['critic_episode_length']
        self.lstm_num_cells = model_settings['critic_lstm_num_cells']

        # Create the critic network
        with tf.variable_scope(self.name + '_critic'):
            self.state, self.action, self.sequence_length_placeholder, self.out = self.create_critic_network()
            self.network_params = tf.trainable_variables(scope=self.name + '_critic')

        # Target Network
        with tf.variable_scope(self.name + '_target_critic'):
            self.target_state, self.target_action, self.target_sequence_length_placeholder, self.target_out = self.create_critic_network()
            self.target_network_params = tf.trainable_variables(scope=self.name + '_target_critic')

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        self.copy_target_network_params = [self.target_network_params[i].assign(self.network_params[i])
                                    for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, None, 1])

        # Define loss and optimization Op
        # self.loss = tf.losses.mean_squared_error(self.predicted_q_value, self.out)
        self.loss = tflearn.mean_square(self.out, self.predicted_q_value)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    def create_critic_network(self):
        sequence_length = tf.placeholder(tf.int32, shape=[None])
        batch_state_x = Input(batch_shape=[None, None, self.s_dim])
        batch_action_x = Input(batch_shape=[None, None, self.a_dim])

        #  state branch
        state_net = TimeDistributed(Dense(400))(batch_state_x)
        state_net = TimeDistributed(BatchNormalization())(state_net)
        state_net = TimeDistributed(Activation('relu'))(state_net)

        # action branch
        action_net = TimeDistributed(Dense(400))(batch_action_x)
        action_net = TimeDistributed(BatchNormalization())(action_net)
        action_net = TimeDistributed(Activation('relu'))(action_net)

        # merge branches
        t1_layer = TimeDistributed(Dense(300))
        t1_layer_out = t1_layer(state_net)
        t2_layer = TimeDistributed(Dense(300))
        t2_layer_out = t2_layer(action_net)

        merged_net = Add()([t1_layer_out, t2_layer_out])
        merged_net = TimeDistributed(Activation('relu'))(merged_net)

        # lstm cell
        rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.lstm_num_cells, state_is_tuple=True)
        lstm_outputs, state = tf.nn.dynamic_rnn(rnn_cell, merged_net,
                                       sequence_length=sequence_length,
                                       time_major=False, dtype=tf.float32)


        # final dense layer
        w_init = RandomUniform(minval=-0.003, maxval=0.003)
        last_layer = TimeDistributed(Dense(1, kernel_initializer=w_init))
        batch_y = last_layer(lstm_outputs)
        return batch_state_x, batch_action_x, sequence_length, batch_y

    def train(self, states, actions, predicted_q_values, sequence_length):
        return self.sess.run([self.out, self.optimize, self.loss], feed_dict={
            self.state: states,
            self.action: actions,
            self.predicted_q_value: predicted_q_values,
            self.sequence_length_placeholder: sequence_length
        })

    def predict(self, states, next_states, sequence_length):
        return self.sess.run(self.out, feed_dict={
            self.state: states,
            self.action: next_states,
            self.sequence_length_placeholder: sequence_length
        })

    def predict_target(self, states, actions, sequence_length):
        return self.sess.run(self.target_out, feed_dict={
            self.target_state: states,
            self.target_action: actions,
            self.target_sequence_length_placeholder: sequence_length
        })

    def action_gradients(self, states, actions, sequence_length):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions,
            self.sequence_length_placeholder: sequence_length
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def init_target_network(self):
        self.sess.run(self.copy_target_network_params)


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
            self.inputs, \
            self.sequence_length_placeholder, \
            self.init_state,\
            self.state, \
            self.out, \
            self.scaled_out = \
                self.create_actor_network()
            self.network_params = tf.trainable_variables(scope=self.name + '_actor')

        # Target Network
        with tf.variable_scope(self.name + '_target_actor'):
            self.target_inputs, \
            self.target_sequence_length_placeholder, \
            self.target_initial_state,\
            self.target_state, \
            self.target_out,\
            self.target_scaled_out \
                = self.create_actor_network()
            self.target_network_params = tf.trainable_variables(scope=self.name + '_target_actor')

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        self.copy_target_network_params = [self.target_network_params[i].assign(self.network_params[i])
                                    for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(
            map(lambda x: tf.div(x, self.batch_size * self.episode_length), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        sequence_length = tf.placeholder(tf.int32, shape=[None])

        batch_state_x = Input(batch_shape=[None, None, self.s_dim])

        #  state branch
        state_net = TimeDistributed(Dense(400))(batch_state_x)
        state_net = TimeDistributed(BatchNormalization())(state_net)
        state_net = TimeDistributed(Activation('relu'))(state_net)

        state_net = TimeDistributed(Dense(400))(state_net)
        state_net = TimeDistributed(BatchNormalization())(state_net)
        state_net = TimeDistributed(Activation('relu'))(state_net)
        # lstm cell
        init_state = tf.placeholder(dtype=tf.float32, shape=[2, None, self.lstm_num_cells])
        rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.lstm_num_cells, state_is_tuple=True)
        initial_state = tf.nn.rnn_cell.LSTMStateTuple(init_state[0], init_state[1])
        val, state = tf.nn.dynamic_rnn(rnn_cell, state_net, initial_state=initial_state, sequence_length=sequence_length,
                                       time_major=False, dtype=tf.float32)
        lstm_outputs = val

        # final dense layer
        batch_action_y = TimeDistributed(Dense(self.a_dim, activation='tanh', kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003)))(lstm_outputs)
        batch_action_y_scaled_out = tf.multiply(batch_action_y, self.action_bound)
        return batch_state_x, sequence_length, init_state, state, batch_action_y, batch_action_y_scaled_out

    def train(self, inputs, a_gradient, sequence_length, batch_size, initial_state=None):
        if initial_state is None:
            initial_state = np.zeros((2, batch_size, self.lstm_num_cells))
        return self.sess.run([self.optimize, self.actor_gradients], feed_dict={
                self.inputs: inputs,
                self.action_gradient: a_gradient,
                self.sequence_length_placeholder: sequence_length,
                self.init_state: initial_state
            })

    def predict(self, inputs, seq_length, batch_size, initial_state=None):
        if initial_state is None:
            initial_state = np.zeros((2, batch_size, self.lstm_num_cells))
        return self.sess.run([self.scaled_out, self.state], feed_dict={
            self.inputs: inputs,
            self.sequence_length_placeholder: seq_length,
            self.init_state: initial_state
        })

    def predict_target(self, inputs, seq_length, batch_size, initial_state=None):
        if initial_state is None:
            initial_state = np.zeros((2, batch_size, self.lstm_num_cells))
        return self.sess.run([self.target_scaled_out, self.target_state], feed_dict={
            self.target_inputs: inputs,
            self.target_sequence_length_placeholder: seq_length,
            self.target_initial_state: initial_state
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def init_target_network(self):
        self.sess.run(self.copy_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


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
        'episode_length': 40,
        'minibatch_size': 256
    }
    return settings


def get_default_actor_settings():
    settings = {
        'actor_learning_rate': 0.0001,
        'actor_tau': 0.01,
        'actor_lstm_num_cells': 50,
        'actor_batch_size': 256
    }
    return settings


def get_default_critic_settings():
    settings = {
        'critic_learning_rate': 0.001,
        'critic_tau': 0.01,
        'critic_lstm_num_cells': 100,
        'critic_gamma': 0.99
    }
    return settings


def get_default_model_settings(env_name):
    actor_settings = get_default_actor_settings()
    critic_settings = get_default_critic_settings()
    env_settings = get_default_env_settings(env_name)
    settings = {**actor_settings,
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

    critic_loss = tf.Variable(0.)
    tf.summary.scalar("Critic loss", critic_loss)

    avg_action = tf.Variable(0.)
    tf.summary.scalar("Average action", avg_action)

    act_grads = tf.placeholder(dtype=tf.float32, shape=None)
    tf.summary.histogram('action_gradients', act_grads)

    actor_grads = tf.placeholder(dtype=tf.float32, shape=None)
    tf.summary.histogram('actor_gradients', actor_grads)

    actor_activations = tf.placeholder(dtype=tf.float32, shape=None)
    tf.summary.histogram('actor_activations', actor_activations)


    variables = [v for v in tf.trainable_variables()]
    [tf.summary.histogram(v.name, v) for v in variables]

    summary_vars = [episode_reward, episode_ave_max_q, actor_ep_loss, critic_loss, avg_action, act_grads, actor_grads, actor_activations]

    summary_vars.extend(variables)
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


# ===========================
#   Agent Training
# ===========================


def train(sess, env, settings, args, actor, critic, actor_noise):
    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()


    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    dt = str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p"))
    writer = tf.summary.FileWriter(args['summary_dir'] + '_' + dt, sess.graph)

    # Initialize target network weights
    actor.init_target_network()
    critic.init_target_network()

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
        critic_loss = 0
        action_grads = 0
        actor_grads = [0, 0]
        a_outs = 0

        zero_episode = (np.reshape(s, (s_dim,)),
                        np.reshape(np.zeros(a_dim), (a_dim,)),
                        np.array(0.),
                        np.array(False),
                        np.reshape(s, (s_dim,)))
        history = deque([zero_episode] * episode_length)
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
        hidden_state = None
        action_avg = 0.
        for j in range(int(args['max_episode_len'])):

            if args['render_env']:
                env.render()

            history_states = np.array([sample_ep[0] for sample_ep in history])

            tic = time.time()
            a, hidden_state = actor.predict(np.reshape(s, (1, 1, s_dim)), [1], 1, hidden_state)
            noise = actor_noise()
            action_avg += np.squeeze(a)
            a = a + noise

            # # Added exploration noise

            s2, r, terminal, info = env.step(a[0])

            history.popleft()
            history.append((np.reshape(s, (actor.s_dim,)), np.reshape(a[:, -1, :], (actor.a_dim,)), np.array(r),
                            np.array(terminal), np.reshape(s2, (actor.s_dim,))))
            if j >= episode_length:
                replay_buffer.add(np.array(history))

            s = s2
            ep_reward += r[0]

            # there are at least minibatch size samples
            # Keep adding experience to the memory until
            if terminal and replay_buffer.size() > minibatch_size:
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(minibatch_size)

                target_actions, _ = actor.predict_target(s2_batch, [episode_length] * minibatch_size, minibatch_size)
                # Calculate targets
                target_q = critic.predict_target(s2_batch, target_actions, [episode_length] * minibatch_size)
                # Bellman equation for target calculation
                gamma = settings['critic_gamma']
                t_q = np.multiply(-1 * np.array(t_batch).astype(int) + 1., np.squeeze(target_q))
                y_i = np.squeeze(r_batch) + gamma * t_q
                # Update the critic given the targets
                y_i = np.reshape(y_i, (minibatch_size, episode_length, 1))
                predicted_q_value, _, critic_loss = critic.train(
                    s_batch, a_batch, y_i, [episode_length] * minibatch_size)
                ep_ave_max_q += np.amax(predicted_q_value)
                # Update the actor policy using the sampled gradient
                a_outs, _ = actor.predict(s_batch, [episode_length] * minibatch_size, minibatch_size)
                action_grads = critic.action_gradients(s_batch, a_outs, [episode_length] * minibatch_size)
                _, actor_grads = actor.train(s_batch, action_grads[0], [episode_length] * minibatch_size, minibatch_size)
                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            if terminal:
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j),
                    summary_vars[2]: loss,
                    summary_vars[3]: critic_loss,
                    summary_vars[4]: np.sum(action_avg) / j,
                    summary_vars[5]: action_grads,
                    summary_vars[6]: actor_grads[0],
                    summary_vars[7]: np.sum(a_outs)

                })

                writer.add_summary(summary_str, i)
                writer.flush()

                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f} | Critic Loss: {:.4f}| Avg action: {:.2f}'.format(int(ep_reward),
                                                                                                                       i,
                                                                                                                       (ep_ave_max_q / float(j)),
                                                                                                                       critic_loss,
                                                                                                                       np.sum(action_avg) / j))

                # for k, v in timer.items():
                #     print("{:<15} {:0.3f}".format(k, v))
                # print(timer)

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
        # print((sum(abs(env.action_space.high + env.action_space.low)) == 0))
        # print(sum(abs(env.observation_space.high + env.observation_space.low)))
        # assert (sum(abs(env.action_space.high + env.action_space.low)) == 0)
        # assert (sum(abs(env.observation_space.high + env.observation_space.low)) == 0)

        actor_settings = get_default_actor_settings()
        critic_settings = get_default_critic_settings()
        env_settings = get_default_env_settings(args['env'])
        settings = {**actor_settings,
                    **critic_settings,
                    **env_settings}
        settings = get_default_model_settings(args['env'])

        critic = LSTMCriticNetwork('c_0', settings)

        actor = LSTMActorNetwork('a_0', settings)

        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim), sigma=action_bound / 20.)

        if args['use_gym_monitor']:
            if not args['render_env']:
                env = wrappers.Monitor(
                    env, args['monitor_dir'], video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, args['monitor_dir'], force=True)

        train(sess, env, settings, args, actor, critic, actor_noise)

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
