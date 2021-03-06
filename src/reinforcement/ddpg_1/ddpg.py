""" 
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task 
and developed with tflearn + Tensorflow

Author: Patrick Emami
"""
import tensorflow as tf
import numpy as np
from collections import deque
import gym
from gym import wrappers
import tflearn
from tensorflow.python.keras.layers import Dense, Input, TimeDistributed, LSTMCell, LSTM, BatchNormalization, Activation, Add
from tensorflow.python.keras.initializers import RandomUniform
import argparse
import pprint as pp
import datetime

from SpeechAcquisitionModel.reinforcement.lstm_ddpg_dynamic.sequence_replay_buffer import ReplayBuffer

# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        self.lstm_num_cells = 100
        self.episode_length = 10

        with tf.variable_scope('actor'):
            # Actor Network
            self.inputs,\
            self.sequence_length_placeholder,\
            self.initial_state,\
            self.state,\
            self.out,\
            self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        with tf.variable_scope('target_actor'):
            # Target Network
            self.target_inputs,\
            self.target_sequence_length_placeholder, \
            self.target_initial_state,\
            self.target_state, \
            self.target_out,\
            self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size * self.episode_length), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        sequence_length = tf.placeholder(tf.int32, shape=[None])
        inputs = Input(batch_shape=[None, None, self.s_dim])

        net = TimeDistributed(Dense(400))(inputs)
        net = TimeDistributed(BatchNormalization())(net)
        net = TimeDistributed(Activation('relu'))(net)
        net = TimeDistributed(Dense(300))(net)
        net = TimeDistributed(BatchNormalization())(net)
        net = TimeDistributed(Activation('relu'))(net)

        init_state = tf.placeholder(dtype=tf.float32, shape=[2, None, self.lstm_num_cells])
        rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.lstm_num_cells, state_is_tuple=True)
        initial_state = tf.nn.rnn_cell.LSTMStateTuple(init_state[0], init_state[1])
        val, state = tf.nn.dynamic_rnn(rnn_cell, net, initial_state=initial_state,
                                       sequence_length=sequence_length,
                                       time_major=False, dtype=tf.float32)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        out = TimeDistributed(Dense(self.a_dim, activation='tanh',
                                               kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003, seed=None)))(val)
        scaled_out = tf.multiply(out, self.action_bound)

        # inputs = tflearn.input_data(shape=[None, None, self.s_dim])
        # net = TimeDistributed(Dense(400))(inputs)
        # # net = tflearn.time_distributed(inputs, tflearn.fully_connected,  [400])
        # net = tflearn.time_distributed(net, tflearn.layers.normalization.batch_normalization)
        # net = tflearn.time_distributed(net, tflearn.activations.relu)
        # net = tflearn.time_distributed(net, tflearn.fully_connected, [300])
        # net = tflearn.time_distributed(net, tflearn.layers.normalization.batch_normalization)
        # net = tflearn.time_distributed(net, tflearn.activations.relu)
        # # Final layer weights are init to Uniform[-3e-3, 3e-3]
        # w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        # out = tflearn.time_distributed(net, tflearn.fully_connected, [self.a_dim, 'tanh', True, w_init])
        # # Scale output to -action_bound to action_bound
        # scaled_out = tf.multiply(out, self.action_bound)
        return inputs, sequence_length, init_state, state, out, scaled_out

    def train(self, inputs, a_gradient, sequence_length, batch_size, initial_state=None):
        if initial_state is None:
            initial_state = np.zeros((2, batch_size, self.lstm_num_cells))
        return self.sess.run([self.optimize, self.actor_gradients], feed_dict={
                self.inputs: inputs,
                self.action_gradient: a_gradient,
                self.sequence_length_placeholder: sequence_length,
                self.initial_state: initial_state
            })

    def predict(self, inputs, seq_length, batch_size, initial_state=None):
        if initial_state is None:
            initial_state = np.zeros((2, batch_size, self.lstm_num_cells))
        return self.sess.run([self.scaled_out, self.state], feed_dict={
            self.inputs: inputs,
            self.sequence_length_placeholder: seq_length,
            self.initial_state: initial_state
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

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.lstm_num_cells = 100

        # Create the critic network
        with tf.variable_scope('critic'):
            self.inputs, self.action, self.sequence_length_placeholder, self.out = self.create_critic_network()

            self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        with tf.variable_scope('target_critic'):
            self.target_inputs, self.target_action, self.target_sequence_length_placeholder, self.target_out = self.create_critic_network()

            self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.out, self.predicted_q_value)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        sequence_length = tf.placeholder(tf.int32, shape=[None])
        inputs = Input(batch_shape=[None, None, self.s_dim])
        action = Input(batch_shape=[None, None, self.a_dim])

        net = TimeDistributed(Dense(400))(inputs)
        net = TimeDistributed(BatchNormalization())(net)
        net = TimeDistributed(Activation('relu'))(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = TimeDistributed(Dense(300))(net)
        t2 = TimeDistributed(Dense(300))(action)

        net = Add()([t1, t2])
        net = TimeDistributed(Activation('relu'))(net)

        rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.lstm_num_cells, state_is_tuple=True)
        val, state = tf.nn.dynamic_rnn(rnn_cell, net,
                                       sequence_length=sequence_length,
                                       time_major=False, dtype=tf.float32)

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        out = TimeDistributed(Dense(1, kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003, seed=None)))(val)

        # inputs = tflearn.input_data(shape=[None, 10, self.s_dim])
        # action = tflearn.input_data(shape=[None, 10, self.a_dim])
        # net = tflearn.time_distributed(inputs, tflearn.fully_connected, [400])
        # net = tflearn.time_distributed(net, tflearn.layers.normalization.batch_normalization)
        # net = tflearn.time_distributed(net, tflearn.activations.relu)
        #
        # # Add the action tensor in the 2nd hidden layer
        # # Use two temp layers to get the corresponding weights and biases
        # t1 = tflearn.time_distributed(net, tflearn.fully_connected, [300])
        # t2 = tflearn.time_distributed(action, tflearn.fully_connected, [300])
        #
        # net = tflearn.merge([t1, t2], mode='elemwise_sum')
        # net = tflearn.time_distributed(net, tflearn.activations.relu)
        #
        # # linear layer connected to 1 output representing Q(s,a)
        # # Weights are init to Uniform[-3e-3, 3e-3]
        # w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        # out = tflearn.time_distributed(net, tflearn.fully_connected, [1, 'linear', True, w_init])
        return inputs, action, sequence_length, out

    def train(self, states, actions, predicted_q_values, sequence_length):
        return self.sess.run([self.out, self.optimize, self.loss], feed_dict={
            self.inputs: states,
            self.action: actions,
            self.predicted_q_value: predicted_q_values,
            self.sequence_length_placeholder: sequence_length
        })

    def predict(self, inputs, action, sequence_length):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.sequence_length_placeholder: sequence_length
        })

    def predict_target(self, inputs, action, sequence_length):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action,
            self.target_sequence_length_placeholder: sequence_length
        })

    def action_gradients(self, inputs, actions, sequence_length):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions,
            self.sequence_length_placeholder: sequence_length
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

    summary_vars = [episode_reward, episode_ave_max_q, actor_ep_loss, critic_loss, avg_action, act_grads, actor_grads,
                    actor_activations]

    summary_vars.extend(variables)
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================

def train(sess, env, args, actor, critic, actor_noise):

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    dt = str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p"))
    writer = tf.summary.FileWriter(args['summary_dir'] + '_' + dt, sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    # Needed to enable BatchNorm. 
    # This hurts the performance on Pendulum but could be useful
    # in other environments.
    # tflearn.is_training(True)
    episode_length = 10
    minibatch_size = int(args['minibatch_size'])
    for i in range(int(args['max_episodes'])):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0
        action_avg = 0

        zero_episode = (np.reshape(s, (actor.s_dim,)),
                        np.reshape(np.zeros(actor.a_dim), (actor.a_dim,)),
                        np.array(0.),
                        np.array(False),
                        np.reshape(s, (actor.s_dim,)))
        history = deque([zero_episode] * episode_length)
        hidden_state = None
        for j in range(int(args['max_episode_len'])):

            if args['render_env']:
                env.render()

            # Added exploration noise
            a, hidden_state = actor.predict(np.reshape(s, (1, 1, actor.s_dim)), [1], 1, hidden_state)
            a = a + actor_noise()
            action_avg += np.squeeze(a)
            s2, r, terminal, info = env.step(a[0])

            history.popleft()
            history.append((np.reshape(s, (actor.s_dim,)), np.reshape(a[:, -1, :], (actor.a_dim,)), np.array(r),
                            np.array(terminal), np.reshape(s2, (actor.s_dim,))))
            if j >= episode_length:
                replay_buffer.add(np.array(history))
            s = s2
            ep_reward += r[0]

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                # Calculate targets
                target_action, _ = actor.predict_target(s2_batch, [episode_length] * minibatch_size, minibatch_size)
                target_q = critic.predict_target(s2_batch, target_action, [episode_length] * minibatch_size)

                # y_i = []
                # for k in range(int(args['minibatch_size'])):
                #     if t_batch[k]:
                #         y_i.append(r_batch[k])
                #     else:
                #         y_i.append(r_batch[k] + critic.gamma * target_q[k])

                t_q = np.multiply(-1 * np.array(t_batch).astype(int) + 1., np.squeeze(target_q))
                y_i = np.squeeze(r_batch) + critic.gamma * t_q

                # Update the critic given the targets
                predicted_q_value, _, critic_loss = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']),  episode_length, 1)), [episode_length] * minibatch_size)
                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs, _ = actor.predict(s_batch, [episode_length] * minibatch_size, minibatch_size)
                grads = critic.action_gradients(s_batch, a_outs, [episode_length] * minibatch_size)
                _, actor_grads = actor.train(s_batch, grads[0], [episode_length] * minibatch_size, minibatch_size)

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

                # print(critic_loss)
                # print(np.max(grads[0]), np.min(grads[0]))
                # print(np.max(actor_grads[0]), np.min(actor_grads[0]))

            s = s2
            ep_reward += r[0]

            if terminal:
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j),
                    summary_vars[2]: 0.,
                    summary_vars[3]: critic_loss,
                    summary_vars[4]: action_avg / j,
                    summary_vars[5]: grads[0],
                    summary_vars[6]: actor_grads[0],
                    summary_vars[7]: a_outs

                })

                writer.add_summary(summary_str, i)
                writer.flush()

                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.2f}| Critic_loss: {:.2f}'.format(int(ep_reward),
                                                                                                  i,
                                                                                                  (ep_ave_max_q / float(j)),
                                                                                                  critic_loss))
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
        # Ensure action bound is symmetric
        assert (env.action_space.high == -env.action_space.low)

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())
        
        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        if args['use_gym_monitor']:
            if not args['render_env']:
                env = wrappers.Monitor(
                    env, args['monitor_dir'], video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, args['monitor_dir'], force=True)

        train(sess, env, args, actor, critic, actor_noise)

        if args['use_gym_monitor']:
            env.monitor.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
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
