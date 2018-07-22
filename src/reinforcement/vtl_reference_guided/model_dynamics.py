import os
import datetime
import random
from collections import deque
from random import randrange
import math

import numpy as np
from scipy import spatial
import tensorflow as tf
from tensorflow.python.keras.initializers import RandomUniform
from tensorflow.python.keras.layers import Dense, Input, BatchNormalization, Activation, Concatenate

class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s0, a, s1):
        if self.count < self.buffer_size:
            self.buffer.append((s0, a, s1))
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append((s0, a, s1))

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s0_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        s1_batch = np.array([_[2] for _ in batch])

        return s0_batch, a_batch, s1_batch

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
        self.a_dim = model_settings['action_dim']
        self.action_bound = model_settings['action_bound']
        self.learning_rate = model_settings['model_dynamics_learning_rate']
        self.batch_size = model_settings['minibatch_size']
        self.state_goal_gamma = 0.5
        self.se_cos_gamma = 0.5


        y_max = [y[1] for y in self.state_bound]
        y_min = [y[0] for y in self.state_bound]
        self._k_state = 2. / (np.subtract(y_max, y_min))
        self._b_state = -0.5 * np.add(y_max, y_min) * self._k_state


        # Model Dynamics Network
        with tf.variable_scope(self.name + 'model_dynamics'):
            self.inputs_state,\
            self.inputs_action,\
            self.state_out,\
            self.scaled_state_out,\
                = self.create_model_dynamics_network()
            self.network_params = tf.trainable_variables(scope=self.name + '_model_dynamics')

        self.ground_truth_state_out = tf.placeholder(tf.float32, [None, self.s_dim])

        # Optimization Op
        self.state_abs_loss = tf.losses.absolute_difference(self.ground_truth_state_out, self.scaled_state_out)
        self.state_cos_loss = tf.losses.cosine_distance(tf.nn.l2_normalize(self.ground_truth_state_out - self.inputs_state, axis=1),
                                                  tf.nn.l2_normalize(self.scaled_state_out - self.inputs_state, axis=1),
                                                  axis=1)
        self.state_mse_loss = tf.losses.mean_squared_error(self.ground_truth_state_out, self.scaled_state_out, reduction=tf.losses.Reduction.MEAN)

        # self.state_loss = tf.add(self.state_abs_loss * self.se_cos_gamma, self.state_cos_loss * (1. - self.se_cos_gamma))

        self.loss = self.state_mse_loss

        self.grads = tf.train.GradientDescentOptimizer(self.learning_rate).compute_gradients(self.loss)

        self.optimize = tf.train.GradientDescentOptimizer(
            self.learning_rate).minimize(self.loss)

        self.num_trainable_vars = len(
            self.network_params)

    def create_model_dynamics_network(self):
        state_x = tf.placeholder(tf.float32, [None, self.s_dim])
        action_x = tf.placeholder(tf.float32, [None, self.s_dim])
        net = tf.concat([state_x, action_x], axis=1)
        self.conc = net
        n_hidden_0 = 100
        w0 = tf.Variable(tf.random_normal([net.get_shape().as_list()[1], n_hidden_0]))
        b0 = tf.Variable(tf.random_normal([n_hidden_0]))
        net = tf.add(tf.matmul(net, w0), b0)
        net = tf.nn.tanh(net)

        n_out = self.s_dim
        w1 = tf.Variable(tf.random_normal([net.get_shape().as_list()[1], n_out]))
        b1 = tf.Variable(tf.random_normal([n_out]))
        net = tf.add(tf.matmul(net, w1), b1)
        net = tf.nn.tanh(net)

        state_y = net
        state_y_scaled = state_y
        #
        # state_x = Input(batch_shape=[None, self.s_dim])
        # action_x = Input(batch_shape=[None, self.a_dim])
        #
        # net = Concatenate()([state_x, action_x])
        # self.conc = net
        # net = Dense(128, activation='tanh')(net)
        # state_y = Dense(self.s_dim, activation='tanh')(net)
        # state_y_scaled = state_y
        return state_x, action_x, state_y, state_y_scaled

    def calc_loss(self, inputs_state, inputs_action, ground_truth_state_out):
        return self.sess.run([self.loss, self.state_abs_loss, self.state_cos_loss, self.scaled_state_out], feed_dict={
                self.inputs_state: inputs_state,
                self.inputs_action: inputs_action,
                self.ground_truth_state_out: ground_truth_state_out,
        })

    def train(self, inputs_state, inputs_action, ground_truth_state_out):
        return self.sess.run([self.optimize, self.loss, self.state_abs_loss, self.state_cos_loss,  self.state_mse_loss, self.scaled_state_out, self.conc, self.grads ], feed_dict={
                self.inputs_state: inputs_state,
                self.inputs_action: inputs_action,
                self.ground_truth_state_out: ground_truth_state_out,
        })

    def predict(self, inputs_state, inputs_action):
        return self.sess.run(self.scaled_state_out, feed_dict={
            self.inputs_state: inputs_state,
            self.inputs_action: inputs_action
        })

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


def build_summaries():

    model_dynamics_loss = tf.Variable(0.)
    tf.summary.scalar("Model dynamics loss", model_dynamics_loss)


    md_abs_loss = tf.Variable(0.)
    tf.summary.scalar("Absolute difference", md_abs_loss)


    md_cos_dist_loss = tf.Variable(0.)
    tf.summary.scalar("Cosine distance", md_cos_dist_loss)


    mse_loss = tf.Variable(0.)
    tf.summary.scalar("Mean squared error", mse_loss)

    variables = [v for v in tf.trainable_variables()]
    [tf.summary.histogram(v.name, v) for v in variables]

    summary_vars = [model_dynamics_loss, md_abs_loss, md_cos_dist_loss, mse_loss]
    # episode_reward, episode_ave_max_q, actor_ep_loss, critic_loss, avg_action, act_grads, actor_grads, actor_activations]

    summary_vars.extend(variables)
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

def train(settings, model_dynamics, replay_buffer, reference_trajectory):

    sess = tf.get_default_session()
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    dt = str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p"))
    writer = tf.summary.FileWriter(settings['summary_dir'] + '/summary_md_' + dt, sess.graph)

    action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(settings['state_dim']), sigma=0.005)
    s_dim = settings['state_dim']

    while replay_buffer.size() < 50000:
        # pick random initial state from the reference trajectory
        s0_index = randrange(0, reference_trajectory.shape[0] - 1)
        s0 = reference_trajectory[s0_index]

        r = []
        # rollout episode
        for j in range(s0_index, len(reference_trajectory) - 1):
            target = reference_trajectory[j + 1]
            # add noise
            a_noise = action_noise()
            action = a_noise
            action = np.reshape(action, (s_dim))
            # make a step
            # temp. just check of md net
            s1 = s0 + action
            # s1 = np.clip(s1, -1., 1.)
            replay_buffer.add(s0, action, s1)
            s0 = s1

    num_train_steps = 50000
    for i in range(num_train_steps):
        # train model_dynamics and policy
        minibatch_size = settings['minibatch_size']
        if replay_buffer.size() > minibatch_size:
            s0_batch, a_batch, s1_batch = \
                replay_buffer.sample_batch(minibatch_size)

            # train model_dynamics
            # md_loss, md_goal_loss, md_abs_loss, md_cos_loss, s1_pred1 = (0, 0, 0, 0, 0)
            _, md_loss, md_abs_loss, md_cos_loss, md_mse_loss, s1_pred1, conc, grads = model_dynamics.train(s0_batch, a_batch, s1_batch)
            if i % 200 == 0:
                s1_pred = model_dynamics.predict(s0_batch, a_batch)
                ds_pred = s1_pred - s0_batch
                ds = s1_batch - s0_batch
                cos_dist = np.mean([spatial.distance.cosine(ds_pred[e], ds[e]) for e in range(minibatch_size)])
                print("cosine distance: ", cos_dist)

            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: md_loss,
                summary_vars[1]: md_abs_loss,
                summary_vars[2]: md_cos_loss,
                summary_vars[3]: md_mse_loss,
            })

            writer.add_summary(summary_str, i)
            writer.flush()

            print(' Episode: {:d} |'
            ' Abs diff loss: {:.4f}|'
            ' Cos dist loss: {:.4f}|'
            ' MSE loss: {:.4f}|'
                  ' Model dynamics loss: {:.4f}|'.format(i,
                                                         md_abs_loss,
                                                         md_cos_loss,
                                                         md_mse_loss,
                                                         md_loss))


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
    s_dim = 30
    state_bound = [(-1., 1)]*s_dim
    # state_bound[0] = (0, 8000)
    settings = {
            'state_dim': s_dim,
            'state_bound': state_bound,
            'action_dim': s_dim,
            'action_bound': state_bound,

            'minibatch_size': 2048,

            'model_dynamics_learning_rate': 0.1,

            'summary_dir': r'C:\Study\SpeechAcquisitionModel\reports\summaries',
            'videos_dir': r'C:\Study\SpeechAcquisitionModel\reports\videos'
        }
    with tf.Session() as sess:
        md = ModelDynamics('MD1', settings)
        replay_buffer = ReplayBuffer(100000)

        sess.run(tf.global_variables_initializer())
        reference_fname = r'C:\Study\SpeechAcquisitionModel\src\VTL\references\a_i.pkl'

        action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(settings['state_dim']), sigma=0.01)

        target_trajectory = [action_noise() for _ in range(50)]
        target_trajectory = np.cumsum(target_trajectory, axis=0)
        train(settings, md, replay_buffer, target_trajectory)
    return


if __name__ == '__main__':
    np.random.seed(1234)
    tf.set_random_seed(1234)
    main()


