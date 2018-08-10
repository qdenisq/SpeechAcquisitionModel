from random import seed
from random import random as rnd
from random import randrange
import random
import datetime
import os
import pickle
from collections import deque
from math import sqrt
from sklearn.metrics import mean_squared_error
import numpy as np
import tensorflow as tf
from src.VTL.vtl_environment import VTLEnv

class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=0):
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

ds_norm_largest = 1000.
ds_largest = 1000.
largest = 1000.


def normalize(data, bound):
    largest = np.array([max(abs(y[0]), abs(y[1])) for y in bound])
    normed_data = data / largest
    return normed_data


def denormalize(normed_data, bound):
    largest = np.array([max(abs(y[0]), abs(y[1])) for y in bound])
    data = normed_data * largest
    return data



def train(settings, env, replay_buffer, reference_trajectory):

    # setup all summaries and dump directories

    num_episodes = 50000
    action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(env.action_dim), sigma=0.001)
    s_dim = settings['state_dim']
    g_dim = settings['goal_dim']
    a_dim = settings['action_dim']
    s_bound = settings['state_bound']
    a_bound = settings['action_bound']
    g_bound = settings['goal_bound']

    ###########################################
    # Create model dynamics and policy
    ###########################################


    ################################################################
    # Model Dynamics
    ################################################################

    # create model dynamics

    # specify input placeholders
    md_input = tf.placeholder(tf.float32, [None, s_dim * 3])

    # specify network structure
    md_initializer = tf.glorot_uniform_initializer()
    md_dense1 = tf.layers.dense(inputs=md_input, units=6 * s_dim, kernel_initializer=md_initializer)
    md_dense2 = tf.layers.dense(inputs=md_dense1, units=4 * s_dim, kernel_initializer=md_initializer)

    # specify output
    md_output = tf.layers.dense(inputs=md_dense2, units=2 * s_dim, kernel_initializer=md_initializer)

    # network target for training
    md_target = tf.placeholder(tf.float32, [None, s_dim * 2])

    # loss
    md_loss = tf.losses.mean_squared_error(md_target, md_output)

    # train step
    md_global_step = tf.train.create_global_step()
    md_train_step = tf.train.AdadeltaOptimizer(learning_rate=1.0).minimize(loss=md_loss, global_step=md_global_step)

    ################################################################
    # Policy
    ################################################################

    # create policy
    with tf.variable_scope('policy'):
        # specify input placeholders
        policy_input = tf.placeholder(tf.float32, [None, s_dim * 3])

        # specify network structure
        policy_initializer = tf.glorot_uniform_initializer()
        policy_dense1 = tf.layers.dense(inputs=policy_input, units=6 * s_dim, kernel_initializer=policy_initializer)
        policy_dense2 = tf.layers.dense(inputs=policy_dense1, units=3 * s_dim, kernel_initializer=policy_initializer)

        # specify output
        policy_output = tf.layers.dense(inputs=policy_dense2, units=1 * s_dim, kernel_initializer=policy_initializer)

        # network target for training
        policy_target = tf.placeholder(tf.float32, [None, s_dim * 1])

        # trainable variables
        trainable_vars = tf.trainable_variables('policy')

        # train step
        policy_global_step = tf.Variable(0., trainable=False)

        ############################################
        # Policy gradient based optimization routine
        ############################################

        policy_md_loss = tf.losses.mean_squared_error(md_target[:, s_dim:], md_output[:, s_dim:])
        action_grads = tf.gradients(ys=policy_md_loss, xs=md_input)[0][:, 2*s_dim:]
        policy_opt = tf.train.AdadeltaOptimizer(learning_rate=1.0)
        optimize_op = policy_opt.minimize(loss=policy_output, global_step=policy_global_step, var_list=trainable_vars,
                                          grad_loss=action_grads)

    # Open the session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(tf.global_variables())
    dt = str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p"))
    writer = tf.summary.FileWriter(settings['summary_dir'] + '/summary_md_' + dt, sess.graph)
    video_dir = settings['videos_dir'] + '/video_md_' + dt
    try:
        os.makedirs(video_dir)
    except:
        print("directory '{}' already exists")

    ###########################################
    # main train loop
    ###########################################
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
            policy_X = np.concatenate((s0_normed, g0_normed, target_normed), axis=1)
            feed_dict = {policy_input: policy_X}
            action_normed = sess.run(policy_output, feed_dict=feed_dict)
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
            s1 = env.step(action)
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
            # some sort of early stopping
            # Need to change for sure in order to avoid going out of bounds at least
            if max(miss) > 0.1 and i % 200 != 0:
                break
        # dump video
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

            # train model dynamics
            md_input_X = np.concatenate((s0_batch_normed,
                                         g0_batch_normed,
                                         a_batch_normed), axis=1)
            md_target_Y = np.concatenate((s1_batch_normed,
                                          g1_batch_normed), axis=1)
            feed_dict = {md_input: md_input_X, md_target: md_target_Y}
            _, md_loss_out = sess.run([md_train_step, md_loss], feed_dict=feed_dict)



            #############################################################
            # train policy
            ##############################################################

            # predict actions
            policy_input_X = np.concatenate((s0_batch_normed,
                                             g0_batch_normed,
                                             target_batch_normed), axis=1)
            actions_normed = sess.run(policy_output, feed_dict={policy_input: policy_input_X})

            # predict state if predicted actions will be applied
            md_input_X = np.concatenate((s0_batch_normed,
                                         g0_batch_normed,
                                         actions_normed), axis=1)
            md_target_Y = np.concatenate((s1_batch_normed,
                                         target_batch_normed), axis=1)

            feed_dict = {policy_input: policy_input_X, md_input: md_input_X, md_target: md_target_Y}
            optimize_op.run(feed_dict=feed_dict)

            expected_actions_normed = target_batch_normed - g0_batch_normed
            policy_loss_out = np.mean(np.sum(np.square(expected_actions_normed - actions_normed), axis=1), axis=0)

            print("| train step: {}| model_dynamics loss: {:.8f}| policy loss: {:.5f}".format(i, md_loss_out, policy_loss_out))
            # summary_str = sess.run(summary_ops, feed_dict={
            #     summary_vars[0]: policy_se_loss,
            #     summary_vars[1]: md_loss,
            #     summary_vars[2]: md_goal_loss,
            #     summary_vars[3]: 0
            # })
            #
            # writer.add_summary(summary_str, i)
            # writer.flush()


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

    replay_buffer = ReplayBuffer(100000)

    reference_fname = r'C:\Study\SpeechAcquisitionModel\src\VTL\references\a_i.pkl'
    with open(reference_fname, 'rb') as f:
        (tract_params, glottis_params) = pickle.load(f)
        target_trajectory = np.hstack((np.array(tract_params), np.array(glottis_params)))
    train(settings, env, replay_buffer, target_trajectory)
    return


if __name__ == '__main__':
    np.random.seed(1)
    tf.set_random_seed(1)
    main()