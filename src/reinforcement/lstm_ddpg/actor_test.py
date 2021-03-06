import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import tflearn
import argparse
import pprint as pp
from reinforcement.model_based_ddpg import mbddpg

from reinforcement.model_based_ddpg.replay_buffer import ReplayBuffer


def train(sess, env, args, actor, actor_noise, gs_noise):
    # Set up summary Ops
    summary_ops, summary_vars = mbddpg.build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

    # Initialize target network weights
    actor.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    # Needed to enable BatchNorm.
    # This hurts the performance on Pendulum but could be useful
    # in other environments.
    # tflearn.is_training(True)

    for i in range(int(args['max_episodes'])):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0
        loss = 0

        for j in range(int(args['max_episode_len'])):

            if args['render_env']:
                env.render()
            gs = s + gs_noise()
            # a = (np.random.rand(actor.a_dim) - 0.5)
            # a_scaled_out = np.multiply(a, 2. * actor.action_bound)
            # a = a_scaled_out
            # a = actor_noise()
            # a = np.expand_dims(a, axis=0)

            a = actor.predict(np.reshape(s, (1, actor.s_dim)),
                              np.reshape(gs, (1, actor.s_dim))) + actor_noise()
            # print(actor.a_dim)
            # # Added exploration noise
            # #a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
            # a = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()

            s2, r, terminal, info = env.step(a[0])

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))


                # Update actor
                loss, _ = actor.train(s_batch, s2_batch, a_batch)

                actor.update_target_network()

            s = s2
            ep_reward += r

            if terminal:
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j),
                    summary_vars[2]: loss
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), \
                                                                             i, (ep_ave_max_q / float(j))))
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

        actor = mbddpg.ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']),
                             0,
                             0)

        actor_noise = mbddpg.OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        gs_noise = mbddpg.OrnsteinUhlenbeckActionNoise(mu=np.zeros(state_dim))


        if args['use_gym_monitor']:
            if not args['render_env']:
                env = wrappers.Monitor(
                    env, args['monitor_dir'], video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, args['monitor_dir'], force=True)

        train(sess, env, args, actor, actor_noise, gs_noise)

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

