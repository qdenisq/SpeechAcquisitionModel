import json
import yaml
from pprint import pprint
import datetime
import copy
import os
from collections import defaultdict

import numpy as np

import matplotlib.pyplot as plt
import dtwalign

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from src.reinforcement_v2.envs.env import EnvironmentManager

from src.reinforcement_v2.common.replay_buffer import SequenceReplayBuffer
from src.reinforcement_v2.common.nn import SoftQNetwork, PolicyNetwork, ModelDynamics, DeterministicPolicyNetwork

from src.reinforcement_v2.common.tensorboard import DoubleSummaryWriter
from src.reinforcement_v2.common.noise import OUNoise
from src.reinforcement_v2.utils.timer import Timer

from src.soft_dtw_awe.model import SiameseDeepLSTMNet
from src.soft_dtw_awe.soft_dtw import SoftDTW



def soft_update(from_net, target_net, soft_tau=0.1):
    for target_param, param in zip(target_net.parameters(), from_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )


def get_empty_episode_entry():
    return {
            'goal': None,
            'states': [],
            'actions': [],
            'next_states': [],
            'done': []
        }


class SequentialBackpropIntoPolicy(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SequentialBackpropIntoPolicy, self).__init__()

        self._dt = str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p"))
        self._env_name = kwargs['env']['env_id']

        self.params = kwargs
        self.use_cuda = kwargs.get('use_cuda', True)
        self.device = torch.device("cuda" if self.use_cuda and torch.cuda.is_available() else "cpu")

        if "preproc_net" in kwargs:
            self.preproc_net = torch.load(kwargs["preproc_net"]['preproc_net_fname']).to(self.device)

        self.agent_state_dim = kwargs['agent_state_dim']
        self.state_dim = kwargs['state_dim']
        self.action_dim = kwargs['action_dim']
        self.reference_mask = kwargs['reference_mask']

        self.model_dynamics = ModelDynamics(**kwargs['model_dynamics']).to(self.device)
        self.model_dynamics_target = copy.deepcopy(self.model_dynamics)

        if kwargs["pretrained_policy"] is not None:
            self.policy_net = torch.load(kwargs['pretrained_policy']).policy_net
            self.policy_net.to(self.device)
        else:
            self.policy_net = DeterministicPolicyNetwork(**kwargs['policy_network']).to(self.device)

        self.use_alpha = kwargs.get('use_alpha', True)
        if not self.use_alpha:
            self.noise = OUNoise(kwargs['policy_network']['action_dim'] * kwargs['env']['num_workers'],
                                 kwargs['env']['seed'])
            self.noise_decay = kwargs['noise_decay']  # decay after each 1000 steps
            self.noise_level = 1.0
            self.noise_min = kwargs['noise_min']

        self.lr = kwargs['lr']

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.md_optimizer = optim.Adam(self.model_dynamics.parameters(), lr=self.lr)

        # init replay buffer
        self.replay_buffer_size = kwargs['replay_buffer']['size']

        self.replay_buffer = SequenceReplayBuffer(self.replay_buffer_size)

        self.gamma = kwargs.get('gamma', 0.99)
        self.num_updates_per_step = kwargs.get("num_updates_per_step", 30)
        self.train_step = 0
        self.step = 0

    def forward(self, *input):
        return self.policy_net(*input)

    def update(self, batch_size):
        # state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        batch = self.replay_buffer.sample(batch_size)
        audio_dim = env.get_attr('audio_dim')[0]

        """
        Model Dynamics Update
        """

        self.md_optimizer.zero_grad()
        md_loss = torch.tensor(0).float().to(self.device)
        num_samples = 0

        for sample in batch:
            state = torch.FloatTensor(sample['states']).to(self.device)
            next_state = torch.FloatTensor(sample['next_states']).to(self.device)
            action = torch.FloatTensor(sample['actions']).to(self.device)
            # reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
            done = torch.FloatTensor(np.float32(sample['done'])).unsqueeze(1).to(self.device)
            num_samples += state.shape[0]



            agent_state = state[:, :self.agent_state_dim]
            agent_next_state = next_state[:, :self.agent_state_dim]
            predicted_next_agent_state = self.model_dynamics(agent_state, action)

            md_loss += torch.nn.SmoothL1Loss(reduction="sum")(agent_next_state, predicted_next_agent_state)
            # md_loss += torch.nn.SmoothL1Loss(reduction="sum")(agent_next_state[:, :-audio_dim-6], predicted_next_agent_state[:, :-audio_dim-6])
            # md_loss += torch.sum(torch.abs(agent_next_state[:, :-audio_dim-6] - predicted_next_agent_state[:, :-audio_dim-6]))

        md_loss /= num_samples
        md_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model_dynamics.parameters(), 1)
        self.md_optimizer.step()
        # if self.train_step % 5 == 0:
        self.soft_update(self.model_dynamics, self.model_dynamics_target, 0.1)

        """
        Policy loss
        """

        self.md_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()

        policy_loss = torch.tensor(0.).float().to(self.device)

        for sample in batch:
            state = torch.FloatTensor(sample['states']).to(self.device)
            next_state = torch.FloatTensor(sample['next_states']).to(self.device)
            action = torch.FloatTensor(sample['actions']).to(self.device)
            # reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
            done = torch.FloatTensor(np.float32(sample['done'])).unsqueeze(1).to(self.device)

            ac_goal = torch.FloatTensor(sample['goal']['acoustics']).to(self.device) / 10.


            # TODO: add articulatory goal and loss
            # ar_goal =


            new_action = self.policy_net(state)
            agent_state = state[:, :self.agent_state_dim]
            predicted_next_agent_state = self.model_dynamics_target(agent_state, new_action)
            predicted_next_agent_state_masked = predicted_next_agent_state[:, self.reference_mask]
            state_masked = state[:, self.agent_state_dim:]

            ar_goal = state_masked[:, :-audio_dim]


            softDTW = SoftDTW(open_end=True, dist='l1')



            for i in range(1, state.shape[0]):
                predicted_seq = torch.cat([agent_state[:i, self.reference_mask], predicted_next_agent_state_masked[i,:].unsqueeze(0)], dim=0)

                predicted_seq_ac = predicted_seq[:, -audio_dim:].reshape(-1, ac_goal.shape[-1])

                predicted_seq_ar = predicted_seq[:, :-audio_dim]

                # 1. soft DTW loss acoustic
                # ac_loss = softDTW(predicted_seq_ac, ac_goal)

                # 2. L1 acoustic loss
                ac_loss = torch.nn.SmoothL1Loss(reduction="sum")(predicted_seq_ac[-1].unsqueeze(0), ac_goal[i].unsqueeze(0))



                # soft DTW loss articulatory
                # ar_loss = torch.nn.SmoothL1Loss(reduction="sum")(predicted_seq_ar[-1], ar_goal[-1])
                ar_loss = torch.nn.SmoothL1Loss(reduction="sum")(predicted_seq_ar[-1].unsqueeze(0), ar_goal[i].unsqueeze(0))
                # ar_loss = softDTW(predicted_seq_ar, ar_goal)

                # action penalty
                action_penalty = torch.sum(torch.abs(new_action[i]))
                # print(action_penalty)

                policy_loss += ac_loss
                policy_loss += ar_loss
                policy_loss += 0.01 * action_penalty


                # print(ac_loss)
                # print(ar_loss)

            # policy_loss /= state.shape[0]
        policy_loss /= num_samples

        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.policy_optimizer.step()

        self.train_step += 1
        return self.train_step, policy_loss.detach().cpu(), md_loss.detach().cpu()

    def soft_update(self, from_net, target_net, soft_tau):
        for target_param, param in zip(target_net.parameters(), from_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

    def add_weights_histogram(self, writer, global_step):
        for name, param in self.policy_net.named_parameters():
            writer.add_histogram("Policynet_" + name, param, global_step)

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries or unnecessary objects.
        del state['replay_buffer']
        return state

    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)
        # Do not restore replay buffer
        # self.replay_buffer = ReplayBuffer(self.replay_buffer_size,
        #                                   self.replay_buffer_csv_filename,
        #                                   self.replay_buffer_csv_filename)
        self.replay_buffer = SequenceReplayBuffer(self.replay_buffer_size)
        # Get device
        self.device = torch.device("cuda" if self.use_cuda and torch.cuda.is_available() else "cpu")
        # Move all the networks to device
        self.policy_net.to(self.device)
        self.model_dynamics.to(self.device)

    def train(self, env, **kwargs):
        """
        Initialize utility objects (e.g. summary writers, timers)
        """

        timer = defaultdict(Timer)
        writer = DoubleSummaryWriter(log_dir=f'../../../runs/{self._env_name}_backprop_{self._dt}/',
                                     light_log_dir=f'../../../runs_light/light_{self._env_name}_backprop_{self._dt}/',
                                     mode=kwargs['log_mode'])



        # TODO: only last called graph is added
        dummy_input = torch.rand(1, self.state_dim).to(self.device)
        writer.add_graph(self.policy_net, dummy_input)

        batch_size = kwargs.get('batch_size', 256)

        self.params['gamma'] = self.gamma
        self.params['use_alpha'] = self.use_alpha
        self.params['train'] = {
            'validate_every': kwargs['validate_every'],
            'sync_every': kwargs['sync_every'],
            'max_steps': kwargs['max_steps'],
            'batch_size': batch_size
        }

        writer.add_text('Hyper parameters', json.dumps(self.params, indent=4, sort_keys=True))

        """
        Train
        """

        local_buffer = [None]*kwargs['num_workers']

        rewards = []
        best_total_reward = -10000.
        reward_running = 0.
        while self.step < self.params['train']['max_steps']:

            if self.step % self.params['train']['validate_every'] == 0:
                self.validate_and_summarize(env, writer, **kwargs)

            if self.step % self.params['train']['sync_every'] == 0 or self.step % self.params['train']['validate_every'] == 0:
                ep_reward = np.zeros(kwargs['num_workers'])
                state = env.reset()
                env.render()
                for i in range(kwargs['num_workers']):
                    local_buffer[i] = get_empty_episode_entry()
                    local_buffer[i]['goal'] = env.get_attr('cur_reference')[i]

                if not self.use_alpha:
                    self.noise.reset()
                    self.noise_level = max(self.noise_min, self.noise_level * self.noise_decay)


                # ep_reward = np.zeros(kwargs['num_workers'])
                # state = env.reset()
                # env.render()
                # for i in range(kwargs['num_workers']):
                #     local_buffer[i] = get_empty_episode_entry()
                #     local_buffer[i]['goal'] = env.get_attr('cur_reference')[i]


            timer['algo'].start()
            action = self.policy_net.get_action(state)
            action = action.detach().cpu().numpy()
            if not self.use_alpha:
                action = action + self.noise.sample().reshape(*action.shape) * self.noise_level
                # just for testing
                # action *= 0
            action = np.clip(action, -1, 1)

            timer['algo'].stop()
            timer['env'].start()

            next_state, reward, done, info = env.step(action)

            # print("ERROR", np.max(np.clip(state[:, :24] + action / 5, -1, 1) - next_state[:, : 24]))

            # print(reward)
            # dtw_dist = info['dtw_dist']
            if kwargs['visualize']:
                env.render()

            timer['env'].stop()
            timer['replay_buffer'].start()

            for worker_idx in range(kwargs['num_workers']):
                # skip steps with undefined reward
                # if np.isnan(reward[worker_idx]):
                #
                #     continue
                # if done[worker_idx]:
                #     continue
                local_buffer[worker_idx]['actions'].append(action[worker_idx])
                local_buffer[worker_idx]['states'].append(state[worker_idx])
                local_buffer[worker_idx]['next_states'].append(next_state[worker_idx])
                local_buffer[worker_idx]['done'].append(done[worker_idx])

                # self.replay_buffer.push((list(state[worker_idx]),
                #                          list(action[worker_idx]),
                #                          reward[worker_idx],
                #                          list(next_state[worker_idx]),
                #                          done[worker_idx]))

            timer['replay_buffer'].stop()

            state = next_state
            ep_reward += reward
            self.step += 1



            timer['utils'].start()
            writer.add_histogram("Action", action, self.step)
            timer['utils'].stop()

            # if len(self.replay_buffer) > 3 * batch_size and self.frame_idx % 25 == 0:
            if len(self.replay_buffer) > 3 * batch_size:

                timer['algo'].start()
                for _ in range(self.num_updates_per_step):
                    _, policy_loss, md_loss = self.update(batch_size)
                timer['algo'].stop()

                timer['utils'].start()
                if not self.use_alpha:
                    writer.add_scalar('Noise_level', self.noise_level, self.step)
                writer.add_scalar('Policy_loss', policy_loss, self.step)
                writer.add_scalar('Model Dynamics Loss', md_loss, self.step)
                print(f"step:{self.step} | policy_loss: {policy_loss:.2f} | md_loss: {md_loss:.2f}")
                # explicitly remove all tensor-related variables just to ensure well-behaved memory management
                del policy_loss, md_loss, action
                timer['utils'].stop()


            # ADD termination condition:
            current_steps = env.get_attr('current_step')
            print(current_steps)
            for w in range(kwargs['num_workers']):
                # if reward[w] < 0.001 and current_steps[w] > 3: # dtw > 7
                #     done[w] = True

                #TODO: study different stop conditions
                last_path_point_diff = abs(info[w]['last_path_point'][0] - info[w]['last_path_point'][1])
                steps_made = len(local_buffer[w]['actions'])
                if (np.mean(info[w]['dtw_dist']) > 10000 or last_path_point_diff > 8) and steps_made >= 2 : # dtw > 7
                    # continue
                    done[w] = True

            if any(done):
                timer['utils'].start()
                for i in range(kwargs['num_workers']):
                    if done[i]:
                        self.replay_buffer.append(copy.deepcopy(local_buffer[i]))

                        # print(np.max(abs(np.array(self.replay_buffer.data[-1]['states'])[:, :24] - np.array(
                        #     self.replay_buffer.data[-1]['next_states'])[:, :24])))


                        local_buffer[i] = get_empty_episode_entry()

                        reward_running = 0.90 * reward_running + 0.10 * ep_reward[i]
                        rewards.append(ep_reward[i])
                        ep_reward[i] = 0.
                        state[i] = env.reset([i])
                        local_buffer[i]['goal'] = env.get_attr('cur_reference')[i]
                env.render()

                writer.add_scalar("Reward", reward_running, self.step)
                for k, v in timer.items():
                    writer.add_scalar(f"Elapsed time: {k}", v.elapsed_time, self.step)
                self.add_weights_histogram(writer, self.step)
                timer['utils'].stop()
                    # break

            # save best performing policy
            if self.step % 200 == 0:
                timer['utils'].start()
                # self.save_networks(episode_reward)
                name = f'{self._env_name}_BackpropIntoPolicy_' + f'{self.step}'
                save_dir = f'../../../models/{self._env_name}_backprop_{self._dt}/'
                try:
                    os.makedirs(save_dir)
                except:
                    pass
                path_agent = f'{save_dir}{name}.bp'
                torch.save(self, path_agent)
                timer['utils'].stop()
                print(f'step={self.step} | reward_avg={reward_running:.2f} | saving agent: {name}')
                if reward_running > best_total_reward:
                    best_total_reward = reward_running

            if self.step % 100 == 0:
                print(f'step={self.step} | reward_avg={reward_running:.2f} |')

        writer.close()

    def validate_and_summarize(self, env, writer, **kwargs):
        state = env.reset()
        env.render()

        episode_done = [False]*kwargs['num_workers']
        dtw_dists = [[] for _ in range(kwargs['num_workers'])]
        last_path_points = [[] for _ in range(kwargs['num_workers'])]
        vt_predictions = [[] for _ in range(kwargs['num_workers'])]
        embeds_predictions = [[] for _ in range(kwargs['num_workers'])]
        while not all(episode_done):
            action = self.policy_net.get_action(state)
            action = action.detach().cpu().numpy()
            action = np.clip(action, -1, 1)
            next_state, reward, done, info = env.step(action)
            agent_state = state[:, :self.agent_state_dim]


            predicted_next_agent_state = self.model_dynamics_target(torch.from_numpy(agent_state).float().to(self.device),
                                       torch.from_numpy(action).float().to(self.device)).detach().cpu().numpy()
            audio_dim = env.get_attr('audio_dim')[0]
            vt_pred = predicted_next_agent_state[:, :-audio_dim]
            embeds_pred = predicted_next_agent_state[:, -audio_dim:]

            # print(reward)
            # dtw_dist = info['dtw_dist']
            env.render()

            state = next_state

            for worker_idx in range(kwargs['num_workers']):
                dtw_dists[worker_idx].append(info[worker_idx]['dtw_dist'])
                last_path_points[worker_idx].append(info[worker_idx]['last_path_point'])
                vt_predictions[worker_idx].append(vt_pred[worker_idx])
                embeds_predictions[worker_idx].append((embeds_pred[worker_idx]))


                if done[worker_idx]:
                    episode_done[worker_idx] = True

                    name = f'{self._env_name}_BackpropIntoPolicy_' + f'{self.step}'
                    save_dir = f'../../../runs/{self._env_name}_backprop_{self._dt}/'
                    fname = os.path.join(save_dir, name + ".mp4")
                    fnames = env.dump_episode(fname=os.path.join(save_dir, name))

                    episode_history = env.get_episode_history(remotes=[worker_idx])[0]
                    video_data = torchvision.io.read_video(fnames[0]+".mp4", start_pts=0, end_pts=None, pts_unit='sec')

                    dtw_res = dtwalign.dtw(episode_history['embeds'], episode_history['ref']['acoustics'], dist=kwargs['distance']['dist'],
                                 step_pattern=kwargs['distance']['step_pattern'],
                                 open_end=False)

                    #prepare predictions
                    vt_preds = np.array(vt_predictions[worker_idx])
                    embeds_preds = np.array(embeds_predictions[worker_idx]).reshape(-1, episode_history['embeds'].shape[-1])
                    if worker_idx == 0:
                        self.summarize(writer,
                                       episode_history,
                                       video_data,
                                       dtw_res,
                                       dtw_dists[worker_idx],
                                       last_path_points[worker_idx],
                                       vt_preds,
                                       embeds_preds)
                        state[worker_idx] = env.reset([worker_idx])


    def summarize(self,
                  writer,
                  episode_history,
                  video_data, dtw_res,
                  dtw_dists,
                  last_path_points,
                  vt_predictions,
                  embeddings_predictions):

        # agent

        fig = plt.figure()
        plt.matshow(episode_history['mfcc'].T, 0)
        plt.colorbar()
        writer.add_figure('agent/MFCC', fig, self.step)

        fig = plt.figure()
        plt.matshow(episode_history['embeds'].T, 0)
        plt.colorbar()
        writer.add_figure('agent/Embeddings', fig, self.step)

        fig = plt.figure()
        plt.matshow(episode_history['vt'].T[:-6], 0)
        plt.colorbar()
        writer.add_figure('agent/VocalTract', fig, self.step)

        # reference

        fig = plt.figure()
        plt.matshow(episode_history['ref']['mfcc'].squeeze().T, 0)
        plt.colorbar()
        writer.add_figure('reference/MFCC', fig, self.step)

        fig = plt.figure()
        plt.matshow(episode_history['ref']['acoustics'].T, 0)
        plt.colorbar()
        writer.add_figure('reference/Embeddings', fig, self.step)

        fig = plt.figure()
        plt.matshow(episode_history['ref']['vt'].T[:-6], 0)
        plt.colorbar()
        writer.add_figure('reference/VocalTract', fig, self.step)

        if 'video_fps' in video_data[2]:
            writer.add_video('agent_video/vt_video', video_data[0].unsqueeze(0).permute(0,1,4,2,3), fps=video_data[2]['video_fps'], global_step=self.step)

        if 'audio_fps' in video_data[2]:
            writer.add_audio('agent/audio', video_data[1], sample_rate=video_data[2]['audio_fps'], global_step=self.step)
            writer.add_audio('ref/audio', episode_history['ref']['audio'].flatten(), sample_rate=video_data[2]['audio_fps'], global_step=self.step)

        # DTW

        fig, ax = dtw_res.plot_path()
        writer.add_figure('DTW/Embeddings', fig, self.step)

        fig = plt.figure()
        plt.plot(np.array(dtw_dists))
        writer.add_figure('DTW/distance_over_episode', fig, self.step)

        fig = plt.figure()
        points = np.array(last_path_points)
        plt.plot(points[:, 0], points[:, 1])
        writer.add_figure('DTW/path', fig, self.step)

        # predictions

        fig = plt.figure()
        plt.matshow(vt_predictions.T[:-6], 0)
        plt.colorbar()
        writer.add_figure('predictions/VocalTract', fig, self.step)

        fig = plt.figure()
        plt.matshow(embeddings_predictions.T, 0)
        plt.colorbar()
        writer.add_figure('predictions/Embeddings', fig, self.step)

        # agent error

        fig = plt.figure()
        err_mfcc = episode_history['mfcc'] - episode_history['ref']['mfcc'].squeeze()[:episode_history['mfcc'].shape[0], :]
        plt.matshow(err_mfcc.squeeze().T, 0)
        plt.colorbar()
        writer.add_figure('agent_error/MFCC', fig, self.step)

        fig = plt.figure()
        err_embeds = episode_history['embeds'] - episode_history['ref']['acoustics'][:episode_history['embeds'].shape[0], :]
        plt.matshow(err_embeds.T, 0)
        plt.colorbar()
        writer.add_figure('agent_error/Embeddings', fig, self.step)

        fig = plt.figure()
        err_vt = episode_history['vt'].T[:-6] - episode_history['ref']['vt'].T[:-6]
        plt.matshow(err_vt, 0)
        plt.colorbar()
        writer.add_figure('agent_error/VocalTract', fig, self.step)

        # agent error totals

        fig = plt.figure()
        err_mfcc_sum = np.sum(abs(err_mfcc), axis=-1)
        plt.plot(err_mfcc_sum)
        writer.add_figure('agent_error/MFCC_sum', fig, self.step)

        fig = plt.figure()
        err_embeds_sum = np.sum(abs(err_embeds), axis=-1)
        plt.plot(err_embeds_sum)
        writer.add_figure('agent_error/Embeddings_sum', fig, self.step)

        fig = plt.figure()
        err_vt_sum = np.sum(abs(err_vt), axis=0)
        plt.plot(err_vt_sum)
        writer.add_figure('agent_error/VocalTract_sum', fig, self.step)







if __name__ == '__main__':
    with open('../configs/SoftDTWBackpropIntoPolicy_e0.yaml', 'r') as data_file:
        kwargs = yaml.safe_load(data_file)
    pprint(kwargs)

    seed = kwargs['seed']
    kwargs['env']['seed'] = seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # create env
    env_mgr = EnvironmentManager()
    env_kwargs = copy.deepcopy(kwargs['env'])
    env_args = [kwargs['env']['lib_path'], kwargs['env']['speaker_fname']]
    env_id = env_kwargs.pop('env_id')
    env = env_mgr.make(env_id, *env_args, **env_kwargs)

    kwargs['train'].update(kwargs['env'])
    kwargs['train']['collect_data'] = kwargs['collect_data']
    kwargs['train']['log_mode'] = kwargs['log_mode']

    kwargs['train']['distance'] = kwargs['env']['distance']

    if kwargs["init_data_fname"] is not None:
        kwargs['train']['init_data_fname'] = kwargs['init_data_fname']

    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    agent_state_dim = env.get_attr('agent_state_dim')[0]
    reference_mask = env.get_attr('reference_mask')[0]
    acoustics_dim = env.get_attr('audio_dim')[0]

    kwargs['action_dim'] = action_dim
    kwargs['agent_state_dim'] = agent_state_dim
    kwargs['state_dim'] = state_dim
    kwargs['reference_mask'] = reference_mask

    kwargs['model_dynamics']['agent_state_dim'] = agent_state_dim
    kwargs['model_dynamics']['action_dim'] = action_dim
    kwargs['model_dynamics']['acoustic_dim'] = acoustics_dim

    kwargs['policy_network']['state_dim'] = state_dim
    kwargs['policy_network']['action_dim'] = action_dim

    if 'agent_fname' in kwargs:
        # load agent
        agent_fname = kwargs['agent_fname']
        print(f'Loading agent from "{agent_fname}"')
        agent = torch.load(kwargs['agent_fname'])
        if not kwargs['use_alpha']:
            agent.noise_level = kwargs['noise_init_level']
            agent.noise = OUNoise(kwargs['soft_q_network']['action_dim'] * kwargs['env']['num_workers'],
                                     kwargs['env']['seed'])

    else:
        # create agent
        agent = SequentialBackpropIntoPolicy(**kwargs)

    # train

    run_dir = f'../../../runs/{agent._env_name}_backprop_{agent._dt}/'
    try:
        os.makedirs(run_dir)
    except:
        pass
    with open(os.path.join(run_dir, 'md_backprop.yaml'), 'w') as f:
        yaml.dump(kwargs, f)


    agent.train(env, **kwargs['train'])
