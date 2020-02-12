import csv
import pandas as pd
import numpy as np
import os
import warnings
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity, fname=None, init_data_fname=None, columns=None):
        self.capacity = capacity
        if columns is None:
            columns = ['state', 'action', 'reward', 'next_state', 'done']
        self.df = pd.DataFrame(columns=columns, index=range(capacity))
        self.last_position = 0
        self.position = 0
        self.full = False
        self.fname = fname
        # if fname == init_data_fname, then just load df and parse it (no need to append new header)
        if fname is not None and fname != init_data_fname:
            with open(self.fname, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(self.df.columns.values)

        if init_data_fname is not None:
            dump = init_data_fname != fname
            self.load_data_(init_data_fname, dump)

    def push(self, *args, dump=True):
        self.df.iloc[self.position] = pd.Series(*args).values
        self.last_position = self.position
        if self.position == self.capacity - 1:
            self.full = True
        self.position = (self.position + 1) % self.capacity

        if self.fname is not None and dump:
            self.save_rows_(self.fname, self.last_position, self.last_position+1)

    def sample(self, batch_size):
        end = self.capacity if self.full else self.position
        bacth_idx = np.random.randint(0, end, batch_size)
        batch = self.df.iloc[bacth_idx]
        # batch = random.sample(self.buffer, batch_size)
        res = map(np.stack, batch.T.values)
        return res

    def save_rows_(self, fname, row_idx_start, row_idx_end):
        with open(fname, 'a') as f:
            self.df.iloc[row_idx_start: row_idx_end].to_csv(f, mode='a', header=False, index=False)

    def load_data_(self, fname, dump):
        if os.path.exists(fname):
            with open(fname, 'r') as f:
                print(f"Load data from '{fname}'")
                dtype_dic = {'state': object, 'action': object, 'next_state': object, 'reward': float, 'done': bool}
                loaded_df = pd.read_csv(f, dtype=dtype_dic)
            if (loaded_df.columns.values != self.df.columns.values).any():
                raise ValueError("Column names don't match")

            # slow but correct as it doesn't mess with the last position pointer and also dumps all aded transitions to
            # newly created csv
            for i, r in loaded_df.iterrows():
                state = np.fromstring(r['state'][1:-1:1], sep=",")
                action = np.fromstring(r['action'][1:-1:1], sep=",")
                next_state = np.fromstring(r['next_state'][1:-1:1], sep=",")
                reward = r['reward']
                done = r['done']

                self.push(state, action, reward, next_state, done, dump)
        else:
            warnings.warn(f"File {fname} not found. Nothing has been loaded to replay buffer", RuntimeWarning)

    def __len__(self):
        return self.capacity if self.full else self.position


class SequenceReplayBuffer:
    def __init__(self, capacity, columns=None):
        if columns is None:
            columns = ['state', 'action', 'reward', 'next_state', 'done']
        self.columns = columns
        self.data = deque(maxlen=capacity)

    def append(self, episode):
        self.data.append(episode)

    def sample(self, batch_size):
        if len(self.data) < batch_size:
            batch = random.sample(self.data, len(self.data))
        else:
            batch = random.sample(self.data, batch_size)

        # s_batch = np.array([_[0] for _ in batch])
        # a_batch = np.array([_[1] for _ in batch])
        # r_batch = np.array([_[2] for _ in batch])
        # t_batch = np.array([_[3] for _ in batch])
        # s2_batch = np.array([_[4] for _ in batch])

        return batch

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    fname_load = r'C:\Study\Osim_rl_by_Melbrus\data\osim_sac_data_06_19_2019_11_57_AM12345.csv'
    fname=r'C:\Study\Osim_rl_by_Melbrus\data\12345_test.csv'
    rb = ReplayBuffer(capacity=100000, fname=fname)
    rb.load_data_(fname_load, False)