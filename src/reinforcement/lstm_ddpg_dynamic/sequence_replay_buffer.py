""" 
Data structure for implementing experience replay

Author: Patrick Emami
"""
from collections import deque
import random
import numpy as np

class ReplayBuffer(object):

    def __init__(self, buffer_size, ep_length, random_seed=123):
        """
        The right side of the deque contains the most recent experiences 
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        self.episode_length = ep_length
        random.seed(random_seed)

    def add(self, h):
        if self.count < self.buffer_size:
            self.buffer.append(h)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(h)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([np.array([sample_ep[0] for sample_ep in history]) for history in batch])
        a_batch = np.array([np.array([sample_ep[1] for sample_ep in history]) for history in batch])
        r_batch = np.array([[sample_ep[2] for sample_ep in history] for history in batch])
        t_batch = np.array([np.array([sample_ep[3] for sample_ep in history]) for history in batch])
        s2_batch = np.array([np.array([sample_ep[4] for sample_ep in history]) for history in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0


