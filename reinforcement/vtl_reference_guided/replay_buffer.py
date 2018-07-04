from collections import deque
import random
import numpy as np

class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s0, a, s1, target, reward):
        if self.count < self.buffer_size:
            self.buffer.append((s0, a, s1, target, reward))
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append((s0, a, s1, target, reward))

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
        target_batch = np.array([_[3] for _ in batch])
        reward_batch = np.array([_[4] for _ in batch])

        return s0_batch, a_batch, s1_batch, target_batch, reward_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0
