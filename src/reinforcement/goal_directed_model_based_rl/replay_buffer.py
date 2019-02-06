import random
from collections import deque
import numpy as np

class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=0):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, sample):
        if self.count < self.buffer_size:
            self.buffer.append(sample)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(sample)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        res = [np.array(t) for t in zip(*batch)]

        return tuple(res)

    def clear(self):
        self.buffer.clear()
        self.count = 0
