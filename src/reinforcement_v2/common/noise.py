import copy
import random
import numpy as np

import torch
import torch.distributions as distrib
import torch.nn as nn


def create_noise(**kwargs):
    if kwargs['type'] == 'state_action':
        return StateActionNoise(**kwargs)
    elif kwargs['type'] == 'ou':
        return OUNoise(**kwargs)
    else:
        return None


class StateActionNoise(nn.Module):
    def __init__(self, *args, **kwargs):
        super(StateActionNoise, self).__init__()

        self.max_sigma = kwargs['max_sigma']
        self.state_dim = kwargs['state_dim']
        self.action_dim = kwargs['action_dim']
        self.hidden_dim = kwargs['hidden_dim']

        self.linears = nn.ModuleList([nn.Linear(self.state_dim + self.action_dim, self.hidden_dim[0])])
        self.linears.extend(
            [nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]) for i in range(1, len(self.hidden_dim))])

        self.last_fc = nn.Linear(self.hidden_dim[-1], self.action_dim)

        self.relu = nn.ReLU()

    def forward(self, states, actions):

        x = torch.cat((states[:, :self.state_dim], actions), -1)
        for linear in self.linears:
            x = self.relu(linear(x))

        sigma = torch.clamp(self.relu(self.last_fc(x)), min=0.0005, max=self.max_sigma)

        dists = distrib.normal.Normal(torch.zeros(sigma.shape), sigma)

        return dists

    def sample(self, states, actions):
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).float()
        dists = self(states, actions)
        return dists.sample().detach().cpu().numpy()

    def log_prob(self, states, actions):
        dists = self(states, actions)
        return dists.log_prob(actions)
        # self.device = torch.device("cuda" if self.use_cuda and torch.cuda.is_available() else "cpu")


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return np.clip(self.state, -1, 1)**3
