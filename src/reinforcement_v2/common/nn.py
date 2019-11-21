import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.init import xavier_uniform_


def init_weights_xavier(m):
    r"""
    Applies xavier initialization to Linear layers in your neural net
    :param m:
    :return:

    Examples::

        >>> my_net = nn.Linear(1000,1)
        >>> my_net = my_net.apply(init_weights_xavier)
    """

    if type(m) == nn.Linear:
        xavier_uniform_(m.weight, gain=1)


class SoftQNetwork(nn.Module):
    def __init__(self, *args, init_w=3e-3, **kwargs):
        super(SoftQNetwork, self).__init__()
        self.state_dim = kwargs['state_dim']
        self.action_dim = kwargs['action_dim']
        self.hidden_dim = kwargs['hidden_dim']
        self.linears = nn.ModuleList([nn.Linear(self.state_dim + self.action_dim, self.hidden_dim[0])])
        self.linears.extend([nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]) for i in range(1, len(self.hidden_dim))])
        self.linears.extend([nn.Linear(self.hidden_dim[-1], 1)])

        self.apply(init_weights_xavier)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        for l in self.linears[:-1]:
            x = l(x)
            x = F.tanh(x)
        x = self.linears[-1](x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super(PolicyNetwork, self).__init__()

        self.state_dim = kwargs['state_dim']
        self.action_dim = kwargs['action_dim']
        self.hidden_dim = kwargs['hidden_dim']
        self.log_std_min = kwargs.get('log_std_min', -10)
        self.log_std_max = kwargs.get('log_std_max', -1)

        self.linears = nn.ModuleList([nn.Linear(self.state_dim, self.hidden_dim[0])])
        self.linears.extend(
            [nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]) for i in range(1, len(self.hidden_dim))])

        self.mean_linear = nn.Linear(self.hidden_dim[-1], self.action_dim)
        self.log_std_linear = nn.Linear(self.hidden_dim[-1], self.action_dim)
        self.apply(init_weights_xavier)
        # with torch.no_grad():
        #     self.mean_linear.bias = torch.nn.Parameter(-1 * torch.ones(self.mean_linear.bias.shape))
        # with torch.no_grad():
        #     self.log_std_linear.bias = torch.nn.Parameter(-3 * torch.ones(self.log_std_linear.bias.shape))

    def forward(self, state):
        x = state

        for l in self.linears:
            x = l(x)
            x = F.tanh(x)

        mean = self.mean_linear(x)
        mean = F.tanh(mean)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(torch.zeros(mean.shape),
                        torch.ones(mean.shape))
        z = normal.sample().to(mean.device)
        z.requires_grad_()
        # TODO: check distribution (done for sparsity)
        action = F.tanh(mean + std * z)**2
        der = 2 * F.tanh(mean + std * z) * (1 - action) * ((mean + std * z) > 0).float()
        log_prob = Normal(mean, std).log_prob(mean + std * z) - torch.log(der + epsilon)
        return action, log_prob, z, mean, log_std

    def get_action(self, state, epsilon=1e-6):
        state = torch.FloatTensor(state).to(next(self.parameters()).device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(torch.zeros(mean.shape),
                        torch.ones(mean.shape))
        z = normal.sample().to(mean.device)
        action = F.tanh(mean + std * z)**2
        der = 2 * F.tanh(mean + std * z) * (1 - action) * ((mean + std * z) > 0).float()
        log_prob = Normal(mean, std).log_prob(mean + std * z) - torch.log(der + epsilon)
        action = action.cpu()  # .detach().cpu().numpy()
        return action, mean, log_std, log_prob


class ModelDynamicsNetwork(nn.Module):
    def __init__(self, *args, init_w=3e-3, **kwargs):
        super(ModelDynamicsNetwork, self).__init__()
        self.state_dim = kwargs['state_dim']
        self.action_dim = kwargs['action_dim']
        self.hidden_dim = kwargs['hidden_dim']
        self.linears = nn.ModuleList([nn.Linear(self.state_dim + self.action_dim, self.hidden_dim[0])])
        self.linears.extend([nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]) for i in range(1, len(self.hidden_dim))])
        self.linears.extend([nn.Linear(self.hidden_dim[-1], self.state_dim)])

        self.apply(init_weights_xavier)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        for l in self.linears[:-1]:
            x = l(x)
            x = F.tanh(x)
        x = self.linears[-1](x)
        return x
