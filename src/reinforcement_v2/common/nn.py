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
        action = F.tanh(mean + std * z)
        der = 1 - action ** 2
        log_prob = Normal(mean, std).log_prob(mean + std * z) - torch.log(der + epsilon)
        return action, log_prob, z, mean, log_std

    def get_action(self, state, epsilon=1e-6):
        state = torch.FloatTensor(state).to(next(self.parameters()).device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(torch.zeros(mean.shape),
                        torch.ones(mean.shape))
        z = normal.sample().to(mean.device)
        action = F.tanh(mean + std * z)
        der = 1 - action ** 2
        log_prob = Normal(mean, std).log_prob(mean + std * z) - torch.log(der + epsilon)
        action = action.cpu()  # .detach().cpu().numpy()
        return action, mean, log_std, log_prob



class DeterministicPolicyNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DeterministicPolicyNetwork, self).__init__()

        self.state_dim = kwargs['state_dim']
        self.action_dim = kwargs['action_dim']
        self.hidden_dim = kwargs['hidden_dim']

        self.linears = nn.ModuleList([nn.Linear(self.state_dim, self.hidden_dim[0])])
        self.linears.extend(
            [nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]) for i in range(1, len(self.hidden_dim))])

        self.out = nn.Linear(self.hidden_dim[-1], self.action_dim)

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

        out = torch.tanh(self.out(x))

        return out

    def get_action(self, state, epsilon=1e-6):
        state = torch.FloatTensor(state).to(next(self.parameters()).device)
        action = self.forward(state)
        return action


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


class ModelDynamics(nn.Module):
    def __init__(self, **kwargs):
        super(ModelDynamics, self).__init__()
        self.__action_dim = kwargs['action_dim']
        self.__state_dim = kwargs['agent_state_dim']
        self.__acoustic_dim = kwargs['acoustic_dim']
        self.__linears_size = kwargs['hidden_dim']

        # input_size = self.__acoustic_state_dim + self.__state_dim + self.__action_dim
        input_size = self.__state_dim + self.__action_dim
        self.__bn1 = torch.nn.BatchNorm1d(input_size)

        self.drop = torch.nn.modules.Dropout(p=0.1)

        # self.artic_state_0 = Linear(self.__state_dim + self.__action_dim - self.__acoustic_dim, 64)
        # self.artic_state_1 = Linear(64, self.__state_dim - self.__acoustic_dim)
        self.artic_state_0 = nn.Linear(self.__state_dim + self.__action_dim - self.__acoustic_dim, self.__state_dim - self.__acoustic_dim)
        # self.artic_state_1 = Linear(64, )

        self.linears = nn.ModuleList([nn.Linear(input_size, self.__linears_size[0])])
        self.linears.extend(
            [nn.Linear(self.__linears_size[i - 1], self.__linears_size[i]) for i in range(1, len(self.__linears_size))])

        self.acoustic_state = nn.Linear(self.__linears_size[-1], self.__acoustic_dim)
        self.state = nn.Linear(self.__state_dim, self.__state_dim)


        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.apply(init_weights_xavier)  # xavier uniform init

    def forward(self, states, actions):
        x = torch.cat((states[:, :self.__state_dim], actions), -1)
        original_dim = x.shape
        x = self.__bn1(x.view(-1, original_dim[-1]))
        x = x.view(original_dim)
        x = self.drop(x)

        # artic
        artic_x = x[:, :self.__state_dim - self.__acoustic_dim]
        actions_x = x[:, -self.__action_dim:]
        # artic_state_delta = self.artic_state_1(self.relu(self.artic_state_0(torch.cat((artic_x, actions_x), -1))))
        artic_state_delta = self.artic_state_0(torch.cat((artic_x, actions_x), -1))

        # acoustic
        for linear in self.linears:
            x = self.relu(linear(x))

        # predict state
        acoustic_state_delta = self.acoustic_state(x)

        states_delta = torch.cat((artic_state_delta, acoustic_state_delta), -1)
        # states_delta = self.tanh(torch.cat((artic_state_delta, acoustic_state_delta), -1))
        out_states = states[:, :self.__state_dim] + states_delta

        return out_states