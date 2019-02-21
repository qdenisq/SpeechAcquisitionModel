from torch.nn.init import xavier_uniform_
from torch.nn import Module, ModuleList, Linear, ReLU, Sigmoid, Tanh, BatchNorm1d, LSTM
from torch.distributions import Normal
import torch


def init_weights(m):
    if type(m) == Linear:
        xavier_uniform_(m.weight, gain=1)


#################################################################################
# POLICY
#################################################################################

class SimpleDeterministicPolicy(Module):
    def __init__(self, **kwargs):
        super(SimpleDeterministicPolicy, self).__init__()
        hidden_size = kwargs['linear_layers_size']
        # actor
        self.bn = BatchNorm1d(kwargs['input_dim'])
        self.linears = ModuleList([Linear(kwargs['input_dim'], hidden_size[0])])
        self.linears.extend([Linear(hidden_size[i - 1], hidden_size[i]) for i in range(1, len(hidden_size))])
        self.out = Linear(hidden_size[-1], kwargs['action_dim'])

        self.relu = ReLU()
        self.tanh = Tanh()

        self.apply(init_weights) # xavier uniform init

    def forward(self, input, action=None):
        x = input
        x = self.bn(x)
        for l in self.linears:
            x = l(x)
            x = self.relu(x)
        action = self.tanh(self.out(x))
        return action


class SimpleStochasticPolicy(Module):
    def __init__(self, **kwargs):
        super(SimpleStochasticPolicy, self).__init__()
        hidden_size = kwargs['linear_layers_size']
        # actor
        self.bn = BatchNorm1d(kwargs['input_dim'])
        self.linears = ModuleList([Linear(kwargs['input_dim'], hidden_size[0])])
        self.linears.extend([Linear(hidden_size[i - 1], hidden_size[i]) for i in range(1, len(hidden_size))])
        self.mu = Linear(hidden_size[-1], kwargs['action_dim'])
        self.log_var = Linear(hidden_size[-1], kwargs['action_dim'])
        # self.log_var = torch.nn.Parameter(torch.zeros(kwargs['action_dim']))

        self.relu = ReLU()
        self.tanh = Tanh()

        self.apply(init_weights) # xavier uniform init

    def forward(self, input, action=None):
        x = input
        x = self.bn(x)
        for l in self.linears:
            x = l(x)
            x = self.relu(x)
        mu = self.tanh(self.mu(x))
        log_var = -self.relu(self.log_var(x))
        sigmas = log_var.exp().sqrt()
        dists = Normal(mu, sigmas + 1.0e-4)
        if action is None:
            action = dists.rsample()
        log_prob = dists.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob, dists.entropy()


class SimpleStochasticActorCritic(Module):
    def __init__(self, **kwargs):
        super(SimpleStochasticActorCritic, self).__init__()
        hidden_size = kwargs['hidden_size']
        # actor
        self.actor_bn = BatchNorm1d(kwargs['state_dim'])
        self.actor_linears = ModuleList([Linear(kwargs['state_dim'], hidden_size[0])])
        self.actor_linears.extend([Linear(hidden_size[i - 1], hidden_size[i]) for i in range(1, len(hidden_size))])
        self.mu = Linear(hidden_size[-1], kwargs['action_dim'])
        self.log_var = Linear(hidden_size[-1], kwargs['action_dim'])
        self.log_var_const = torch.zeros(kwargs['action_dim'])

        # critic
        self.critic_bn = BatchNorm1d(kwargs['state_dim'])
        self.critic_linears = ModuleList([Linear(kwargs['state_dim'], hidden_size[0])])
        self.critic_linears.extend([Linear(hidden_size[i - 1], hidden_size[i]) for i in range(1, len(hidden_size))])
        self.v = Linear(hidden_size[-1], 1)

        self.relu = ReLU()
        self.tanh = Tanh()

        self.apply(init_weights) # xavier uniform init

    def forward(self, state, action=None):
        x = state
        x = self.actor_bn(x)
        for l in self.actor_linears:
            x = l(x)
            x = self.relu(x)
        mu = self.tanh(self.mu(x))

        # log_var = - 3. - self.relu(self.log_var(x))
        log_var = - 5. - self.relu(self.log_var_const)
        sigmas = log_var.exp().sqrt() + 1.0e-5
        dists = Normal(mu, sigmas)
        if action is None:
            action = dists.sample()

        log_prob = dists.log_prob(action).sum(dim=-1, keepdim=True)

        x = state
        x = self.critic_bn(x)
        for l in self.critic_linears:
            x = l(x)
            x = self.relu(x)
        v = self.v(x)
        return action, log_prob, dists.entropy(), v

    def get_actor_parameters(self):
        return [*self.actor_bn.parameters(), *self.actor_linears.parameters(), *self.mu.parameters(), *self.log_var.parameters()]

    def get_critic_parameters(self):
        return [*self.critic_bn.parameters(), *self.critic_linears.parameters(), *self.v.parameters()]


#################################################################################
# MODEL DYNAMICS
#################################################################################

class SimpleStochasticModelDynamics(Module):
    def __init__(self, **kwargs):
        super(SimpleStochasticModelDynamics, self).__init__()
        self.__acoustic_state_dim = kwargs['goal_dim']
        self.__action_dim = kwargs['action_dim']
        self.__state_dim = kwargs['state_dim']
        self.__linears_size = kwargs['linear_layers_size']

        input_size = self.__acoustic_state_dim + self.__state_dim + self.__action_dim
        self.__bn1 = torch.nn.BatchNorm1d(input_size)

        self.linears = ModuleList([Linear(input_size, self.__linears_size[0])])
        self.linears.extend(
            [Linear(self.__linears_size[i - 1], self.__linears_size[i]) for i in range(1, len(self.__linears_size))])

        self.goal_mu = Linear(self.__linears_size[-1], kwargs['goal_dim'])
        self.goal_log_var = Linear(self.__linears_size[-1], kwargs['goal_dim'])

        self.state_mu = Linear(self.__linears_size[-1], kwargs['state_dim'])
        self.state_log_var = Linear(self.__linears_size[-1], kwargs['state_dim'])
        self.state_log_var_const = torch.zeros(kwargs['state_dim'])

        self.relu = ReLU()
        self.tanh = Tanh()

        self.apply(init_weights)  # xavier uniform init

    def forward(self, states, actions):
        x = torch.cat((states, actions), -1)
        original_dim = x.shape
        x = self.__bn1(x.view(-1, original_dim[-1]))
        x = x.view(original_dim)

        for linear in self.linears:
            x = self.relu(linear(x))

        # predict state
        state_mu = self.tanh(self.state_mu(x))
        # state_log_var = -self.relu(self.state_log_var(x))
        state_log_var = -5. -self.relu(self.state_log_var_const)
        state_sigmas = state_log_var.exp().sqrt()
        state_dists = Normal(state_mu, state_sigmas + 1.0e-4)
        states = state_dists.rsample()
        state_log_prob = state_dists.log_prob(states).sum(dim=-1, keepdim=True)

        # predict goal
        goal_mu = self.tanh(self.goal_mu(x))
        goal_log_var = -self.relu(self.goal_log_var(x))
        goal_sigmas = goal_log_var.exp().sqrt()
        goal_dists = Normal(goal_mu, goal_sigmas + 1.0e-4)
        goals = goal_dists.rsample()
        goal_log_prob = goal_dists.log_prob(goals).sum(dim=-1, keepdim=True)

        return states, goals, state_log_prob, goal_log_prob, state_dists, goal_dists


class StochasticLstmModelDynamics(Module):
    def __init__(self, **kwargs):
        super(StochasticLstmModelDynamics, self).__init__()
        self.__acoustic_state_dim = kwargs['goal_dim']
        self.__action_dim = kwargs['action_dim']
        self.__state_dim = kwargs['state_dim']
        self.__lstm_sizes = kwargs['lstm_layers_size']
        self.__linears_size = kwargs['linear_layers_size']

        input_size = self.__acoustic_state_dim + self.__state_dim + self.__action_dim
        self.__bn1 = torch.nn.BatchNorm1d(input_size)

        self.lstms = ModuleList([LSTM(input_size, self.__lstm_sizes[0], batch_first=True)])
        self.lstms.extend([LSTM(self.__lstm_sizes[i - 1], self.__lstm_sizes[i], batch_first=True) for i in range(1, len(self.__lstm_sizes))])

        self.linears = ModuleList([Linear(self.__lstm_sizes[-1], self.__linears_size[0])])
        self.linears.extend([Linear(self.__linears_size[i - 1], self.__linears_size[i]) for i in range(1, len(self.__linears_size))])

        self.goal_mu = Linear(self.__linears_size[-1], kwargs['goal_dim'])
        self.goal_log_var = Linear(self.__linears_size[-1], kwargs['goal_dim'])

        self.state_mu = Linear(self.__linears_size[-1], kwargs['state_dim'])
        self.state_log_var = Linear(self.__linears_size[-1], kwargs['state_dim'])

        self.relu = ReLU()
        self.tanh = Tanh()

        self.apply(init_weights)  # xavier uniform init

    def forward(self, states, goal_states, actions, hidden=None):
        x = torch.cat((goal_states, states, actions), -1)
        original_dim = x.shape
        x = self.__bn1(x.view(-1, original_dim[-1]))
        x = x.view(original_dim)

        for lstm in self.lstms:
            x, _ = lstm(x)

        for linear in self.linears:
            x = self.relu(linear(x))

        # predict state
        state_mu = self.tanh(self.state_mu(x))
        state_log_var = -self.relu(self.state_log_var(x))
        state_sigmas = state_log_var.exp().sqrt()
        state_dists = Normal(state_mu, state_sigmas + 1.0e-4)
        states = state_dists.rsample()
        state_log_prob = state_dists.log_prob(states).sum(dim=-1, keepdim=True)

        # predict goal
        goal_mu = self.tanh(self.goal_mu(x))
        goal_log_var = -self.relu(self.goal_log_var(x))
        goal_sigmas = goal_log_var.exp().sqrt()
        goal_dists = Normal(goal_mu, goal_sigmas + 1.0e-4)
        goals = goal_dists.rsample()
        goal_log_prob = goal_dists.log_prob(goals).sum(dim=-1, keepdim=True)

        return states, goals, state_log_prob, goal_log_prob, state_dists, goal_dists


class DeterministicLstmModelDynamics(Module):
    def __init__(self, **kwargs):
        super(DeterministicLstmModelDynamics, self).__init__()
        self.__acoustic_state_dim = kwargs['goal_dim']
        self.__action_dim = kwargs['action_dim']
        self.__state_dim = kwargs['state_dim']
        self.__lstm_sizes = kwargs['lstm_layers_size']
        self.__linears_size = kwargs['linear_layers_size']

        input_size = self.__acoustic_state_dim + self.__state_dim + self.__action_dim
        self.__bn1 = torch.nn.BatchNorm1d(input_size)

        self.lstms = ModuleList([LSTM(input_size, self.__lstm_sizes[0], batch_first=True)])
        self.lstms.extend([LSTM(self.__lstm_sizes[i - 1], self.__lstm_sizes[i], batch_first=True) for i in range(1, len(self.__lstm_sizes))])
        self.hiddens = [None] * len(self.__lstm_sizes)

        self.linears = ModuleList([Linear(self.__lstm_sizes[-1], self.__linears_size[0])])
        self.linears.extend([Linear(self.__linears_size[i - 1], self.__linears_size[i]) for i in range(1, len(self.__linears_size))])

        self.goal = Linear(self.__linears_size[-1], kwargs['goal_dim'])

        self.state = Linear(self.__linears_size[-1], kwargs['state_dim'])

        self.relu = ReLU()
        self.tanh = Tanh()

        self.apply(init_weights)  # xavier uniform init

    def forward(self, states, actions, hidden=None):
        x = torch.cat((states, actions), -1)
        original_dim = x.shape
        x = self.__bn1(x.view(-1, original_dim[-1]))
        x = x.view(original_dim)

        for i, lstm in enumerate(self.lstms):
            x, self.hiddens[i] = lstm(x, self.hiddens[i])

        for linear in self.linears:
            x = self.relu(linear(x))

        # predict state
        states_out = self.state(x) + states[:, :self.__state_dim]

        # predict goal
        goals_out = self.goal(x) + states[:, self.__state_dim:]

        return states_out, goals_out

    def reset_hidden_state(self):
        self.hiddens = [None] * len(self.__lstm_sizes)


class SimpleDeterministicModelDynamics(Module):
    def __init__(self, **kwargs):
        super(SimpleDeterministicModelDynamics, self).__init__()
        self.__acoustic_state_dim = kwargs['goal_dim']
        self.__action_dim = kwargs['action_dim']
        self.__state_dim = kwargs['state_dim']
        self.__linears_size = kwargs['linear_layers_size']

        input_size = self.__acoustic_state_dim + self.__state_dim + self.__action_dim
        self.__bn1 = torch.nn.BatchNorm1d(input_size)

        self.linears = ModuleList([Linear(input_size, self.__linears_size[0])])
        self.linears.extend(
            [Linear(self.__linears_size[i - 1], self.__linears_size[i]) for i in range(1, len(self.__linears_size))])

        self.goal = Linear(self.__linears_size[-1], kwargs['goal_dim'])
        self.state = Linear(self.__linears_size[-1], kwargs['state_dim'])

        self.relu = ReLU()
        self.tanh = Tanh()

        self.apply(init_weights)  # xavier uniform init

    def forward(self, states, actions):
        x = torch.cat((states, actions), -1)
        original_dim = x.shape
        x = self.__bn1(x.view(-1, original_dim[-1]))
        x = x.view(original_dim)

        for linear in self.linears:
            x = self.relu(linear(x))

        # predict state
        states = self.tanh(self.state(x))
        # predict goal
        goals = self.tanh(self.goal(x))

        return states, goals


class SimpleDeterministicModelDynamicsDeltaPredict(Module):
    def __init__(self, **kwargs):
        super(SimpleDeterministicModelDynamicsDeltaPredict, self).__init__()
        self.__acoustic_state_dim = kwargs['goal_dim']
        self.__action_dim = kwargs['action_dim']
        self.__state_dim = kwargs['state_dim']
        self.__linears_size = kwargs['linear_layers_size']

        input_size = self.__acoustic_state_dim + self.__state_dim + self.__action_dim
        self.__bn1 = torch.nn.BatchNorm1d(input_size)

        self.linears = ModuleList([Linear(input_size, self.__linears_size[0])])
        self.linears.extend(
            [Linear(self.__linears_size[i - 1], self.__linears_size[i]) for i in range(1, len(self.__linears_size))])

        self.goal = Linear(self.__linears_size[-1], kwargs['goal_dim'])
        self.state = Linear(self.__linears_size[-1], kwargs['state_dim'])

        self.relu = ReLU()
        self.tanh = Tanh()

        self.apply(init_weights)  # xavier uniform init

    def forward(self, states, actions):
        x = torch.cat((states, actions), -1)
        original_dim = x.shape
        x = self.__bn1(x.view(-1, original_dim[-1]))
        x = x.view(original_dim)

        for linear in self.linears:
            x = self.relu(linear(x))

        # predict state
        states_delta = self.state(x)
        out_states = states[:, :self.__state_dim] + states_delta

        # predict goal
        goals_delta = self.goal(x)
        out_goals = states[:, self.__state_dim:] + goals_delta
        return out_states, out_goals

