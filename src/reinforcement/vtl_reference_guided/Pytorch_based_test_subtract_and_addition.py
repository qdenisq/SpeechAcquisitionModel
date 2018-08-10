from random import seed
from random import random as rnd
from math import sqrt
from sklearn.metrics import mean_squared_error
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

ds_norm_largest = 1000.
ds_largest = 1000.
largest = 1000.


def generate_model_dynamics_training_data(n_examples):
    s0 = np.array([[(rnd() - 0.5) * 1. * largest for n in range(n_dim)] for _ in range(n_examples)])
    ds = np.array([[(rnd() - 0.5) * 1. * ds_largest for n in range(n_dim)] for _ in range(n_examples)])
    t = s0 + ds
    a = np.array([[(rnd() - 0.5) * 1. * ds_largest for n in range(n_dim)] for _ in range(n_examples)])

    # real model dynamics
    s1 = s0 + a

    # normalise staff
    s0 = normalize(s0, largest)
    t = normalize(t, largest)
    a = normalize(a, ds_norm_largest)
    s1 = normalize(s1, largest)

    X_md = np.concatenate((s0, a), axis=1)
    Y_md = s1

    return X_md, Y_md


def generate_policy_training_data(n_examples):
    s0 = np.array([[(rnd() - 0.5) * 1. * largest for n in range(n_dim)] for _ in range(n_examples)])
    ds = np.array([[(rnd() - 0.5) * 1. * ds_largest for n in range(n_dim)] for _ in range(n_examples)])
    t = s0 + ds
    # normalise staff
    s0 = normalize(s0, largest)
    t = normalize(t, largest)

    X = np.concatenate((s0, t), axis=1)

    a = ds
    a = normalize(a, ds_norm_largest)

    return X, s0, t, a


# invert normalization
def denormalize(value, largest):
    return value * float(largest)


# apply normalization
def normalize(value, largest):
    return value / largest

# define data parameters
seed(1)
n_examples = 10000
n_dim = 20
# define training options
n_batch = 200
n_epoch = 50


################################################################
# Model Dynamics
################################################################


class ModelDynamicsNet(nn.Module):
    def __init__(self, n_dim):
        super(ModelDynamicsNet, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(n_dim * 2, n_dim * 4)
        torch.nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(n_dim * 4, n_dim * 2)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(n_dim * 2,  n_dim)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

md_net = ModelDynamicsNet(n_dim)
md_criterion = nn.MSELoss()
md_optimizer = torch.optim.Adadelta(md_net.parameters())

################################################################
# Policy
################################################################



class PolicyNet(nn.Module):
    def __init__(self, n_dim):
        super(PolicyNet, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(n_dim * 2, n_dim * 4)
        torch.nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(n_dim * 4, n_dim * 2)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(n_dim * 2,  n_dim)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

policy_net = PolicyNet(n_dim)
policy_optimizer = torch.optim.Adadelta(policy_net.parameters())


########################################################
# Train Model Dynamics
########################################################

# train loop
for _ in range(n_epoch):
    # generate new dataset for each epoch
    X, Y = generate_model_dynamics_training_data(n_examples)

    for i in range(n_examples // n_batch):
        # get next batch
        x_batch = X[i * n_batch : (i + 1) * n_batch]
        y_batch = Y[i * n_batch: (i + 1) * n_batch]

        md_optimizer.zero_grad()  # Intialize the hidden weight to all zeros
        outputs = md_net(torch.from_numpy(x_batch).float())
        loss = md_criterion(outputs,
                            torch.from_numpy(y_batch).float())
        loss.backward()  # Backward pass: compute the weight
        # train step
        md_optimizer.step()


    # calculate loss on the whole dataset
    y_pred = md_net(torch.from_numpy(X).float()).detach().numpy()
    total_loss = np.mean(np.sum(np.square(Y - y_pred), axis=1), axis=0)
    y_pred_denormed = denormalize(y_pred, largest)
    y_denormed = denormalize(Y, largest)
    total_loss_denormed = np.mean(np.sum(np.square(y_denormed - y_pred_denormed), axis=1), axis=0)
    print('|epoch : {}| model dynamics loss: {:.10f}| unnormalized loss: {:.7f}'
          .format(_, total_loss, total_loss_denormed))

# Final evaluation on some new patterns
X, Y = generate_model_dynamics_training_data(n_examples)
y_pred = md_net(torch.from_numpy(X).float()).detach().numpy()
# calculate error
expected = denormalize(Y, largest)
predicted = denormalize(y_pred, largest)
rmse = sqrt(mean_squared_error(expected, predicted))
print('RMSE: %f' % rmse)
# show some examples
for i in range(5):
    error = expected[i] - predicted[i]
    print('Expected={}, Predicted={} (err={})'.format(expected[i], predicted[i], error))


########################################################
# Train Policy
########################################################

# train loop
for _ in range(n_epoch):
    # generate new dataset for each epoch
    X, s0, t, Y = generate_policy_training_data(n_examples)

    for i in range(n_examples // n_batch):
        # get next batch
        x_batch = X[i * n_batch: (i + 1) * n_batch]
        y_batch = Y[i * n_batch: (i + 1) * n_batch]
        s0_batch = s0[i * n_batch: (i + 1) * n_batch]
        t_batch = t[i * n_batch: (i + 1) * n_batch]

        policy_optimizer.zero_grad()
        # train step
        y_pred = policy_net(torch.from_numpy(x_batch).float())
        md_x_batch = torch.cat((torch.from_numpy(s0_batch).float(), y_pred), 1)
        md_pred = md_net(md_x_batch)

        loss = md_criterion(md_pred, torch.from_numpy(t_batch).float())
        action_grads = torch.autograd.grad(loss, y_pred)

        torch.autograd.backward(y_pred, action_grads)
        policy_optimizer.step()


    # calculate loss on the whole dataset
    y_pred = policy_net(torch.from_numpy(X).float()).detach().numpy()
    total_loss = np.mean(np.sum(np.square(Y - y_pred), axis=1), axis=0)
    y_pred_denormed = denormalize(y_pred, ds_norm_largest)
    y_denormed = denormalize(Y, ds_norm_largest)
    total_loss_denormed = np.mean(np.sum(np.square(y_denormed - y_pred_denormed), axis=1), axis=0)
    print('|epoch : {}|policy loss: {:.10f}| unnormalized loss: {:.7f}'.format(_, total_loss, total_loss_denormed))

# Final evaluation on some new patterns
X, s0, t, Y = generate_policy_training_data(n_examples)
y_pred = policy_net(torch.from_numpy(X).float()).detach().numpy()
# calculate error
expected = denormalize(Y, ds_norm_largest)
predicted = denormalize(y_pred, ds_norm_largest)
rmse = sqrt(mean_squared_error(expected, predicted))
print('RMSE: %f' % rmse)
# show some examples
for i in range(5):
    error = expected[i] - predicted[i]
    print('Expected={}, Predicted={} (err={})'.format(expected[i], predicted[i], error))