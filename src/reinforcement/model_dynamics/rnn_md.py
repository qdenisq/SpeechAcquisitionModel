import json
import numpy as np
import pandas as pd
import torch

from torch.nn import Linear, LSTM, Tanh, ReLU, Module, MSELoss
from torchvision import transforms
from pprint import pprint

from src.speech_classification.audio_processing import AudioPreprocessor
from src.speech_classification.pytorch_conv_lstm import LstmNet


class LstmModelDynamics(Module):
    def __init__(self, **kwargs):
        super(LstmModelDynamics, self).__init__()
        self.__acoustic_state_dim = kwargs['acoustic_state_dim']
        self.__state_dim = kwargs['action_dim']
        self.__action_dim = kwargs['state_dim']
        self.__hidden_cells = kwargs['hidden_reccurent_cells_count']

        self.__bn1 = torch.nn.BatchNorm1d(self.__acoustic_state_dim + self.__state_dim + self.__action_dim)

        self.__lstm = LSTM(self.__acoustic_state_dim + self.__state_dim + self.__action_dim,
                              self.__hidden_cells, batch_first=True)
        self.__linear_layer = Linear(self.__hidden_cells, 256)
        self.__output_layer = Linear(256, self.__acoustic_state_dim)


    def forward(self, acoustic_states, states, actions, hidden=None):

        x = torch.cat((acoustic_states, states, actions), -1)
        original_dim = x.shape
        x = self.__bn1(x.view(-1, original_dim[-1]))
        x = x.view(original_dim)
        lstm_output, hidden = self.__lstm(x, hidden)
        x = lstm_output
        x = Tanh()(self.__linear_layer(x))
        x = self.__output_layer(x)
        return lstm_output, x


def train(*args, **kwargs):
    print(kwargs)

    device = kwargs['train']['device']

    # 1. Init audio preprocessing
    preproc = AudioPreprocessor(**kwargs['preprocessing_params'])
    sr = kwargs['preprocessing_params']['sample_rate']

    # 2. Load preprocessing net
    preproc_net = torch.load(kwargs['preproc_net_fname']).to(device)

    # 3. Init model dynamics net
    md_net = LstmModelDynamics(**kwargs['model_dynamics_params']).to(device)
    optim = torch.optim.Adam(md_net.parameters(), lr=kwargs['train']['learning_rate'], eps=kwargs['train']['learning_rate_eps'])

    # 4. Load training set
    data_fname = kwargs['data_fname']
    df = pd.read_pickle(data_fname)

    # 5. Train loop
    params = kwargs['train']
    md_net.train()
    for i in range(params['num_steps']):
        sample = df.sample(n=kwargs['train']['minibatch_size'])
        states = np.stack(sample.loc[:, 'states'].values)
        actions = np.stack(sample.loc[:, 'actions'].values)
        audio = np.stack(sample.loc[:, 'audio'].values)

        preproc_audio = np.array([preproc(audio[j], sr) for j in range(audio.shape[0])])

        acoustic_states = torch.from_numpy(preproc_audio).float().to(device)
        # acoustic_states = acoustic_states.view(-1, kwargs['model_dynamics_params']["acoustic_state_dim"])
        # mean_norm = acoustic_states.mean(dim=0)
        # mean_std = acoustic_states.std(dim=0)
        # acoustic_states = (acoustic_states - mean_norm.view(1, -1)) / mean_std.view(1, -1)
        # acoustic_states = acoustic_states.view(kwargs['train']['minibatch_size'], -1, kwargs['model_dynamics_params']["acoustic_state_dim"])
        _, _, acoustic_states = preproc_net(torch.from_numpy(preproc_audio).float().to(device),
                                 seq_lens=np.array([preproc_audio.shape[-2]]))



        seq_len = actions.shape[1]
        acoustic_state_dim = kwargs['model_dynamics_params']["acoustic_state_dim"]

        # forward prop
        lstm_outs, predicted_acoustic_states = md_net(acoustic_states,
                                           torch.from_numpy(states[:, :seq_len, :]).float().to(device),
                                           torch.from_numpy(actions).float().to(device))

        # compute error
        loss = MSELoss(reduction='sum')(predicted_acoustic_states[:, :-1, :].contiguous().view(-1, acoustic_state_dim),
                                        acoustic_states[:, 1:, :].contiguous().view(-1, acoustic_state_dim)) / (seq_len * kwargs['train']['minibatch_size'])

        # backprop
        optim.zero_grad()
        loss.backward()
        optim.step()

        print("\rstep: {} | loss: {:.4f}".format(i, loss.detach().cpu().item()), end="")


if __name__ == '__main__':
    with open('rnn_md_config.json') as data_file:
        data = json.load(data_file)

    pprint(data)
    kwargs = data

    train(**kwargs)