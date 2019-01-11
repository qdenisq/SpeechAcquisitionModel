import json
import numpy as np
import pandas as pd
import torch

from torch import nn
from pprint import pprint

from src.speech_classification.audio_processing import AudioPreprocessor
from src.speech_classification.pytorch_conv_lstm import LstmNet


class LstmModelDynamics(nn.Module):
    def __init__(self, **kwargs):
        super(LstmModelDynamics, self).__init__()
        self.__acoustic_state_dim = kwargs['acoustic_state_dim']
        self.__state_dim = kwargs['action_dim']
        self.__action_dim = kwargs['state_dim']
        self.__hidden_cells = kwargs['hidden_reccurent_cells_count']

        self.__lstm = nn.LSTM(self.__acoustic_state_dim + self.__state_dim + self.__action_dim ,
                              self.__hidden_cells, batch_first=True)
        self.__output_layer = nn.Linear(self.__hidden_cells, self.__acoustic_state_dim)

    def forward(self, acoustic_states, states, actions, hidden=None):
        x = torch.cat((acoustic_states, states, actions), -1)
        lstm_output, hidden = self.__lstm(x, hidden)
        x = self.__output_layer(lstm_output)
        return lstm_output, x


def train(*args, **kwargs):
    print(kwargs)

    # 1. Init audio preprocessing
    preproc = AudioPreprocessor(**kwargs['preprocessing_params'])
    sr = kwargs['preprocessing_params']['sample_rate']

    # 2. Load preprocessing net
    preproc_net = torch.load(kwargs['preproc_net_fname'])

    # 3. Init model dynamics net
    md_net = LstmModelDynamics(**kwargs['model_dynamics_params'])
    optim = torch.optim.Adam(md_net.parameters(), lr=kwargs['train']['learning_rate'], eps=kwargs['train']['learning_rate_eps'])

    # 4. Load training set
    data_fname = kwargs['data_fname']
    df = pd.read_pickle(data_fname)

    # 5. Train loop
    params = kwargs['train']
    for i in range(params['num_steps']):
        sample = df.sample(n=kwargs['train']['minibatch_size'])
        states = np.stack(sample.loc[:, 'states'].values)
        actions = np.stack(sample.loc[:, 'actions'].values)
        audio = np.stack(sample.loc[:, 'audio'].values)

        preproc_audio = np.array([preproc(audio[j], sr) for j in range(audio.shape[0])])
        _, _, acoustic_states = preproc_net(torch.from_numpy(preproc_audio).float(),
                                 seq_lens=np.array([preproc_audio.shape[-2]]))



        seq_len = actions.shape[1]

        # forward prop
        lstm_outs, predicted_acoustic_states = md_net(acoustic_states,
                                           torch.from_numpy(states[:, :seq_len, :]).float(),
                                           torch.from_numpy(actions).float())

        # compute error
        loss = nn.MSELoss()(acoustic_states[:, 1:, :], predicted_acoustic_states[:, :-1, :])

        # backprop
        optim.zero_grad()
        loss.backward()
        optim.step()

        print("\rstep :{} | loss: {:.4f}".format(i, loss.detach().item()), end="")


if __name__ == '__main__':
    with open('rnn_md_config.json') as data_file:
        data = json.load(data_file)

    pprint(data)
    kwargs = data

    train(**kwargs)