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
        self.__n_window_height = kwargs['dct_coefficient_count']
        self.__n_hidden_cells = kwargs['hidden_reccurent_cells_count']

        self.__lstm = nn.LSTM(self.__n_window_height, self.__n_hidden_cells, batch_first=True)
        # self.__output_layer = nn.Linear(self.__n_hidden_cells, self.__n_classes)

    def forward(self, x, seq_lens, hidden=None):
        x, hidden = self.__lstm(x, hidden)
        last_elements = seq_lens - 1
        # adjust x to the actual seq length for every sequence
        x = x[np.arange(0, len(seq_lens)), last_elements, :]
        # x = self.__output_layer(x_1)
        return x, hidden


def train(*args, **kwargs):
    print(kwargs)

    # 1. Init audio preprocessing
    preproc = AudioPreprocessor(**kwargs['preprocessing_params'])
    sr = kwargs['preprocessing_params']['sample_rate']

    # 2. Load preprocessing net
    preproc_net = torch.load(kwargs['preproc_net_fname'])

    # 3. Init model dynamics net
    md_net = LstmModelDynamics(**kwargs['model_dynamics_params'])

    # 4. Load training set
    data_fname = kwargs['data_fname']
    df = pd.read_pickle(data_fname)

    # 5. Train loop
    params = kwargs['train']
    for i in range(params['num_steps']):
        sample = df.sample(n=1)
        states = sample.loc[:, 'states'].values[0]
        actions = sample.loc[:, 'actions'].values[0]
        audio = sample.loc[:, 'audio'].values[0]

        preproc_audio = preproc(audio, sr)[np.newaxis, :, :]
        _, _, proc_audio = preproc_net(torch.from_numpy(preproc_audio).float(),
                                 seq_lens=np.array([preproc_audio.shape[-2]]))



        print(proc_audio.shape, states.shape, actions.shape)


if __name__ == '__main__':
    with open('rnn_md_config.json') as data_file:
        data = json.load(data_file)

    pprint(data)
    kwargs = data

    train(**kwargs)