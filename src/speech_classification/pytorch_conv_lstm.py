import src.speech_classification.utils as utils
from src.speech_classification.audio_processing import AudioPreprocessorFbank, SpeechCommandsDataCollector
import torch.nn.functional as F
import torch.nn as nn
import torch

import os
import datetime


from python_speech_features import mfcc
import scipy.io.wavfile as wav

import numpy as np

class ConvLstmNet(nn.Module):
    def __init__(self, model_settings):
        super(ConvLstmNet, self).__init__()
        self.__n_window_height = model_settings['dct_coefficient_count']
        self.__n_window_width = model_settings['window_size']
        self.__n_window_sequence_length = model_settings['sequence_length']
        self.__n_classes = model_settings['label_count']
        self.__n_hidden_cells = model_settings['hidden_reccurent_cells_count']

        self.__n_first_conv_filter_height = 21
        self.__n_first_conv_filter_width = 9
        self.__n_first_conv_depth = 1
        self.__n_first_conv_channels = 64
        k_size = (self.__n_first_conv_depth,
                  self.__n_first_conv_filter_height,
                  self.__n_first_conv_filter_width
                  )
        pad = (e // 2 for e in k_size)
        padding = torch.nn.ReplicationPad3d(pad)
        self.__first_conv = torch.nn.Conv3d(in_channels=1,
                                            out_channels=self.__n_first_conv_channels,
                                            kernel_size=k_size,
                                            padding=padding,
                                            stride=1)
        self.__dropout_prob = 0.5
        self.__first_dropout = torch.nn.Dropout(p=self.__dropout_prob)
        self.__first_max_pool = torch.nn.MaxPool3d(kernel_size=(1, 2, 2, 1), stride=(1, 2, 2, 1))

        self.__n_second_conv_filter_height = 10
        self.__n_second_conv_filter_width = 4
        self.__n_second_conv_filter_depth = 1
        self.__n_second_conv_channels = 64
        k_size = (self.__n_second_conv_depth,
                  self.__n_second_conv_filter_height,
                  self.__n_second_conv_filter_width
                  )
        pad = (e // 2 for e in k_size)
        padding = torch.nn.ReplicationPad3d(pad)
        self.__second_conv = torch.nn.Conv3d(in_channels=self.__n_first_conv_channels,
                                             out_channels=self.__n_second_conv_channels,
                                             kernel_size=k_size,
                                             padding=padding,
                                             stride=1)



        self.__lstm = nn.LSTMCell(n_dim, n_hidden)
        self.out = nn.Linear(n_hidden, n_out)

    def forward(self, x, hidden=None):
        # if hidden is None:
        #     hidden = torch.zeros(x.size(0), self.n_hidden)

        x = self.__first_conv(x)
        x = torch.nn.ReLU(x)
        x = self.__first_dropout(x)
        x = self.__first_max_pool(x)

        x = self.__second_conv(x)
        x = torch.nn.ReLU(x)

        x, hidden = self.l1(x, hidden)

        x = self.out(x)
        return x, hidden


class LstmNet(nn.Module):
    def __init__(self, model_settings):
        super(LstmNet, self).__init__()
        self.__n_window_height = model_settings['dct_coefficient_count']
        self.__n_classes = model_settings['label_count']
        self.__n_hidden_cells = model_settings['hidden_reccurent_cells_count']

        self.__bn1 = nn.BatchNorm1d(self.__n_window_height)
        self.__compress_layer = nn.Linear(self.__n_hidden_cells, 10)
        self.__lstm = nn.LSTM(self.__n_window_height, self.__n_hidden_cells, batch_first=True)
        self.__output_layer = nn.Linear(self.__n_hidden_cells, self.__n_classes)

    def forward(self, x, seq_lens, hidden=None):
        # if hidden is None:
        #     hidden = torch.zeros(x.size(0), self.n_hidden)
        orig_shape = x.shape
        x = x.view(-1, self.__n_window_height)
        x = self.__bn1(x)
        x = x.view(orig_shape)
        x, hidden = self.__lstm(x, hidden)
        lstm_out = x
        last_elements = seq_lens - 1
        # adjust x to the actual seq length for every sequence
        x_1 = x[np.arange(0, len(seq_lens)), last_elements, :]
        x = self.__output_layer(x_1)
        return x, hidden, lstm_out


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels, axis=None) / float(labels.size)


if __name__ == '__main__':


    #############################################################
    # Simple Lstm train script
    #############################################################

    # init model settings

    # instantiate preproc cand lstm net

    wanted_words = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin',
                    'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two',
                    'up', 'wow', 'yes', 'zero']
    wanted_words_tanh_transition = ['a_a', 'a_i', 'a_u', 'a_o', 'a_e',
                                    'i_a', 'i_i', 'i_u', 'i_o', 'i_e',
                                    'u_a', 'u_i', 'u_u', 'u_o', 'u_e',
                                    'o_a', 'o_i', 'o_u', 'o_o', 'o_e',
                                    'e_a', 'e_i', 'e_u', 'e_o', 'e_e']
    model_settings = {
        'dct_coefficient_count': 26,
        'label_count': len(wanted_words_tanh_transition) + 2,
        'hidden_reccurent_cells_count': 64,
        'winlen': 0.04,
        'winstep': 0.04
    }

    save_dir = r'C:\Study\SpeechAcquisitionModel\models\speech_classification'
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except:
            pass
    best_acc = 0.0
    lowest_loss = 100.0

    preproc = AudioPreprocessorFbank(model_settings['dct_coefficient_count'], winlen=model_settings['winlen'], winstep=model_settings['winstep'])
    data_iter = SpeechCommandsDataCollector(preproc,
                                            data_dir=r'C:/Study/SpeechAcquisitionModel/data/raw/Simple_transitions/02_25_2019_03_29_PM_08',
                                            wanted_words=wanted_words_tanh_transition,
                                            testing_percentage=10,
                                            validation_percentage=10
                                            )
    net = LstmNet(model_settings)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # configure training procedure
    n_train_steps = 500
    n_mini_batch_size = 256

    for i in range(n_train_steps):
        # collect data
        d = data_iter.get_data(n_mini_batch_size, 0, 'training')
        data = d['x']
        labels = d['y']
        seq_lengths = d['seq_len']

        data = torch.from_numpy(data).float()
        labels = torch.from_numpy(labels).long()
        # seq_lengths = torch.from_numpy(seq_lengths)

        # zero grad
        optimizer.zero_grad()

        pred, hidden, _ = net(data, seq_lengths)
        loss = torch.nn.CrossEntropyLoss()(pred, labels)
        loss.backward()
        optimizer.step()

        acc = accuracy(pred.detach().numpy(), labels.detach().numpy())
        print("|train_step: {}| loss: {:.4f}| accuracy: {:.4f}|".format(i, loss.detach(), acc))
        # validate each 100 train steps
        if i % 100 == 0:
            d = data_iter.get_data(n_mini_batch_size, 0, 'validation')
            data = d['x']
            labels = d['y']
            seq_lengths = d['seq_len']

            data = torch.from_numpy(data).float()
            labels = torch.from_numpy(labels).long()
            # zero grad
            optimizer.zero_grad()

            pred, hidden, _ = net(data, seq_lengths)
            validation_loss = torch.nn.CrossEntropyLoss()(pred.detach(), labels.detach())
            acc = accuracy(pred.detach().numpy(), labels.detach().numpy())
            print("Validation loss: {:.4f}| accuracy: {:.4f}|".format(validation_loss.detach(), acc))

            if acc > best_acc or validation_loss < lowest_loss:
                best_acc = acc
                lowest_loss = validation_loss
                dt = str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p"))
                fname = os.path.join(save_dir, '{}_{}_acc_{:.4f}.pt'.format("simple_lstm", dt, acc))
                torch.save(net, fname)

    # Final test accuracy
    d = data_iter.get_data(n_mini_batch_size, 0, 'testing')
    data = d['x']
    labels = d['y']
    seq_lengths = d['seq_len']

    # load best model
    net = torch.load(fname)

    data = torch.from_numpy(data).float()
    labels = torch.from_numpy(labels).long()
    # zero grad
    optimizer.zero_grad()

    pred, hidden, _ = net(data, seq_lengths)
    test_loss = torch.nn.CrossEntropyLoss()(pred, labels)
    acc = accuracy(pred.detach().numpy(), labels.detach().numpy())
    print("Test loss: {:.4f}| accuracy: {:.4f}|".format(test_loss.detach(), acc))