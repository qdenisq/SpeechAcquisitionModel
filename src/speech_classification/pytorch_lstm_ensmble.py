import src.speech_classification.utils as utils
from src.speech_classification.audio_processing import AudioPreprocessorFbank, SpeechCommandsDataCollector
import torch.nn.functional as F
import torch.nn as nn
import torch
import random

import os
import datetime


from python_speech_features import mfcc
import scipy.io.wavfile as wav

import numpy as np

class LstmNet(nn.Module):
    def __init__(self, model_settings):
        super(LstmNet, self).__init__()
        self.__n_window_height = model_settings['dct_coefficient_count']
        self.__n_classes = model_settings['label_count']
        self.__n_hidden_cells = model_settings['hidden_reccurent_cells_count']
        self.__dropout = nn.Dropout(p=0.2)
        self.__bn1 = nn.BatchNorm1d(self.__n_window_height)
        self.__compress_layer = nn.Linear(self.__n_hidden_cells, 10)
        self.__lstm = nn.LSTM(self.__n_window_height, self.__n_hidden_cells, batch_first=True, bidirectional=False)
        self.__output_layer = nn.Linear(self.__n_hidden_cells, 256)
        self.__output_layer_1 = nn.Linear(256, self.__n_classes)

    def forward(self, x, seq_lens, hidden=None):
        # if hidden is None:
        #     hidden = torch.zeros(x.size(0), self.n_hidden)
        orig_shape = x.shape
        x = x.view(-1, self.__n_window_height)
        x = self.__bn1(x)
        x = self.__dropout(x)
        x = x.view(orig_shape)
        x, hidden = self.__lstm(x, hidden)
        lstm_out = x
        last_elements = seq_lens - 1
        # adjust x to the actual seq length for every sequence
        x = torch.nn.ReLU()(self.__output_layer(lstm_out))
        pred_full = self.__output_layer_1(x)
        x = pred_full[np.arange(0, len(seq_lens)), last_elements, :]
        return x, hidden, lstm_out, pred_full


class LstmNetEnsemble(nn.Module):
    def __init__(self, model_settings):
        super(LstmNetEnsemble, self).__init__()
        self.__n_window_height = model_settings['dct_coefficient_count']
        self.__n_classes = model_settings['label_count']
        self.__n_hidden_cells = model_settings['hidden_reccurent_cells_count']
        self.__num_nets = model_settings['num_nets']

        self.nets = nn.ModuleList([LstmNet(model_settings) for _ in range(self.__num_nets)])

    def forward(self, x, seq_lens, hidden=None, average=False):
        #
        if hidden is None:
            hidden = [None] * len(self.nets)
        x, hidden, lstm_out, pred_full = zip(*[self.nets[i](x, seq_lens, hidden[i]) for i in range(len(self.nets))])
        if average:
            x = x.mean(dim=0)
            hidden = hidden.mean(dim=0)
            lstm_out = lstm_out.mean(dim=0)
            pred_full = pred_full.mean(dim=0)
        return x, hidden, lstm_out, pred_full


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
    wanted_words_tanh_transition = ['a_a', 'a_i', 'a_u', 'a_o',
                                    'i_a', 'i_i', 'i_u', 'i_o',
                                    'u_a', 'u_i', 'u_u', 'u_o',
                                    'o_a', 'o_i', 'o_u', 'o_o' ]

    wanted_words_combined = wanted_words_tanh_transition


    model_settings = {
        'dct_coefficient_count': 26,
        'label_count': len(wanted_words_combined) + 2,
        'hidden_reccurent_cells_count': 128,
        'winlen': 0.04,
        'winstep': 0.04,
        'num_nets': 20
    }

    save_dir = r'C:\Study\SpeechAcquisitionModel\models\speech_classification'
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except:
            pass
    best_acc = 0.0
    lowest_loss = 100.0

    preproc = AudioPreprocessorFbank(nfilt=model_settings['dct_coefficient_count'], winlen=model_settings['winlen'], winstep=model_settings['winstep'])

    data_iter = SpeechCommandsDataCollector(preproc,
                                            data_dir=r'C:\Study\Speech_command_classification\speech_dataset',
                                            wanted_words=wanted_words_combined,
                                            testing_percentage=10,
                                            validation_percentage=10
                                            )
    net = LstmNetEnsemble(model_settings)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # configure training procedure
    n_train_steps = 5000
    n_mini_batch_size = 256

    for i in range(n_train_steps):
        # collect data
        d = data_iter.get_data(n_mini_batch_size, 0, 'training')
        data = d['x']
        # cut first 2 steps
        data = d['x'][:,2:,:]
        labels = d['y']
        seq_lengths = d['seq_len']
        max_seq_len = seq_lengths[0]-2
        seq_len = np.random.randint(10, max_seq_len)
        seq_lengths = np.array([seq_len] * len(seq_lengths))
        data = data[:, :seq_len, :]



        data = torch.from_numpy(data).float()
        labels = torch.from_numpy(labels).long()
        # seq_lengths = torch.from_numpy(seq_lengths)

        # zero grad

        losses = []
        accs = []
        pred, hidden, _, _ = net(data, seq_lengths)
        for e, p in enumerate(pred):
            # train just a few nets on a single batch (this is similar to training each net with an individual seed)
            if random.random() < 0.2 or (e == len(pred) - 1 and len(accs) == 0):
                optimizer.zero_grad()
                # pred = torch.stack(pred)
                # extend labels by number of nets
                loss = torch.nn.CrossEntropyLoss()(p, labels)
                loss.backward()
                optimizer.step()

                losses.append(loss.detach())

                acc = accuracy(p.detach().numpy(), labels.detach().numpy())
                accs.append(acc)
        print("|train_step: {}| loss: {:.4f} std:{:.4f}| accuracy: {:.4f} std:{:.4f}|".format(i, np.mean(losses), np.std(losses), np.mean(accs), np.std(accs)))
        # validate each 100 train steps
        if i % 100 == 0:
            d = data_iter.get_data(n_mini_batch_size, 0, 'validation')
            data = d['x'][:, 2:, :]
            labels = d['y']
            seq_lengths = d['seq_len'] - 2

            data = torch.from_numpy(data).float()
            labels = torch.from_numpy(labels).long()
            # zero grad
            optimizer.zero_grad()

            pred, hidden, _, _ = net(data, seq_lengths)

            pred = torch.stack(pred).mean(dim=0)

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

    pred, hidden, _, _ = net(data, seq_lengths)
    test_loss = torch.nn.CrossEntropyLoss()(pred, labels)
    acc = accuracy(pred.detach().numpy(), labels.detach().numpy())
    print("Test loss: {:.4f}| accuracy: {:.4f}|".format(test_loss.detach(), acc))