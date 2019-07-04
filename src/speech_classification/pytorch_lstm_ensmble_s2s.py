import src.speech_classification.utils as utils
from src.speech_classification.audio_processing import AudioPreprocessorFbank, SpeechCommandsDataCollector
from sklearn import preprocessing
import torch.nn.functional as F
import torch.nn as nn
import torch
import random

import os
import datetime
from multiprocessing import Pool
import pandas as pd


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



    phonemes = ['a', 'i', 'o', 'u']
    phoneme_transitions = ['ai', 'au', 'ao',
                            'ia', 'iu', 'io',
                            'ua', 'ui', 'uo',
                            'oa', 'oi', 'ou']
    total_classes =  ['sil', 'none'] + phonemes + phoneme_transitions

    le = preprocessing.LabelEncoder()
    le.fit(total_classes)
    model_settings = {
        'dct_coefficient_count': 26,
        'label_count': len(total_classes),
        'hidden_reccurent_cells_count': 128,
        'winlen': 0.04,
        'winstep': 0.04,
        'num_nets': 10
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

    data_dir = r'C:\Study\Speech_command_classification\speech_dataset'
    dataset_fname = r'C:\Study\SpeechAcquisitionModel\data\raw\Simple_transitions_s2s\07_04_2019_01_11_PM_31\07_04_2019_01_11_PM_31.pd'
    df = pd.read_pickle(dataset_fname)

    processed_audio = []
    labels_int = []
    print("preprocessing audio...")

    for i in range(df.shape[0]):
        print(f"\r{i}", end="")
        sr = 22050
        sample = df.iloc[i]
        audio = sample['audio'].flatten()
        audio_proc = preproc(audio, sr)
        processed_audio.append(audio_proc)
        labels_int.append(le.transform(sample['labels']))

    processed_audio = np.array(processed_audio)
    labels_int = np.array(labels_int)

    net = LstmNetEnsemble(model_settings)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # configure training procedure
    n_train_steps = 5000
    n_mini_batch_size = 256

    for i in range(n_train_steps):
        # collect data
        batch_idx = np.random.randint(0, df.shape[0], n_mini_batch_size)

        data = processed_audio[batch_idx, :]
        labels = labels_int[batch_idx, :]
        # labels_int = np.array([le.transform(s) for s in labels])



        seq_lengths = [labels.shape[-1]] * n_mini_batch_size
        max_seq_len = seq_lengths[0]-2
        # seq_len = np.random.randint(10, max_seq_len)
        seq_begin = 2
        # seq_begin = np.random.randint(2, max_seq_len-2)
        # seq_end = np.random.randint(seq_begin + 1, max_seq_len)
        seq_end = seq_lengths[0]
        seq_len = seq_end - seq_begin
        seq_lengths = np.array([seq_len] * len(seq_lengths))
        data = data[:, seq_begin:seq_end, :]
        labels = labels[:, seq_begin:seq_end]
        data = torch.from_numpy(data).float()
        labels = torch.from_numpy(labels).long()
        # seq_lengths = torch.from_numpy(seq_lengths)

        # zero grad

        losses = []
        accs = []
        pred, hidden, _, pred_full = net(data, seq_lengths)
        for e, p in enumerate(pred_full):
            # train just a few nets on a single batch (this is similar to training each net with an individual seed)
            if random.random() < 1.0 or (e == len(pred) - 1 and len(accs) == 0):
                optimizer.zero_grad()
                # pred = torch.stack(pred)
                # extend labels by number of nets
                loss = torch.nn.CrossEntropyLoss()(p.view(-1, len(total_classes)), labels.flatten())
                loss.backward()
                optimizer.step()

                losses.append(loss.detach())

                acc = accuracy(p.view(-1, len(total_classes)).detach().numpy(), labels.flatten().detach().numpy())
                accs.append(acc)
        print("|train_step: {}| loss: {:.4f} std:{:.4f}| accuracy: {:.4f} std:{:.4f}|".format(i, np.mean(losses), np.std(losses), np.mean(accs), np.std(accs)))
        # validate each 100 train steps
        if i % 100 == 0:
            batch_idx = np.random.randint(0, df.shape[0], n_mini_batch_size)

            data = processed_audio[batch_idx, :]
            labels = labels_int[batch_idx, :]
            # labels_int = np.array([le.transform(s) for s in labels])

            seq_lengths = [labels.shape[-1]] * n_mini_batch_size
            max_seq_len = seq_lengths[0] - 2
            # seq_len = np.random.randint(10, max_seq_len)
            seq_begin = 2
            # seq_begin = np.random.randint(2, max_seq_len-2)
            # seq_end = np.random.randint(seq_begin + 1, max_seq_len)
            seq_end = seq_lengths[0]
            seq_len = seq_end - seq_begin
            seq_lengths = np.array([seq_len] * len(seq_lengths))
            data = data[:, seq_begin:seq_end, :]
            labels = labels[:, seq_begin:seq_end]
            data = torch.from_numpy(data).float()
            labels = torch.from_numpy(labels).long()


            optimizer.zero_grad()

            pred, hidden, _, pred_full = net(data, seq_lengths)

            pred_full = torch.stack(pred_full).mean(dim=0)

            validation_loss = torch.nn.CrossEntropyLoss()(pred_full.detach().view(-1, len(total_classes)), labels.detach().flatten())
            acc = accuracy(pred_full.detach().view(-1, len(total_classes)).detach().numpy(), labels.detach().flatten().numpy())
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