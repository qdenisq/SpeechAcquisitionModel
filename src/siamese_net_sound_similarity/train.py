import os
import datetime

import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from six.moves import xrange

from python_speech_features import mfcc
import scipy.io.wavfile as wav

from src.speech_classification.audio_processing import AudioPreprocessorFbank, SpeechCommandsDataCollector
import src.speech_classification.utils as utils


class SiameseSpeechCommandsDataCollector(SpeechCommandsDataCollector):
    def get_duplicates(self, labels, offset, mode):
        # Pick one of the partitions to choose samples from.
        candidates = self.data_index[mode]
        how_many = len(labels)
        duplicate_labels = np.zeros(len(labels))
        if how_many == -1:
            sample_count = len(candidates)
        else:
            sample_count = max(0, min(how_many, len(candidates) - offset))
        # Data and labels will be populated and returned.
        data = []
        seq_lens = []
        # labels = np.zeros(sample_count)
        pick_deterministically = (mode != 'training')
        # Use the processing graph we created earlier to repeatedly to generate the
        # final output sample data we'll use in training.
        for i in xrange(offset, offset + sample_count):
            # Pick which audio sample to use.
            if how_many == -1 or pick_deterministically:
                sample_index = i
            else:
                while True:
                    sample_index = np.random.randint(len(candidates))
                    if self.word_to_index[candidates[sample_index]['label']] == labels[i]:
                        break
            sample = candidates[sample_index]

            sample_data = self.get_sequential_data_sample(sample['file'])
            seq_len = sample_data.shape[0]
            data.append(sample_data)
            seq_lens.append(seq_len)
            label_index = self.word_to_index[sample['label']]
            duplicate_labels[i - offset] = label_index
        max_seq_len = max(seq_lens)
        zero_padded_data = [np.append(s, np.zeros((max_seq_len - s.shape[0], s.shape[1])), axis=0) for s in data]
        data = np.stack(zero_padded_data)
        seq_lens = np.array(seq_lens)
        duplicate_labels = np.array(duplicate_labels)
        return {'x': data,
                'y': duplicate_labels,
                'seq_len': seq_lens}

    def get_nonduplicates(self, labels, offset, mode):
        # Pick one of the partitions to choose samples from.
        candidates = self.data_index[mode]
        how_many = len(labels)
        nonduplicate_labels = np.zeros(len(labels))
        if how_many == -1:
            sample_count = len(candidates)
        else:
            sample_count = max(0, min(how_many, len(candidates) - offset))
        # Data and labels will be populated and returned.
        data = []
        seq_lens = []
        # labels = np.zeros(sample_count)
        pick_deterministically = (mode != 'training')
        # Use the processing graph we created earlier to repeatedly to generate the
        # final output sample data we'll use in training.
        for i in xrange(offset, offset + sample_count):
            # Pick which audio sample to use.
            if how_many == -1 or pick_deterministically:
                sample_index = i
            else:
                while True:
                    sample_index = np.random.randint(len(candidates))
                    if self.word_to_index[candidates[sample_index]['label']] != labels[i]:
                        break
            sample = candidates[sample_index]

            sample_data = self.get_sequential_data_sample(sample['file'])
            seq_len = sample_data.shape[0]
            data.append(sample_data)
            seq_lens.append(seq_len)
            label_index = self.word_to_index[sample['label']]
            nonduplicate_labels[i - offset] = label_index
        max_seq_len = max(seq_lens)
        zero_padded_data = [np.append(s, np.zeros((max_seq_len - s.shape[0], s.shape[1])), axis=0) for s in data]
        data = np.stack(zero_padded_data)
        seq_lens = np.array(seq_lens)
        duplicate_labels = np.array(nonduplicate_labels)
        return {'x': data,
                'y': duplicate_labels,
                'seq_len': seq_lens}

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


def train():
    #############################################################
    # Simple Lstm train script
    #############################################################

    """
    Initialize
    """

    wanted_words = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left',
                    'marvin',
                    'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two',
                    'up', 'wow', 'yes', 'zero']

    wanted_words_combined = wanted_words

    model_settings = {
        'dct_coefficient_count': 26,
        'label_count': len(wanted_words_combined) + 2,
        'hidden_reccurent_cells_count': 128,
        'winlen': 0.04,
        'winstep': 0.04,
        'num_nets': 20
    }

    save_dir = r'C:\Study\SpeechAcquisitionModel\models\siamese_net_sound_similarity'
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except:
            pass

    preproc = AudioPreprocessorFbank(nfilt=model_settings['dct_coefficient_count'], winlen=model_settings['winlen'],
                                     winstep=model_settings['winstep'])

    data_iter = SiameseSpeechCommandsDataCollector(preproc,
                                            data_dir=r'C:\Study\Speech_command_classification\speech_dataset',
                                            wanted_words=wanted_words_combined,
                                            testing_percentage=10,
                                            validation_percentage=10
                                            )

    # configure training procedure
    n_train_steps = 1000
    n_mini_batch_size = 256

    """
    Train 
    """

    for i in range(n_train_steps):
        # collect data
        d = data_iter.get_data(n_mini_batch_size, 0, 'training')
        data = d['x']
        # cut first 2 steps
        data = d['x'][:, 2:, :]
        labels = d['y']

        duplicates = data_iter. get_duplicates(labels, 0, 'training')
        non_duplicates = data_iter.get_nonduplicates(labels, 0, 'training')

        k = 2
        # seq_lengths = d['seq_len']






if __name__ == '__main__':
    train()
    print('done')
