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


class SiameseLSTMNet(nn.Module):
    def __init__(self, model_settings):
        super(SiameseLSTMNet, self).__init__()
        self.__n_window_height = model_settings['dct_coefficient_count']
        self.__n_classes = model_settings['label_count']
        self.__n_hidden_cells = model_settings['hidden_reccurent_cells_count']
        self.__dropout = nn.Dropout(p=0.2)
        self.__bn1 = nn.BatchNorm1d(self.__n_window_height)
        # self.__compress_layer = nn.Linear(self.__n_hidden_cells, 10)
        self.__lstm = nn.LSTM(self.__n_window_height, self.__n_hidden_cells, batch_first=True, bidirectional=False)


        self.__output_layer = nn.Linear(self.__n_hidden_cells * 2, 512)
        self.__output_layer_1 = nn.Linear(512, 1)

        self.__output_layer_cce = nn.Linear(self.__n_hidden_cells, 512)
        self.__output_layer_cce_1 = nn.Linear(512, self.__n_classes)

    def forward(self, input, hidden=None):
        # if hidden is None:
        #     hidden = torch.zeros(x.size(0), self.n_hidden)
        lstm_out = []
        for i in range(2):
            x = input[i]
            orig_shape = x.shape
            x = x.view(-1, self.__n_window_height)
            x = self.__bn1(x)
            # # x = self.__dropout(x)
            x = x.view(orig_shape)
            x, hidden = self.__lstm(x, None)
            lstm_out.append(hidden[0])

        # cce path
        cce_output = torch.cat(lstm_out, dim=1).squeeze()
        cce_output = torch.nn.ReLU()(self.__output_layer_cce(cce_output))
        cce_output = self.__output_layer_cce_1(cce_output)

        # bce path
        lstm_out = torch.cat(lstm_out, dim=-1).squeeze()
        output = torch.nn.Tanh()(self.__output_layer(lstm_out))
        output = torch.nn.Sigmoid()(self.__output_layer_1(output))
        return output, cce_output


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels, axis=None) / float(labels.size)


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
        'winstep': 0.04
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

    # Summary writer
    dt = str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p"))
    writer = SummaryWriter(f'../../reports/seamise_net_{dt}')

    # configure training procedure
    n_train_steps = 10000
    n_mini_batch_size = 64

    siamese_net = SiameseLSTMNet(model_settings).to('cuda')
    optimizer = torch.optim.Adam(siamese_net.parameters(), lr=0.0005)

    dummy_input = torch.zeros(2, 1, 25, 26).cuda()
    writer.add_graph(siamese_net, dummy_input)
    """
    Train 
    """

    for i in range(n_train_steps):
        # collect data
        data = data_iter.get_data(n_mini_batch_size, 0, 'training')
        labels = data['y']

        duplicates = data_iter.get_duplicates(labels, 0, 'training')
        assert np.any(labels == duplicates['y'])

        non_duplicates = data_iter.get_nonduplicates(labels, 0, 'training')
        assert np.any(labels != non_duplicates['y'])

        # construct a tensor of a form [data duplicates, data non-duplicates]
        x = torch.from_numpy(np.array([np.concatenate([data['x'], data['x']]),
                                       np.concatenate([duplicates['x'], non_duplicates['x']])])).float().to('cuda')
        y_target = np.array([0] * len(labels) + [1]*len(labels))
        y_target = torch.from_numpy(y_target).float().to('cuda')
        # forward
        y, predicted_labels = siamese_net(x)

        #
        target_labels = torch.from_numpy(np.array([np.concatenate([data['y'],
                                                                   data['y'],
                                                                   duplicates['y'],
                                                                   non_duplicates['y']])]
                                                  )).long().to('cuda').squeeze()



        # backward and update
        optimizer.zero_grad()
        bce_loss = nn.BCELoss()(y, y_target)
        ce_loss = torch.nn.CrossEntropyLoss()(predicted_labels, target_labels)
        # loss = nn.BCELoss()(y, y_target) + torch.nn.CrossEntropyLoss()(predicted_labels, target_labels)
        loss = bce_loss + ce_loss
        loss.backward()
        optimizer.step()

        acc = accuracy(predicted_labels.detach().cpu().numpy(), target_labels.detach().cpu().numpy())
        writer.add_scalar('BCE', bce_loss.detach().cpu(), i)
        writer.add_scalar('CCE', ce_loss.detach().cpu(), i)
        writer.add_scalar('Classification Accuracy', acc, i)

        if i % 100 == 0:
            for name, param in siamese_net.named_parameters():
                writer.add_histogram("SiameseNet_" + name, param, i)

        print(i, loss.detach().cpu(), bce_loss.detach().cpu(), ce_loss.detach().cpu())
        # seq_lengths = d['seq_len']


if __name__ == '__main__':
    train()
    print('done')
