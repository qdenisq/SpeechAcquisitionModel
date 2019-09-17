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
from src.siamese_net_sound_similarity.soft_dtw import SoftDTW


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
        pick_deterministically = False
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
        pick_deterministically = False
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
        self.__n_hidden_reccurent_cells = model_settings['hidden_reccurent_cells_count']
        self.__n_hidden_cells = model_settings['hidden_latent_count']
        self.__dropout = nn.Dropout(p=0.2)
        self.__bn1 = nn.BatchNorm1d(self.__n_window_height)
        # self.__compress_layer = nn.Linear(self.__n_hidden_cells, 10)
        self.__lstm = nn.LSTM(self.__n_window_height, self.__n_hidden_reccurent_cells, batch_first=True, bidirectional=False)

        self.__linear_0 = nn.Linear(self.__n_hidden_reccurent_cells, self.__n_hidden_cells)

        self.__output_layer = nn.Linear(self.__n_hidden_cells * 2, 128)
        self.__output_layer_1 = nn.Linear(128, 1)

        self.__output_layer_cce = nn.Linear(self.__n_hidden_cells, 256)
        self.__output_layer_cce_1 = nn.Linear(256, self.__n_classes)

    def single_forward(self, input, hidden=None):
        x = input
        orig_shape = x.shape
        x = x.view(-1, self.__n_window_height)
        x = self.__bn1(x)
        x = self.__dropout(x)
        x = x.view(orig_shape)
        x, hidden = self.__lstm(x, None)
        x = self.__linear_0(x)
        hidden = x[:, -1, :]
        return x, hidden

    def forward(self, input, hidden=None):
        # if hidden is None:
        #     hidden = torch.zeros(x.size(0), self.n_hidden)
        lstm_out = []
        zs = []
        for i in range(2):
            x = input[i]
            x, hidden = self.single_forward(x)
            lstm_out.append(hidden)
            zs.append(x)

        # cce path
        cce_output = torch.cat(lstm_out, dim=0).squeeze()
        cce_output = torch.nn.ReLU()(self.__output_layer_cce(cce_output))
        cce_output = self.__output_layer_cce_1(cce_output)

        # bce path
        lstm_out = torch.cat(lstm_out, dim=-1).squeeze()
        output = torch.nn.Tanh()(self.__output_layer(lstm_out))
        output = torch.nn.Sigmoid()(self.__output_layer_1(output))
        return zs, output, cce_output


def anneal_function(anneal_func, step, k, x0):
    if anneal_func == 'logistic':
        return float(1 / (1 + np.exp(-k * (step - x0))))
    elif anneal_func == 'linear':
        return min(1, step / x0)


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
        'hidden_latent_count': 16,
        'winlen': 0.04,
        'winstep': 0.02
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
    n_train_steps = 100000
    n_mini_batch_size = 128

    siamese_net = SiameseLSTMNet(model_settings).to('cuda')
    optimizer = torch.optim.Adam(siamese_net.parameters(), lr=0.0005)

    # dummy_input = torch.zeros(2, 1, 25, 26).cuda()
    # writer.add_graph(siamese_net, dummy_input)
    """
    Train 
    """
    max_bce_acc = 0.
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
        y_target = np.array([0] * len(labels) + [1] * len(labels))
        y_target = torch.from_numpy(y_target).float().to('cuda')
        # forward
        zs, y, predicted_labels = siamese_net(x)

        #
        target_labels = torch.from_numpy(np.array([np.concatenate([data['y'],
                                                                   data['y'],
                                                                   duplicates['y'],
                                                                   non_duplicates['y']])]
                                                  )).long().to('cuda').squeeze()

        # backward and update
        optimizer.zero_grad()

        # bce loss
        bce_loss = nn.BCELoss()(y, y_target)

        # ce loss
        ce_loss = torch.nn.CrossEntropyLoss()(predicted_labels, target_labels)

        DTW_weight = anneal_function('logistic', i, 0.0025, 2500)

        # DTWLoss (want to minimize dtw between duplica)
        DTW_loss = torch.tensor([0]).float().cuda()
        for k in range(n_mini_batch_size):
            DTW_loss += SoftDTW()(zs[0][k], zs[1][k])
        for k in range(n_mini_batch_size, 2 * n_mini_batch_size):
            DTW_loss -= SoftDTW()(zs[0][k], zs[1][k])

        DTW_loss = torch.nn.functional.relu(DTW_loss + 5.0)

        DTW_loss /= (2 * n_mini_batch_size * zs[0].shape[1])

        # loss = bce_loss + ce_loss + KL_loss * KL_weight
        loss = bce_loss + ce_loss + DTW_loss * DTW_weight
        loss.backward()
        optimizer.step()

        cce_acc = accuracy(predicted_labels.detach().cpu().numpy(), target_labels.detach().cpu().numpy())
        y = np.array([[1.0 - v, v] for v in y.detach().cpu().numpy()]).squeeze()
        bce_acc = accuracy(y, y_target.detach().cpu().numpy())
        writer.add_scalar('BCE', bce_loss.detach().cpu(), i)
        writer.add_scalar('CCE', ce_loss.detach().cpu(), i)
        writer.add_scalar('DTW_loss', DTW_loss.detach().cpu(), i)
        writer.add_scalar('CCE_Accuracy', cce_acc, i)
        writer.add_scalar('BCE_Accuracy', bce_acc, i)
        writer.add_scalar('DTW_Weight', DTW_weight, i)

        if i % 500 == 0:
            for name, param in siamese_net.named_parameters():
                writer.add_histogram("SiameseNet_" + name, param, i)

        if bce_acc > max_bce_acc or i % 1000 == 0:
            if bce_acc > max_bce_acc:
                max_bce_acc = bce_acc
            fname = os.path.join(f'../../reports/seamise_net_{dt}/net_{bce_acc}.net')
            torch.save(siamese_net, fname)

        print(f"{i} "
              f"| L: {loss.detach().cpu().numpy()} "
              f"| BCE {bce_loss.detach().cpu().numpy()} "
              f"| CE {ce_loss.detach().cpu().numpy()} "
              f"| DTW {DTW_loss.detach().cpu()}")

        # TODO: add validation each 1k steps for example
        if i % 500 == 0:
            # collect data
            data = data_iter.get_data(n_mini_batch_size, 0, 'validation')
            labels = data['y']

            duplicates = data_iter.get_duplicates(labels, 0, 'validation')
            assert np.any(labels == duplicates['y'])

            non_duplicates = data_iter.get_nonduplicates(labels, 0, 'validation')
            assert np.any(labels != non_duplicates['y'])

            # construct a tensor of a form [data duplicates, data non-duplicates]
            x = torch.from_numpy(np.array([np.concatenate([data['x'], data['x']]),
                                           np.concatenate([duplicates['x'], non_duplicates['x']])])).float().to('cuda')
            y_target = np.array([0] * len(labels) + [1] * len(labels))
            y_target = torch.from_numpy(y_target).float().to('cuda')
            # forward
            zs, y, predicted_labels = siamese_net(x)

            #
            target_labels = torch.from_numpy(np.array([np.concatenate([data['y'],
                                                                       data['y'],
                                                                       duplicates['y'],
                                                                       non_duplicates['y']])]
                                                      )).long().to('cuda').squeeze()

            # backward and update
            # bce loss
            bce_loss = nn.BCELoss()(y, y_target)

            # CrossEntropy loss
            ce_loss = torch.nn.CrossEntropyLoss()(predicted_labels, target_labels)

            # KL divergence regularization
            loss = bce_loss + ce_loss

            y = np.array([[1.0 - v, v] for v in y.detach().cpu().numpy()]).squeeze()

            cce_acc = accuracy(predicted_labels.detach().cpu().numpy(), target_labels.detach().cpu().numpy())
            bce_acc = accuracy(y, y_target.detach().cpu().numpy())
            writer.add_scalar('valid_BCE', bce_loss.detach().cpu(), i)
            writer.add_scalar('valid_CCE', ce_loss.detach().cpu(), i)
            writer.add_scalar('valid_CCE_Accuracy', cce_acc, i)
            writer.add_scalar('valid_BCE_Accuracy', bce_acc, i)

            print('validation: ', i, loss.detach().cpu(), bce_loss.detach().cpu(), ce_loss.detach().cpu())


if __name__ == '__main__':
    train()
    print('done')
