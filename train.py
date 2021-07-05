import os
import torch
from scipy import io
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
import math
import argparse
import pickle
from tqdm import tqdm
import network.cnn as CNN
import network.lstm as LSTM
import network.dataset as DS

parser = argparse.ArgumentParser(description='Training CombSleepNet')
parser.add_argument('--data_dir', type=str, default="./pre-processing/",
                    help='pre-processed data dir')
parser.add_argument('--input_type', type=str,
                    help='SleepEDF, SHHS, male_SHHS, female_SHHS', default='SleepEDF')
parser.add_argument('--out_dir', type=str, default='./parameter/',
                    help='path where to save the parameters')
parser.add_argument('--seq_len', type=int, default=20,
                    help='sequence length (default: 20)')
parser.add_argument('--cnn_lr', type=float, default=1e-5,
                    help='learning rate of cnn')
parser.add_argument('--lstm_lr', type=float, default=1e-3,
                    help='learning rate of lstm')
parser.add_argument('--cnn_epoch', type=int, default=30,
                    help='epoch number of cnn')
parser.add_argument('--lstm_epoch', type=int, default=15,
                    help='epoch number of lstm')
parser.add_argument('--cv', type=int, default=20,
                    help='number of cross-validation')
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

GPU_NUM = args.gpu
device = 'cuda:{:d}'.format(GPU_NUM)
torch.cuda.set_device(device)
print('Current device ', torch.cuda.current_device())  # check
use_cuda = device
print(torch.cuda.get_device_name(device))
print('Memory Usage:')
print('Allocated:', round(torch.cuda.memory_allocated(device) / 1024 ** 3, 1), 'GB')

def preprocess_data(path, filename):
    f = io.loadmat(path + filename)
    out = f.get('psg')
    return out

def load_header(path, filename):
    f = io.loadmat(path + filename)
    out = f.get('hyp')[0]
    return out

def loss(model_output, true_label, cf):
    out = 0
    for i, item in enumerate(model_output):
        item2 = torch.unsqueeze(item, 0)
        t = torch.unsqueeze(true_label[i], 0)
        if model_output[i].argmax() == true_label[i]:
            w = 1
        else:
            if cf[true_label[i]][model_output[i].argmax()] < 0.01:
                w = 1
            else:
                w = 100 * cf[true_label[i]][model_output[i].argmax()]
        out += w * F.cross_entropy(item2, t)
    return out

if args.input_type == 'SleepEDF':
    _psg = 'SleepEDF/psg/'
    _hyp = 'SleepEDF/hyp/'

elif args.input_type == 'SHHS':
    _psg = 'SHHS/psg/'
    _hyp = 'SHHS/hyp/'
    args.cv = 1

elif args.input_type == 'male_SHHS':
    _psg = 'male_SHHS/psg/'
    _hyp = 'male_SHHS/hyp/'
    args.cv = 1

elif args.input_type == 'female_SHHS':
    _psg = 'female_SHHS/psg/'
    _hyp = 'female_SHHS/hyp/'
    args.cv = 1


path = args.data_dir
psg_filepath = path + _psg
hyp_filepath = path + _hyp

psg_filelist = os.listdir(psg_filepath)
hyp_filelist = os.listdir(hyp_filepath)

psg_train = []
hyp_train = []
psg_val = []
hyp_val = []
psg_test = []
hyp_test = []

if args.input_type == 'SHHS' or args.input_type == 'male_SHHS' or args.input_type == 'female_SHHS':
    n_files = len(psg_filelist)
    n_train = round(n_files * 0.5)
    n_val = round(n_files * 0.3)
    n_test = round(n_files * 0.2)
    if n_train + n_val + n_test > n_files:
        while n_train + n_val + n_test != n_files:
            n_train = n_train - 1
    elif n_train + n_val + n_test < n_files:
        while n_train + n_val + n_test != n_files:
            n_train = n_train + 1
    print('Total number of dataset: {:d}'.format(n_files))
    print('Number of training set: {:d}, validation set: {:d}, test set: {:d}'.format(n_train, n_val, n_test))
    number_list = [x for x in range(n_files)]
    train = number_list[0: n_train]
    val = number_list[n_train: (n_train + n_val)]
    test = number_list[(n_train + n_val):]
    print('Loading training dataset...')
    for i in tqdm(train):
        psg_train.append(preprocess_data(psg_filepath, psg_filelist[i]))
        hyp_train.append(load_header(hyp_filepath, hyp_filelist[i]))
    print('Loading validation dataset...')
    for i in tqdm(val):
        psg_val.append(preprocess_data(psg_filepath, psg_filelist[i]))
        hyp_val.append(load_header(hyp_filepath, hyp_filelist[i]))
    print('Data loading completed')
    for i in tqdm(test):
        psg_test.append(preprocess_data(psg_filepath, psg_filelist[i]))
        hyp_test.append(load_header(hyp_filepath, hyp_filelist[i]))
    print('Loading test dataset...')

else:
    for i in range(len(psg_filelist)):
        psg_train.append(preprocess_data(psg_filepath, psg_filelist[i]))
        hyp_train.append(load_header(hyp_filepath, hyp_filelist[i]))

    if args.cv == 14:
        i = [26]
    elif args.cv < 14:
        i = [2 * (args.cv - 1), 2 * (args.cv - 1) + 1]
    elif args.cv > 14:
        i = [2 * (args.cv - 1) - 1, 2 * (args.cv - 1)]

    for ii in i:
        psg_val.append(preprocess_data(psg_filepath, psg_filelist[ii]))
        hyp_val.append(load_header(hyp_filepath, hyp_filelist[ii]))

    if len(i) == 1:
        del psg_train[i[0]]
        del hyp_train[i[0]]
    else:
        del psg_train[i[0]:i[1] + 1]
        del hyp_train[i[0]:i[1] + 1]

num_layers = 2
cnn_batch_size = 64
rnn_batch_size = 16
hidden_size = 5
input_size = 5

trainCNN = DS.CustomDataset(psg_train, hyp_train, True, True, args.seq_len)
trainLSTM = DS.CustomDataset(psg_train, hyp_train, True, False, args.seq_len)

if args.input_type == 'SHHS' or args.input_type == 'male_SHHS' or args.input_type == 'female_SHHS':
    testCNN = DS.CustomDataset(psg_val, hyp_val, False, True, args.seq_len)
    testLSTM = DS.CustomDataset(psg_val, hyp_val, False, False, args.seq_len)
    with open('{:s}_testset.pickle'.format(args.input_type), 'wb') as f:
        pickle.dump(test, f, pickle.HIGHEST_PROTOCOL)
        print("Saved indices for test dataset list as '{:s}_'testset.pickle'...".format(args.input_type))

else:
    testCNN = DS.CustomDataset(psg_val, hyp_val, False, True, args.seq_len)
    testLSTM = DS.CustomDataset(psg_val, hyp_val, False, False, args.seq_len)

test_dataset = DS.CustomDataset(psg_test, hyp_test, False, False, args.seq_len)

trainDataloader1 = DataLoader(trainCNN, batch_size=cnn_batch_size, shuffle=True)
trainDataloader2 = DataLoader(trainLSTM, batch_size=rnn_batch_size, shuffle=True)
n_channel = psg_train[0].shape[1]
print("Input type: {:s}".format(args.input_type))
print('Number of frequencies:', n_channel)

cnn = CNN.CNNClassifier(channel=n_channel)
if args.input_type == 'SHHS' or args.input_type == 'male_SHHS' or args.input_type == 'female_SHHS':
    cnn = CNN.CNNClassifier(channel=n_channel, SHHS=True)

criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(cnn.parameters(), lr=args.cnn_lr, weight_decay=0.003)
cnn_num_batches = len(trainDataloader1)
interval = cnn_num_batches // 10
lstm = LSTM.LSTMClassifier(input_size, hidden_size, num_layers)
optimizer2 = optim.Adam(lstm.parameters(), lr=args.lstm_lr, weight_decay=0.003)
rnn_num_batches = len(trainDataloader2)

acc = 0
F1 = 0
max_acc = 0
max_F1 = 0

for epoch in tqdm(range(args.cnn_epoch)):
    train_loss = 0.0
    pred_list_tr = []
    corr_list_tr = []

    for i, data in enumerate(trainDataloader1):
        train_x, train_y = data
        train_x = train_x.view(train_x.size(0), 1, train_x.size(1), train_x.size(2))
        train_y = train_y.type(dtype=torch.int64)

        if use_cuda:
            train_x = train_x.cuda()
            train_y = train_y.cuda()

        optimizer1.zero_grad()
        train_output = F.softmax(cnn(train_x, True), 1)

        if epoch < 1:
            train_l = criterion(train_output, train_y)
        else:
            train_l = criterion(train_output, train_y)
        expected_train_y = train_output.argmax(dim=1)
        train_l.backward()
        optimizer1.step()

        corr_list_tr.extend(list(np.hstack(train_y.cpu())))
        pred_list_tr.extend(list(np.hstack(expected_train_y.cpu())))

        train_loss += train_l.item()
        del train_l
        del train_output

        if (i + 1) % interval == 0:
            with torch.no_grad():
                corr_num = 0
                total_num = 0
                pred_list = []
                corr_list = []

                for j, test_x in enumerate(testCNN.x_data):
                    test_y = testCNN.y_data[j]
                    test_x = torch.as_tensor(test_x)
                    test_x = test_x.view(test_x.size(0), 1, 1, -1)
                    test_y = torch.as_tensor(test_y)
                    test_y = test_y.type(dtype=torch.int64)

                    if use_cuda:
                        test_x = test_x.cuda()
                        test_y = test_y.cuda()
                    test_output = F.softmax(cnn(test_x, True), 1)
                    expected_test_y = test_output.argmax(dim=1)
                    corr_list.append(test_y.view(-1).detach().cpu().numpy())
                    pred_list.append(expected_test_y.view(-1).detach().cpu().numpy())

            corr_list = np.array(np.concatenate(corr_list))
            pred_list = np.array(np.concatenate(pred_list))
            train_loss = 0.0
            test_cf = confusion_matrix(corr_list, pred_list)

            acc = int(sum(pred_list == corr_list)) / len(corr_list)
            F1 = f1_score(corr_list, pred_list, average='macro')

            if max_F1 < F1:
                torch.save(cnn.state_dict(),
                           args.out_dir + "cnn_IP({:s})_SL({:d})_CV({:d}).pt"
                           .format(args.input_type, args.seq_len, args.cv))
                max_F1 = F1
                print("epoch: {}/{} | step: {}/{} | acc: {:.2f} | F1 score: {:.2f}"
                      .format(epoch + 1, args.cnn_epoch, i + 1, cnn_num_batches, acc * 100, F1 * 100))
                print(test_cf)
            if max_acc < acc:
                max_acc = acc

    train_cf = confusion_matrix(corr_list_tr, pred_list_tr)
    cf_F1 = []
    for ii in range(5):
        for jj in range(5):
            cf_F1.append((2 * train_cf[ii][jj]) / (sum(train_cf[ii]) + sum(np.transpose(train_cf)[jj])))

    cf_F1 = torch.tensor(cf_F1).reshape([5, 5])
    if use_cuda:
        cf_F1 = cf_F1.cuda()

    print("epoch: {}/{} | step: {}/{} | acc: {:.2f} | F1 score: {:.2f}"
          .format(epoch + 1, args.cnn_epoch, i + 1, cnn_num_batches, acc * 100, F1 * 100))
    print(train_cf)
    print(cf_F1)

train_cf = []
acc = 0
max_acc = 0
max_acc = 0
max_F1 = 0
cnn.load_state_dict(torch.load(
    args.out_dir + "cnn_IP({:s})_SL({:d})_CV({:d}).pt".format(args.input_type, args.seq_len, args.cv)))

print('CNN stage is done, starting Bi-LSTM stage...')
for epoch in tqdm(range(args.lstm_epoch)):
    print('Epoch:', epoch)
    train_loss = 0.0
    pred_list_tr = []
    corr_list_tr = []
    for i, data in enumerate(trainDataloader2):
        train_x, train_y = data
        hidden, cell = lstm.init_hidden(train_x.shape[0])
        train_x = train_x.squeeze().view(-1, 1, train_x.size(2), train_x.size(3))
        train_y = train_y.squeeze().type(dtype=torch.int64)

        if use_cuda:
            train_x = train_x.cuda()
            train_y = train_y.cuda()

        optimizer2.zero_grad()
        output = F.softmax(cnn(train_x, True), 1)
        output = output.view(-1, args.seq_len, 5)
        train_output = F.softmax(lstm(output, hidden, cell, args.seq_len), 1)
        train_y = train_y.view(train_y.shape[0] * train_y.shape[1])
        if epoch < args.lstm_epoch // 3:
            train_l = criterion(train_output, train_y)
        else:
            train_l = criterion(train_output, train_y)

        expected_train_y = train_output.argmax(dim=1)

        train_l.backward()
        optimizer2.step()
        corr_list_tr.extend(list(np.hstack(train_y.cpu())))
        pred_list_tr.extend(list(np.hstack(expected_train_y.cpu())))
        train_loss += train_l.item()
        del train_l
        del train_output
        del output

        if (i + 1) % interval == 0:
            with torch.no_grad():
                corr_num = 0
                total_num = 0
                pred_list = []
                corr_list = []

                for j, x in enumerate(testLSTM.x_data):
                    y = testLSTM.y_data[j]
                    for jj, test_x in enumerate(x):
                        hidden, cell = lstm.init_hidden(1)
                        test_y = y[jj]
                        test_x = torch.as_tensor(test_x)
                        test_x = test_x.squeeze().view(test_x.size(0), -1, test_x.size(1), test_x.size(2))
                        test_y = torch.as_tensor(test_y)
                        test_y = test_y.type(dtype=torch.int64)

                        if use_cuda:
                            test_x = test_x.cuda()
                            test_y = test_y.cuda()

                        output = F.softmax(cnn(test_x, True), 1)
                        test_output = F.softmax(lstm(output, hidden, cell, args.seq_len), 1)
                        expected_test_y = test_output.argmax(dim=1)

                        corr = test_y[test_y == expected_test_y].size(0)
                        corr_num += corr

                        total_num += test_y.size(0)
                        corr_list.extend(list(np.hstack(test_y.cpu())))
                        pred_list.extend(list(np.hstack(expected_test_y.cpu())))

            train_loss = 0.0
            test_cf = confusion_matrix(corr_list, pred_list)

            acc = corr_num / total_num
            F1 = f1_score(corr_list, pred_list, average='macro')

            if max_F1 < F1:
                torch.save(lstm.state_dict(),
                           args.out_dir + "lstm_IP({:s})_SL({:d})_CV({:d}).pt"
                           .format(args.input_type, args.seq_len, args.cv))
                print("epoch: {}/{} | step: {}/{} | acc: {:.2f} | F1 score: {:.2f}"
                      .format(epoch + 1, args.lstm_epoch, i + 1, rnn_num_batches, acc * 100, F1 * 100))
                print(test_cf)
            if max_acc < acc:
                max_acc = acc

    train_cf = confusion_matrix(corr_list_tr, pred_list_tr)
    cf_F1 = []
    for ii in range(5):
        for jj in range(5):
            cf_F1.append((2 * train_cf[ii][jj]) / (sum(train_cf[ii]) + sum(np.transpose(train_cf)[jj])))

    cf_F1 = torch.tensor(cf_F1).reshape([5, 5])
    if use_cuda:
        cf_F1 = cf_F1.cuda()

    print("epoch: {}/{} | step: {}/{} | acc: {:.2f} | F1 score: {:.2f}"
          .format(epoch + 1, args.lstm_epoch, i + 1, rnn_num_batches, acc * 100, F1 * 100))
    print(train_cf)
    print(cf_F1)
