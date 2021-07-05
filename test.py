import os
import torch
from scipy import io
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np
import argparse
import pickle
import network.cnn as CNN
import network.lstm as LSTM
import network.dataset as DS

parser = argparse.ArgumentParser(description='Training CombSleepNet')
parser.add_argument('--data_dir', type=str, default='./pre-processing/',
                    help='pre-processed data dir')
parser.add_argument('--parameter_dir', type=str, default='./parameter/',
                    help='parameter dir')
parser.add_argument('--out_dir', type=str, default='./result/',
                    help='path where to save the results')
parser.add_argument('--input_type', type=str,
                    help='SleepEDF, SHHS, male_SHHS, female_SHHS', default='SleepEDF')
parser.add_argument('--seq_len', type=int, default=20,
                    help='sequence length (default: 20)')
parser.add_argument('--cnn_lr', type=float, default=1e-5,
                    help='learning rate of cnn')
parser.add_argument('--lstm_lr', type=float, default=1e-3,
                    help='learning rate of lstm')
parser.add_argument('--cv', type=int, default=20,
                    help='number of cross-validation (1-20)')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

GPU_NUM = args.gpu
device = torch.device('cuda:{:d}'.format(GPU_NUM) if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
print('Current device ', torch.cuda.current_device()) # check
if device.type == 'cuda':
    use_cuda = device
    print(torch.cuda.get_device_name(device))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(device)/1024**3,1), 'GB')

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

psg_test = []
hyp_test = []
test_list = []
if args.input_type == 'SHHS' or args.input_type == 'male_SHHS' or args.input_type == 'female_SHHS':
    with open('{:s}_testset.pickle'.format(args.input_type), 'rb') as file:
        test_list = pickle.load(file)
    print('Loading test dataset...')
    print('Number of test dataset: {:d}'.format(len(test_list)))
    for i in tqdm(test_list):
        psg_test.append(preprocess_data(psg_filepath, psg_filelist[i]))
        hyp_test.append(load_header(hyp_filepath, hyp_filelist[i]))

else:
    if args.cv == 14:
        i = [26]
    elif args.cv < 14:
        i = [2 * (args.cv - 1), 2 * (args.cv - 1) + 1]
    elif args.cv > 14:
        i = [2 * (args.cv - 1) - 1, 2 * (args.cv - 1)]

    for ii in i:
        psg_test.append(preprocess_data(psg_filepath, psg_filelist[ii]))
        hyp_test.append(load_header(hyp_filepath, hyp_filelist[ii]))

num_layers = 2
hidden_size = 5
input_size = 5

testDataset1 = DS.CustomDataset(psg_test, hyp_test, False, True, args.seq_len)
testDataset2 = DS.CustomDataset(psg_test, hyp_test, False, False, args.seq_len)

criterion = nn.CrossEntropyLoss()

n_channel = psg_test[0].shape[1]
cnn = CNN.CNNClassifier(channel=n_channel)
if args.input_type == 'SHHS' or args.input_type == 'male_SHHS' or args.input_type == 'female_SHHS':
    cnn = CNN.CNNClassifier(channel=n_channel, SHHS=True)

lstm = LSTM.LSTMClassifier(input_size, hidden_size, num_layers)

cnn.load_state_dict(torch.load(args.parameter_dir + "cnn_IP({:s})_SL({:d})_CV({:d}).pt"
                               .format(args.input_type, args.seq_len, args.cv)))
lstm.load_state_dict(torch.load(args.parameter_dir + "lstm_IP({:s})_SL({:d})_CV({:d}).pt"
                                .format(args.input_type, args.seq_len, args.cv)))

test_loss_list = []

flabel1 = open(args.out_dir + "label1_IP({:s})_SL({:d})_CV({:d}).pt"
               .format(args.input_type, args.seq_len, args.cv), 'w')
flabel2 = open(args.out_dir + "label2_IP({:s})_SL({:d})_CV({:d}).pt"
               .format(args.input_type, args.seq_len, args.cv), 'w')

with torch.no_grad():
    corr_num = 0
    total_num = 0
    pred_list = []
    corr_list = []

    for j, x in tqdm(enumerate(testDataset2.x_data)):
        y = testDataset2.y_data[j]
        hidden, cell = lstm.init_hidden(1)
        for jj, test_x in enumerate(x):
            test_y = y[jj]
            test_x = torch.as_tensor(test_x)
            test_x = test_x.squeeze().view(test_x.size(0), 1, test_x.size(1), test_x.size(2))
            test_y = torch.as_tensor(test_y)
            test_y = test_y.type(dtype=torch.int64)

            if use_cuda:
                test_x = test_x.cuda()
                test_y = test_y.cuda()

            output = F.softmax(cnn(test_x, True), 1)
            test_output2 = F.softmax(lstm(output, hidden, cell, args.seq_len), 1)
            expected_test_y2 = test_output2.argmax(dim=1)

            corr = test_y[test_y == expected_test_y2].size(0)
            corr_num += corr

            total_num += test_y.size(0)
            corr_list.extend(list(np.hstack(test_y.cpu())))
            pred_list.extend(list(np.hstack(expected_test_y2.cpu())))

            if j == 0:
                for k in test_output2:
                    for kk in k:
                        flabel1.write(str(float(kk)))
                        flabel1.write(" ")
                    flabel1.write(";")
                flabel1.write("\n\n")

            if j == 1:
                for k in test_output2:
                    for kk in k:
                        flabel2.write(str(float(kk)))
                        flabel2.write(" ")
                    flabel2.write(";")
                flabel2.write("\n\n")

test_cf = confusion_matrix(corr_list, pred_list)

cf_F1 = []
for ii in range(5):
    for jj in range(5):
        cf_F1.append((2 * test_cf[ii][jj]) / (sum(test_cf[ii]) + sum(np.transpose(test_cf)[jj])))

cf_F1 = torch.tensor(cf_F1).reshape([5, 5])
if use_cuda:
    cf_F1 = cf_F1.cuda()
acc = corr_num / total_num * 100
F1 = (cf_F1[0][0] + cf_F1[1][1] + cf_F1[2][2] + cf_F1[3][3] + cf_F1[4][4]) / 5
print("acc: {:.2f}".format(corr_num / total_num * 100))
print("F1 score: {:.2f}".format(F1 * 100))
print(test_cf)

flabel1.close()
flabel2.close()
