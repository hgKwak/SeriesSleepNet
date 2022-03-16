import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
class CNN(nn.Module):
    def __init__(self, fs=100, n_classes=5):
        super(CNN, self).__init__()
        conv1 = nn.Conv2d(1, 16, (1, fs*4), stride=(1, fs//16))
        conv2 = nn.Conv2d(16, 32, (1, fs//12))
        conv3 = nn.Conv2d(32, 64, (1, 3))
        depth_conv = nn.Conv2d(64, 16, 1)

        self.conv_module = nn.Sequential(
            conv1, nn.ReLU(inplace=True),
            conv2, nn.ReLU(inplace=True),
            conv3, nn.ReLU(inplace=True), nn.Dropout(),
            depth_conv, nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Dropout())
        if fs == 100:
            fc1 = nn.Linear(6800, 1024)
        else:
            fc1 = nn.Linear(7264, 1024)
        fc2 = nn.Linear(1024, 256)
        self.fc_out = nn.Linear(256, n_classes)
        self.fc_module = nn.Sequential(fc1, nn.ReLU(inplace=True), fc2, nn.ReLU(inplace=True), nn.Dropout())
        self.ln = nn.LayerNorm(256)
        # self.lstm_module = LSTM(input_size=256, hidden_size=256, num_layers=2)

    def forward(self, x, pretrain=False):
        b, c, s, i = x.shape
        x = x.view(b*c, 1, s, i)
        out = self.conv_module(x).permute(0, 2, 1, 3).contiguous().flatten(-2)
        # out = out.view(b, c, s, -1).mean(1)

        out = self.fc_module(out)
        out = out.view(b, c, s, -1).mean(1)
        # out = self.ln(out.view(b, c, s, -1)).mean(1)
        if pretrain:
            return self.fc_out(out)
        else:
            # print(out.shape)
            return out
        # b, s, _ = out.shape
        # print(out.shape)
        # out = self.lstm_module(out)
        # print(out.shape)
        # return out
'''
class CNN(nn.Module):
    def __init__(self, fs=100):
        super(CNN, self).__init__()
        conv1 = nn.Conv2d(1, 10, (1, fs*2))
        conv2 = nn.Conv2d(10, 20, (1, 32))
        conv3 = nn.Conv2d(20, 30, (1, 128))
        conv4 = nn.Conv2d(30, 40, (1, 512))

        self.conv_module = nn.Sequential(
            conv1, nn.MaxPool2d((1,2)), nn.ReLU(inplace=True),
            conv2, nn.ReLU(inplace=True), conv3, nn.ReLU(inplace=True),
            conv4, nn.MaxPool2d((1,2)), nn.ReLU(inplace=True))
        fc1 = nn.Linear(365*40, 100)
        fc2 = nn.Linear(100, 5)
        self.fc_out = nn.Linear(256, 5)
        self.fc_module = nn.Sequential(fc1, nn.ReLU(inplace=True), fc2)
        # self.lstm_module = LSTM(input_size=256, hidden_size=256, num_layers=2)

    def forward(self, x, pretrain=False):
        out = self.conv_module(x).permute(0, 2, 1, 3).flatten(-2)
        # print(out.shape)
        out = self.fc_module(out)
        if pretrain:
            return F.softmax(out, dim=-1)
        else:
            return out
'''
class LSTM(nn.Module):
    def __init__(self, input_size=256, hidden_size=256, num_layers=2, drop_rate=0.1, n_classes=5):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=drop_rate)
        self.dense = nn.Sequential(nn.Linear(input_size, hidden_size * self.num_directions), nn.ReLU(inplace=True),
                                   nn.Linear(hidden_size * self.num_directions, hidden_size * self.num_directions))
        # self.fc = nn.Linear(hidden_size * self.num_directions + input_size, 5)
        # self.dense1 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(inplace=True))
        self.dense_cnn = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU(inplace=True), nn.Dropout(drop_rate))
        self.dense_lstm = nn.Sequential(nn.Linear(hidden_size * self.num_directions, hidden_size * self.num_directions),
                                    nn.ReLU(inplace=True), nn.Dropout(drop_rate))
        self.dense1 = nn.Sequential(nn.Linear(hidden_size, 1), nn.ReLU(inplace=True), nn.Dropout(drop_rate))
        self.dense2 = nn.Sequential(nn.Linear(hidden_size * self.num_directions,
                                              hidden_size), nn.LayerNorm(hidden_size),
                                              nn.ReLU(inplace=True), nn.Dropout(drop_rate),
                                              nn.Linear(hidden_size, hidden_size * self.num_directions))

        # self.dense1 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(inplace=True))
        # self.dense2 = nn.Sequential(nn.Linear(hidden_size * self.num_directions, hidden_size), nn.ReLU(inplace=True))

        self.ln = nn.LayerNorm(hidden_size * self.num_directions)
        self.fc = nn.Linear(hidden_size * self.num_directions + hidden_size, n_classes)
        self.fc2 = nn.Linear(hidden_size * self.num_directions, n_classes)

        self.in_emb = nn.Linear(input_size, input_size)
        self.out_emb = nn.Linear(hidden_size * self.num_directions, hidden_size * self.num_directions)
        self.out_fc = nn.Sequential(nn.Linear(input_size, input_size//2),
                                    nn.ReLU(), nn.Linear(input_size//2, n_classes))

        self.dense1 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.dense2 = nn.Sequential(nn.Linear(hidden_size * self.num_directions,
                                              hidden_size * self.num_directions), nn.ReLU())
    def forward(self, in_feature, state, dense_connect=True):
        # in_feature = nn.utils.rnn.pack_padded_sequence(in_feature, lens, enforce_sorted=False)
        output, _ = self.lstm(in_feature, state)
        # self.hidden, self.cell = .detach(), cell
        # print(self.cell.grad)
        # output, _ = nn.utils.rnn.pad_sequence(packed_output)
        # print(output.shape)

        if dense_connect:
            # '''
            # dense = torch.cat([in_feature, output], dim=-1)
            in_feature = in_feature.unsqueeze(-2)
            output = output.unsqueeze(-1)
            dense = F.relu(torch.matmul(output, in_feature))
            dense = self.dense2(dense.mean(-1))
            # print(dense.shape)
            # dense = F.relu(self.dense2(dense).mean(-1))
            # print(dense.shape)
            # output = F.relu(dense.mean(-1))
            # output = self.dense2(output)
            return self.fc2(dense)
            #''' #matmul dense

            # '''
            # in_feature = self.dense1(in_feature)
            # output = self.dense2(output)
            # output = torch.matmul(in_feature, output.transpose(-1, -2))
            # output = torch.cat([in_feature, output], dim=-1)
            # return self.fc(output)
            #''' #concat dense

            # print(in_feature.shape, output.shape)
            # dense_feature = self.dense1(in_feature)
            # output = self.dense2(output)

            # in_feature = self.in_emb(in_feature)
            # output = self.out_emb(output)
            # dense = torch.matmul(output, in_feature.transpose(-1, -2))
            # dense = F.softmax(dense, dim=-1)
            # dense = torch.matmul(dense, output)
            # return self.out_fc(dense)


            # output = torch.cat([dense_feature, output], dim=-1)

            # print(output.shape, dense_feature.transpose(-1, -2).shape)
            # dense_feature = F.softmax(torch.matmul(output, dense_feature.transpose(-1, -2)), dim=-1)
            # print(dense_feature.shape)

            # output = self.ln(torch.matmul(dense_feature, output))
            # output = torch.cat([self.dense2(output), self.dense1(in_feature)], -1)
        else:
            return self.fc2(output)

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).cuda())
        cell = Variable(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, requires_grad=True).cuda())
        state = (hidden, cell)
        return state