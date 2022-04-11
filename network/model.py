import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
            conv3, nn.ReLU(inplace=True),
            depth_conv, nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Dropout())
        if fs == 100:
            fc1 = nn.Linear(6800, 1024)
        else:
            fc1 = nn.Linear(7264, 1024)
        fc2 = nn.Linear(1024, 256)
        self.fc_out = nn.Linear(256, n_classes)
        self.fc_module = nn.Sequential(fc1, nn.ReLU(inplace=True), fc2, nn.ReLU(inplace=True), nn.Dropout())

    def forward(self, x, pretrain=False):
        b, c, s, i = x.shape
        x = x.view(b*c, 1, s, i)
        out = self.conv_module(x).permute(0, 2, 1, 3).contiguous().flatten(-2)
        out = self.fc_module(out)
        out = out.view(b, c, s, -1).mean(1)
        if pretrain:
            return self.fc_out(out)
        else:
            return out

class LSTM(nn.Module):
    def __init__(self, input_size=256, hidden_size=256, num_layers=2, drop_rate=0.1, n_classes=5):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=drop_rate)
        self.fc = nn.Linear(hidden_size * self.num_directions, n_classes)

        self.dense = nn.Sequential(nn.Linear(hidden_size * self.num_directions,
                                              hidden_size * self.num_directions), nn.ReLU())
    def forward(self, in_feature, state):
        output, _ = self.lstm(in_feature, state)
        in_feature = in_feature.unsqueeze(-2)
        output = output.unsqueeze(-1)
        dense = F.relu(torch.matmul(output, in_feature))
        dense = self.dense(dense.mean(-1))
        return self.fc(dense)

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).cuda())
        cell = Variable(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, requires_grad=True).cuda())
        state = (hidden, cell)
        return state