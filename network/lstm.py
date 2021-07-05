import torch
import torch.nn as nn
use_cuda = torch.cuda.is_available()

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size * self.num_directions + 5, 5)
        self.linear2 = nn.Linear(hidden_size * self.num_directions, 5)

        if use_cuda:
            self.lstm = self.lstm.cuda()
            self.linear = self.linear.cuda()
            self.linear2 = self.linear2.cuda()

    def forward(self, x, hidden, cell, seq_len):
        x = x.view(-1, seq_len, self.input_size)
        out, (hidden, cell) = self.lstm(x, (hidden, cell))
        out = out.squeeze()
        try:
            output = torch.cat((out, x.squeeze()), 2)
        except:
            output = torch.cat((out, x.squeeze()), 1)
        output = self.linear(output)
        try:
            return output.view(output.shape[1]*output.shape[0], output.shape[2])
        except:
            return output.view(output.shape[0], output.shape[1])

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
        cell = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)

        if use_cuda:
            hidden = hidden.cuda()
            cell = cell.cuda()

        return hidden, cell