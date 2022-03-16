import logging
from collections import OrderedDict
import numpy as np
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + '\nTrainable parameters: {}'.format(params)


class RnnModel(BaseModel):
    def __init__(self):
        super().__init__()

        # Assign parameters
        self.filter_base = 4
        self.kernel_size = 3
        self.max_pooling = 2
        self.num_blocks = 7
        self.num_channels = 1
        self.num_classes = 5
        self.rnn_bidirectional = True
        self.rnn_num_layers = 1
        self.rnn_num_units = 1024

        if self.num_channels != 1:
            self.mixing_block = nn.Sequential(OrderedDict([
                ('mix_conv', nn.Conv2d(1, self.num_channels, (self.num_channels, 1))),
                ('mix_batchnorm', nn.BatchNorm2d(self.num_channels)),
                ('mix_relu', nn.ReLU())
            ]))
        self.shortcuts = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('shortcut_conv_{}'.format(k), nn.Conv2d(
                    in_channels=self.num_channels if k == 0 else 4 *
                    self.filter_base * (2 ** (k - 1)),
                    out_channels=4 * self.filter_base * (2 ** k),
                    kernel_size=(1, 1)))
            ])) for k in range(self.num_blocks)
        ])
        self.blocks = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv_{}_1".format(k), nn.Conv2d(
                    in_channels=self.num_channels if k == 0 else 4 * self.filter_base *
                    (2 ** (k - 1)),
                    out_channels=self.filter_base * (2 ** k),
                    kernel_size=(1, 1))),
                ("batchnorm_{}_1".format(k), nn.BatchNorm2d(
                    self.filter_base * (2 ** k))),
                ("relu_{}_1".format(k), nn.ReLU()),
                ("conv_{}_2".format(k), nn.Conv2d(
                    in_channels=self.filter_base * (2 ** k),
                    out_channels=self.filter_base * (2 ** k),
                    kernel_size=(1, self.kernel_size),
                    padding=(0, self.kernel_size // 2))),
                ("batchnorm_{}_2".format(k), nn.BatchNorm2d(
                    self.filter_base * (2 ** k))),
                ("relu_{}_2".format(k), nn.ReLU()),
                ("conv_{}_3".format(k), nn.Conv2d(
                    in_channels=self.filter_base * (2 ** k),
                    out_channels=4 * self.filter_base * (2 ** k),
                    kernel_size=(1, 1))),
                ("batchnorm_{}_3".format(k), nn.BatchNorm2d(
                    4 * self.filter_base * (2 ** k)))
            ])) for k in range(self.num_blocks)
        ])
        self.maxpool = nn.MaxPool2d(kernel_size=(1, self.max_pooling))
        self.relu = nn.ReLU()

        if self.rnn_num_units == 0:
            self.classification = nn.Conv1d(
                in_channels=4 * self.filter_base * (2 ** (self.num_blocks - 1)),
                out_channels=self.num_classes,
                kernel_size=1)
        else:
            self.temporal_block = nn.GRU(
                input_size=4 * self.filter_base * (2 ** (self.num_blocks - 1)),
                hidden_size=self.rnn_num_units, num_layers=self.rnn_num_layers,
                batch_first=True, dropout=0, bidirectional=self.rnn_bidirectional)
            self.temporal_block.flatten_parameters()

            self.classification = nn.Conv1d(
                in_channels=(1 + self.rnn_bidirectional) * self.rnn_num_units,
                out_channels=self.num_classes,
                kernel_size=1)

    def forward(self, x):
        try:
            z = self.mixing_block(x)
        except:
            z = x
        for block, shortcut in zip(self.blocks, self.shortcuts):
            y = shortcut(z)
            z = block(z)
            z += y
            z = self.relu(z)
            # print(z.shape)
            z = self.maxpool(z)

        if self.rnn_num_units == 0:
            z = self.classification(z.squeeze(2))
        else:
            # print(z.shape)
            z = self.temporal_block(z.squeeze(2).transpose(1, 2))
            z = self.classification(z[0].transpose(1, 2))
            z = z[:, :, 0]
        # print(z.shape)
        return z
