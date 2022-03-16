import torch
import torch.nn as nn

use_cuda = torch.cuda.is_available()
class CNNClassifier(nn.Module):
    def __init__(self, channel, fs=100, SHHS=False):
        super(CNNClassifier, self).__init__()
        conv1 = nn.Conv2d(1, 16, (1, fs*4), stride=fs//16)
        conv2 = nn.Conv2d(16, 32, (1, fs//4))
        conv3 = nn.Conv2d(32, 64, (1, 3))
        conv4 = nn.Conv2d(64, 64, (1, 3))
        depth_conv = nn.Conv2d(64, 16, (1,1))
        self.conv_module = nn.Sequential(
            conv1, nn.BatchNorm2d(16), nn.ReLU(),
            conv2, nn.BatchNorm2d(32), nn.ReLU(),
            conv3, nn.BatchNorm2d(64), nn.ReLU(),
            conv4, nn.BatchNorm2d(64), nn.ReLU())

        fc = nn.Linear(freq * 40 * 553, 100)
        fc2 = nn.Linear(100, 5)

        self.fc_module = nn.Sequential(fc1, nn.ReLU(), fc2)

        if use_cuda:
            self.conv_module = self.conv_module.cuda()
            self.fc_module = self.fc_module.cuda()

    def forward(self, x, isfc):
        out = self.conv_module(x)
        return out