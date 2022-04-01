import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNBlock(nn.Module):
    def __init__(self, in_ch, hidden_dim=64, bias=False):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Sequential(
                nn.Conv2d(in_ch, hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, 2*hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(2*hidden_dim),
                nn.ReLU(),
                nn.Conv2d(2*hidden_dim, 4*hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(4*hidden_dim),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=3, padding=1),
                )

    def forward(self, x):
        return self.cnn(x)

class Discriminator(nn.Module):
    def __init__(self, num_blocks, n_hidden=8):
        super(Discriminator, self).__init__()
        self.cnn1 = CNNBlock(2, hidden_dim=n_hidden, bias=True)
        self.cnn2 = CNNBlock(4*n_hidden, hidden_dim=4*n_hidden, bias=True)
        self.fc1 = nn.Linear(16*n_hidden, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, sdfs, nsdf):
        # s: n x h x w
        # g: n x h x w
        s, g = sdfs
        sdfs = torch.cat([s.unsqueeze(1), g.unsqueeze(1)], 1)[:nsdf]   # n x 2 x h x w
        N, _, H, W = sdfs.shape
        x_conv1 = self.cnn1(sdfs)
        x_conv2 = self.cnn2(x_conv1)
        x_pool = torch.mean(x_conv2, dim=(2, 3))      # n x cout
        x_fc1 = F.relu(self.fc1(x_pool))
        p = F.sigmoid(self.fc2(x_fc1))
        predict = p.view([N])
        return predict

    '''
    def forward(self, sdfs):
        # s: bs x n x h x w
        # g: bs x n x h x w
        s, g = sdfs
        sdfs = torch.cat([s.unsqueeze(2), g.unsqueeze(2)], 2)   # bs x n x 2 x h x w
        B, N, _, H, W = sdfs.shape
        sdfs_spread = sdfs.reshape([-1, 2, H, W])     # bs*n x 2 x h x w
        x_conv1 = self.cnn1(sdfs_spread)
        x_conv2 = self.cnn2(x_conv1)
        x_pool = torch.mean(x_conv2, dim=(2, 3))      # bs*n x cout
        x_fc1 = F.relu(self.fc1(x_pool))
        p = F.sigmoid(self.fc2(x_fc1))
        predict = p.view([B, N])
        return predict
    '''

