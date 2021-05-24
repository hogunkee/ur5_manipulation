import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

dtype = torch.FloatTensor #torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class QNet(nn.Module):
    def __init__(self, n_actions, in_channel):
        super(FC_QNet, self).__init__()
        self.n_actions = n_actions

        # FC layers
        self.mlp = nn.Sequential(
                Dense(64, input_dim=in_channel, kernel_initializer='normal', activation='relu'),
                Dense(64, input_dim=64, kernel_initializer='normal', activation='relu'),
                )

        self.fc1 = Dense(2, input_dim=64, kernel_initializer='normal')
        self.fc2 = Dense(n_actions, input_dim=64, kernel_initializer='normal', activation='softmax')


    def forward(self, x):
        x = self.mlp(x)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        return torch.cat([x1, x2])
