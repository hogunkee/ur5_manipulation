import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

dtype = torch.FloatTensor #torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class QNet(nn.Module):
    def __init__(self, n_actions, in_channel):
        super(QNet, self).__init__()
        self.n_actions = n_actions

        # FC layers
        self.mlp = nn.Sequential(
                nn.Linear(in_channel, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                )

        self.fc1 = nn.Linear(64, 2)
        self.fc2 = nn.Linear(64, n_actions)


    def forward(self, x):
        x = self.mlp(x)
        x1 = self.fc1(x)
        x2 = F.softmax(self.fc2(x))
        return torch.cat([x1, x2], -1)
