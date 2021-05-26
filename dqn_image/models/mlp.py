import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

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

        self.fc1 = nn.Linear(in_channel, 256)
        self.fc2 = nn.Linear(256, 2)
        self.fc3 = nn.Linear(in_channel+2, 256)
        self.fc4 = nn.Linear(256, n_actions)


    def forward(self, x, pose=None):
        x1 = F.relu(self.fc1(x))
        if pose is None:
            pose = self.fc2(x1)

        x2 = torch.cat([x, pose], -1)
        x2 = F.relu(self.fc3(x2))
        q = self.fc4(x2)

        return pose, q
