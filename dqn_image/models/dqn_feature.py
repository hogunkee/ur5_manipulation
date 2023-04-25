import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QNet(nn.Module):
    def __init__(self, n_actions, dim_features=6):
        super(QNet, self).__init__()
        self.fc1 = nn.Sequential(
                nn.Linear(dim_features, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                )
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, state_im, state_feature):
        x = self.fc1(state_feature)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)

