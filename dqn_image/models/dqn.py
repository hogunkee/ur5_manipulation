import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QNet(nn.Module):
    def __init__(self, n_actions):
        super(QNet, self).__init__()
        self.cnn1 = nn.Sequential( 
                # 6 x Conv 64,5,1 #
                nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                # Max Pool 3 #
                nn.MaxPool2d(3, stride=3, padding=1),
                )
        self.fc1 = nn.Sequential(
                nn.Linear(6, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                )
        self.conv2 = nn.Sequential(
                # 6 x Conv 64,3,1 #
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                # Max Pool 2 #
                nn.MaxPool2d(2, stride=2, padding=1),
                # 3 x Conv 64,3,1 #
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                )
        self.fc2 = nn.Linear(4096, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, state_im, state_feature):
        x_im = self.cnn1(state_im)
        print('x_im:', x_im.shape)
        x_feature = self.fc1(state_feature)
        print('x_feature:', x_feature.shape)
        x_feature = x_feature.view(-1, 64, 1, 1)
        print('x_feature:', x_feature.shape)
        x_sum = x_im + x_feature
        print('x_sum:', x_sum.shape)
        x = self.conv2(x_sum)
        print('x_conv2:', x.shape)
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        #x = x.view(x.size(0), -1)
        print('x:', x.shape)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)

class QNetOld(nn.Module):
    def __init__(self, n_actions):
        super(QNet, self).__init__()
        self.cnn1 = nn.Sequential( 
                # 1 x Conv 64,6,2 #
                nn.Conv2d(3, 64, kernel_size=6, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                # Max Pool 3 #
                nn.MaxPool2d(3, stride=3, padding=1),
                # 6 x Conv 64,5,1 #
                nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                # Max Pool 3 #
                nn.MaxPool2d(3, stride=3, padding=1),
                )
        self.fc1 = nn.Sequential(
                nn.Linear(6, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                )
        self.conv2 = nn.Sequential(
                # 6 x Conv 64,3,1 #
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                # Max Pool 2 #
                nn.MaxPool2d(2, stride=2, padding=1),
                # 3 x Conv 64,3,1 #
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                )
        self.fc2 = nn.Linear(4096, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, state_im, state_feature):
        x_im = self.cnn1(state_im)
        print('x_im:', x_im.shape)
        x_feature = self.fc1(state_feature)
        print('x_feature:', x_feature.shape)
        x_feature = x_feature.view(-1, 64, 1, 1)
        print('x_feature:', x_feature.shape)
        x_sum = x_im + x_feature
        print('x_sum:', x_sum.shape)
        x = self.conv2(x_sum)
        print('x_conv2:', x.shape)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)

