import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class FC_QNet(nn.Module):
    def __init__(self, n_actions, task):
        super(FC_QNet, self).__init__()
        self.n_actions = n_actions

        # CNN layers
        self.cnn = nn.Sequential(
                nn.Conv2d(3*(task+1), 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2, ceil_mode=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2, ceil_mode=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        # FC layers
        self.fully_conv = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=1),
                nn.ReLU(),
                nn.Dropout2d(),
                nn.Conv2d(512, 512, kernel_size=1),
                nn.ReLU(),
                nn.Dropout2d(),
                nn.Conv2d(512, 8, kernel_size=1),
                )

        # self.upscore = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        self.upscore = nn.Sequential(
            nn.ConvTranspose2d(8, 8, 3, stride=2, bias=False, padding=1, output_padding=1),
            nn.ConvTranspose2d(8, 8, 3, stride=2, bias=False, padding=1, output_padding=1),
            nn.ConvTranspose2d(8, 8, 3, stride=2, bias=False, padding=1, output_padding=1),
        )

        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     m.weight.data.zero_()
            #     if m.bias is not None:
            #         m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)


    def forward(self, x, is_volatile=False, rotation=-1, debug=False):
        x_pad = F.pad(x, (20, 20, 20, 20), mode='constant')
        h = self.cnn(x_pad)
        h = self.fully_conv(h)
        h = self.upscore(h)
        h = h[:, :, 20:20 + x.size()[2], 20:20 + x.size()[3]]

        return h

