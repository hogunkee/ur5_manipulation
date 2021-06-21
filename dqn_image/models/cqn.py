import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

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
    def __init__(self, n_actions, in_channel):
        super(FC_QNet, self).__init__()
        self.n_actions = n_actions

        # CNN layers
        self.cnn1 = nn.Sequential(
                nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
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
        self.cnn2 = nn.Sequential(
                nn.Conv2d(in_channel+8, 64, kernel_size=3, stride=1, padding=1),
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
        self.fully_conv1 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=1),
                nn.ReLU(),
                nn.Dropout2d(),
                nn.Conv2d(512, 512, kernel_size=1),
                nn.ReLU(),
                nn.Dropout2d(),
                nn.Conv2d(512, 1, kernel_size=1),
                )
        self.fully_conv2 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=1),
                nn.ReLU(),
                nn.Dropout2d(),
                nn.Conv2d(512, 512, kernel_size=1),
                nn.ReLU(),
                nn.Dropout2d(),
                nn.Conv2d(512, 1, kernel_size=1),
                )

        # self.upscore = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        self.upscore1 = nn.Sequential(
            nn.ConvTranspose2d(1, 1, 3, stride=2, bias=False, padding=1, output_padding=1),
            nn.ConvTranspose2d(1, 1, 3, stride=2, bias=False, padding=1, output_padding=1),
            nn.ConvTranspose2d(1, 1, 3, stride=2, bias=False, padding=1, output_padding=1),
        )
        self.upscore2 = nn.Sequential(
            nn.ConvTranspose2d(1, 1, 3, stride=2, bias=False, padding=1, output_padding=1),
            nn.ConvTranspose2d(1, 1, 3, stride=2, bias=False, padding=1, output_padding=1),
            nn.ConvTranspose2d(1, 1, 3, stride=2, bias=False, padding=1, output_padding=1),
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


    def forward(self, x):
        # torch.cuda.empty_cache()
        x_pad = F.pad(x, (20, 20, 20, 20), mode='constant')
        # x = F.pad(x, (20, 20, 20, 20), mode='reflect')
        output_prob = []
        for r_idx in range(self.n_actions):
            theta = r_idx * (2*np.pi / self.n_actions)

            affine_mat_before = np.asarray([
                [np.cos(-theta), np.sin(-theta), 0],
                [-np.sin(-theta), np.cos(-theta), 0]])
            affine_mat_before.shape = (2, 3, 1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
            affine_mat_before = affine_mat_before.repeat(x.size()[0], 1, 1)
            flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).type(dtype), x_pad.size())
            x_rotate = F.grid_sample(x_pad, flow_grid_before, mode='nearest')
            #x_rotate = F.grid_sample(Variable(x, volatile=True).cuda(), flow_grid_before, mode='nearest')

            h = self.cnn1(x_rotate)
            h = self.fully_conv1(h)
            h = self.upscore1(h)

            affine_mat_after = np.asarray([
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0]
                ])
            affine_mat_after.shape = (2, 3, 1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
            affine_mat_after = affine_mat_after.repeat(x.size()[0], 1, 1)
            flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).type(dtype), h.size())

            h_after = F.grid_sample(h, flow_grid_after, mode='nearest')
            h_after = h_after[:, :, 20:20 + x.size()[2], 20:20 + x.size()[3]].contiguous()
            output_prob.append(h_after)
        q1 = torch.cat(output_prob, 1)

        q1_pad = F.pad(q1, (20, 20, 20, 20), mode='constant')
        x2_pad = torch.cat([x_pad, q1_pad], 1)
        print(f'x2: {x2.shape}')

        output_prob = []
        for r_idx in range(self.n_actions):
            theta = r_idx * (2*np.pi / self.n_actions)

            affine_mat_before = np.asarray([
                [np.cos(-theta), np.sin(-theta), 0],
                [-np.sin(-theta), np.cos(-theta), 0]])
            affine_mat_before.shape = (2, 3, 1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
            affine_mat_before = affine_mat_before.repeat(x.size()[0], 1, 1)
            flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).type(dtype), x2_pad.size())
            x_rotate = F.grid_sample(x2_pad, flow_grid_before, mode='nearest')
            #x_rotate = F.grid_sample(Variable(x, volatile=True).cuda(), flow_grid_before, mode='nearest')

            h = self.cnn2(x_rotate)
            h = self.fully_conv2(h)
            h = self.upscore2(h)

            affine_mat_after = np.asarray([
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0]
                ])
            affine_mat_after.shape = (2, 3, 1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
            affine_mat_after = affine_mat_after.repeat(x.size()[0], 1, 1)
            flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).type(dtype), h.size())

            h_after = F.grid_sample(h, flow_grid_after, mode='nearest')
            h_after = h_after[:, :, 20:20 + x.size()[2], 20:20 + x.size()[3]].contiguous()
            output_prob.append(h_after)
        q2 = torch.cat(output_prob, 1) # bs x 8 x 96 x 96

        return torch.cat([q1.unsqueeze(1), q2.unsqueeze(1)], 1) # bs x 2(nb) x 8 x 96 x 96


class FC_QNet_half(nn.Module):
    def __init__(self, n_actions, in_channel):
        super(FC_QNet_half, self).__init__()
        self.n_actions = n_actions

        # CNN layers
        self.cnn1 = nn.Sequential(
                nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2, ceil_mode=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
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
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.cnn2 = nn.Sequential(
                nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2, ceil_mode=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
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
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        # FC layers
        self.fully_conv1 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=1),
                nn.ReLU(),
                nn.Dropout2d(),
                nn.Conv2d(256, 256, kernel_size=1),
                nn.ReLU(),
                nn.Dropout2d(),
                nn.Conv2d(256, 1, kernel_size=1),
                )
        self.fully_conv2 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=1),
                nn.ReLU(),
                nn.Dropout2d(),
                nn.Conv2d(256, 256, kernel_size=1),
                nn.ReLU(),
                nn.Dropout2d(),
                nn.Conv2d(256, 1, kernel_size=1),
                )

        # self.upscore = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        self.upscore1 = nn.Sequential(
            nn.ConvTranspose2d(1, 1, 3, stride=2, bias=False, padding=1, output_padding=1),
            nn.ConvTranspose2d(1, 1, 3, stride=2, bias=False, padding=1, output_padding=1),
            nn.ConvTranspose2d(1, 1, 3, stride=2, bias=False, padding=1, output_padding=1),
        )
        self.upscore2 = nn.Sequential(
            nn.ConvTranspose2d(1, 1, 3, stride=2, bias=False, padding=1, output_padding=1),
            nn.ConvTranspose2d(1, 1, 3, stride=2, bias=False, padding=1, output_padding=1),
            nn.ConvTranspose2d(1, 1, 3, stride=2, bias=False, padding=1, output_padding=1),
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


    def forward(self, x):
        # torch.cuda.empty_cache()
        x_pad = F.pad(x, (20, 20, 20, 20), mode='constant')
        # x = F.pad(x, (20, 20, 20, 20), mode='reflect')
        output_prob = []
        for r_idx in range(self.n_actions):
            theta = r_idx * (2*np.pi / self.n_actions)

            affine_mat_before = np.asarray([
                [np.cos(-theta), np.sin(-theta), 0],
                [-np.sin(-theta), np.cos(-theta), 0]])
            affine_mat_before.shape = (2, 3, 1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
            affine_mat_before = affine_mat_before.repeat(x.size()[0], 1, 1)
            flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).type(dtype), x_pad.size())
            x_rotate = F.grid_sample(x_pad, flow_grid_before, mode='nearest')
            #x_rotate = F.grid_sample(Variable(x, volatile=True).cuda(), flow_grid_before, mode='nearest')

            h = self.cnn1(x_rotate)
            h = self.fully_conv1(h)
            h = self.upscore1(h)

            affine_mat_after = np.asarray([
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0]
                ])
            affine_mat_after.shape = (2, 3, 1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
            affine_mat_after = affine_mat_after.repeat(x.size()[0], 1, 1)
            flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).type(dtype), h.size())

            h_after = F.grid_sample(h, flow_grid_after, mode='nearest')
            h_after = h_after[:, :, 20:20 + x.size()[2], 20:20 + x.size()[3]].contiguous()
            output_prob.append(h_after)
        q1 = torch.cat(output_prob, 1)

        q1_pad = F.pad(q1, (20, 20, 20, 20), mode='constant')
        x2_pad = torch.cat([x_pad, q1_pad], 1)
        print(f'x2: {x2.shape}')

        output_prob = []
        for r_idx in range(self.n_actions):
            theta = r_idx * (2*np.pi / self.n_actions)

            affine_mat_before = np.asarray([
                [np.cos(-theta), np.sin(-theta), 0],
                [-np.sin(-theta), np.cos(-theta), 0]])
            affine_mat_before.shape = (2, 3, 1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
            affine_mat_before = affine_mat_before.repeat(x.size()[0], 1, 1)
            flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).type(dtype), x2_pad.size())
            x_rotate = F.grid_sample(x2_pad, flow_grid_before, mode='nearest')
            #x_rotate = F.grid_sample(Variable(x, volatile=True).cuda(), flow_grid_before, mode='nearest')

            h = self.cnn2(x_rotate)
            h = self.fully_conv2(h)
            h = self.upscore2(h)

            affine_mat_after = np.asarray([
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0]
                ])
            affine_mat_after.shape = (2, 3, 1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
            affine_mat_after = affine_mat_after.repeat(x.size()[0], 1, 1)
            flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).type(dtype), h.size())

            h_after = F.grid_sample(h, flow_grid_after, mode='nearest')
            h_after = h_after[:, :, 20:20 + x.size()[2], 20:20 + x.size()[3]].contiguous()
            output_prob.append(h_after)
        q2 = torch.cat(output_prob, 1) # bs x 8 x 96 x 96

        return torch.cat([q1.unsqueeze(1), q2.unsqueeze(1)], 1) # bs x 2(nb) x 8 x 96 x 96
