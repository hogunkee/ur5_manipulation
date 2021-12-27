import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

#dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
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


class GraphConvolution(nn.Module):
    def __init__(self, in_ch, hidden_dim=64, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_ch = in_ch
        self.out_ch = 4*hidden_dim

        self.conv_root = CNNBlock(in_ch, hidden_dim=hidden_dim, bias=bias)
        self.conv_support = CNNBlock(in_ch, hidden_dim=hidden_dim, bias=bias)

    def forward(self, sdfs, adj_matrix):
        # sdfs: bs x n x c x h x w
        B, N, C, Hin, Win = sdfs.shape
        print('B, N, C, Hin, Win')
        print(B, N, C, Hin, Win)

        sdfs_spread = sdfs.reshape([B*N, C, Hin, Win])    # bs*n x c x h x w
        x_root = self.conv_root(sdfs_spread)            # bs*n x cout x h x w
        x_support = self.conv_support(sdfs_spread)      # bs*n x cout x h x w

        Cout, Hout, Wout = x_root.shape[-3:]
        x_root_flat = x_root.view([B, N, Cout * Hout * Wout])
        x_support_flat = x_support.view([B, N, Cout * Hout * Wout])
        x_neighbor_flat = torch.matmul(adj_matrix, x_support_flat)

        out = x_root_flat + x_neighbor_flat
        out = out.view([B, N, Cout, Hout, Wout])
        return out

    def __repr__(self):
        return self.__class__.__name__ + f'({self.in_ch} -> {self.out_ch})'


class SDFCNNQNetV1(nn.Module):
    def __init__(self, num_blocks, n_actions=8, n_hidden=16, bias=False):
        super(SDFCNNQNetV1, self).__init__()
        self.n_actions = n_actions
        self.num_blocks = num_blocks

        self.cnn1 = CNNBlock(2*num_blocks, hidden_dim=n_hidden, bias=bias)
        self.cnn2 = CNNBlock(4*n_hidden, hidden_dim=4*n_hidden, bias=bias)
        self.fc1 = nn.Linear(16*n_hidden, 256)
        self.fc2 = nn.Linear(256, num_blocks*n_actions)

    def forward(self, sdfs):
        # sdfs: 2 x bs x nb x h x w
        # ( current_sdfs, goal_sdfs )
        s, g = sdfs
        sdfs = torch.cat([s, g], 1)         # bs x 2nb x h x w

        x_conv1 = self.cnn1(sdfs)           # bs x c1 x h x w
        x_conv2 = self.cnn2(x_conv1)        # bs x c2 x h x w
        x_average = torch.mean(x_conv2, dim=(2, 3))         # bs x c2
        x_fc = F.relu(self.fc1(x_average))
        q = self.fc2(x_fc)                                  # bs x nb*na
        Q = q.view([-1, self.num_blocks, self.n_actions])   # bs x nb x na
        return Q


class SDFCNNQNetV2(nn.Module):
    def __init__(self, num_blocks, n_actions=8, n_hidden=16, bias=False):
        super(SDFCNNQNetV2, self).__init__()
        self.n_actions = n_actions
        self.num_blocks = num_blocks

        self.cnn1 = CNNBlock(2, hidden_dim=n_hidden, bias=bias)
        self.cnn2 = CNNBlock(4*n_hidden, hidden_dim=4*n_hidden, bias=bias)
        self.fc1 = nn.Linear(16*n_hidden, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, sdfs):
        # sdfs: 2 x bs x nb x h x w
        # ( current_sdfs, goal_sdfs )
        s, g = sdfs
        sdfs = torch.cat([s, g], 1)         # bs x 2nb x h x w

        B, _, H, W = sdfs.shape
        # sdfs_sg: bs x nb x 2 x h x w
        sdfs_sg = sdfs.view([B, 2, self.num_blocks, H, W]).transpose(1, 2)
        sdfs_spread = sdfs_sg.reshape([self.num_blocks*B, 2, H, W])    # bs*nb x 2 x h x w
        x_conv1 = self.cnn1(sdfs_spread)                    # bs*nb x c1 x h x w
        x_conv2 = self.cnn2(x_conv1)
        x_average = torch.mean(x_conv2, dim=(2, 3))         # bs*nb x c2
        x_fc = F.relu(self.fc1(x_average))
        q = self.fc2(x_fc)                                  # bs*nb x na
        Q = q.view([-1, self.num_blocks, self.n_actions])   # bs x nb x na
        return Q


class SDFGCNQNet(nn.Module):
    def __init__(self, num_blocks, n_actions=8, n_hidden=16):
        super(SDFGCNQNet, self).__init__()
        self.n_actions = n_actions
        self.num_blocks = num_blocks

        self.adj_matrix = self.generate_adj()

        self.gcn1 = GraphConvolution(1, n_hidden, False)
        self.gcn2 = GraphConvolution(4*n_hidden, 4*n_hidden, False)
        self.fc1 = nn.Linear(16*n_hidden, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def generate_adj(self):
        adj_matrix = torch.zeros([self.num_blocks, 2 * self.num_blocks, 2 * self.num_blocks])
        for nb in range(1, self.num_blocks + 1):
            adj_matrix[nb - 1, :nb, :nb] = torch.ones([nb, nb])
            adj_matrix[nb - 1, self.num_blocks:self.num_blocks + nb, :nb] = torch.eye(nb)
            adj_matrix[nb - 1, :nb, self.num_blocks:self.num_blocks + nb] = torch.eye(nb)
            adj_matrix[nb - 1, self.num_blocks:self.num_blocks + nb, self.num_blocks:self.num_blocks + nb] = torch.eye(nb)
        return adj_matrix.to(device)

    def forward(self, sdfs, nsdf):
        # sdfs: 2 x bs x nb x h x w
        # ( current_sdfs, goal_sdfs )
        s, g = sdfs
        sdfs = torch.cat([s, g], 1)         # bs x 2nb x h x w

        B, NS, H, W = sdfs.shape
        adj_matrix = self.adj_matrix[nsdf]

        x_conv1 = self.gcn1(sdfs.unsqueeze(2), adj_matrix)  # bs x 2nb x c x h x w
        x_conv2 = self.gcn2(x_conv1, adj_matrix)            # bs x 2nb x cout x h x w
        x_average = torch.mean(x_conv2, dim=(3, 4))         # bs x 2nb x cout

        # x_current: bs*nb x cout
        x_currents = x_average[:, :self.num_blocks].reshape([B*self.num_blocks, -1])
        x_fc = F.relu(self.fc1(x_currents))
        q = self.fc2(x_fc)                                  # bs*nb x na
        Q = q.view([-1, self.num_blocks, self.n_actions])   # bs x nb x na

        return Q

class SDFGCNQNetV1(nn.Module):
    def __init__(self, num_blocks, n_actions=8, n_hidden=16):
        super(SDFGCNQNetV1, self).__init__()
        self.n_actions = n_actions
        self.num_blocks = num_blocks

        adj1 = torch.cat([torch.ones([num_blocks, num_blocks]), torch.eye(num_blocks)])
        adj2 = torch.cat([torch.eye(num_blocks), torch.eye(num_blocks)])
        self.adj_matrix = torch.cat([adj1, adj2], 1).to(device)

        self.gcn1 = GraphConvolution(1, n_hidden, False)
        self.gcn2 = GraphConvolution(4*n_hidden, 4*n_hidden, False)
        self.fc1 = nn.Linear(16*n_hidden, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, sdfs):
        # sdfs: 2 x bs x nb x h x w
        # ( current_sdfs, goal_sdfs )
        s, g = sdfs
        sdfs = torch.cat([s, g], 1)         # bs x 2nb x h x w

        B, NS, H, W = sdfs.shape
        if NS!=2*self.num_blocks:
            num_blocks = NS//2
            adj1 = torch.cat([torch.ones([num_blocks, num_blocks]), torch.eye(num_blocks)])
            adj2 = torch.cat([torch.eye(num_blocks), torch.eye(num_blocks)])
            adj_matrix = torch.cat([adj1, adj2], 1).to(device)
        else:
            num_blocks = self.num_blocks
            adj_matrix = self.adj_matrix

        x_conv1 = self.gcn1(sdfs.unsqueeze(2), adj_matrix)  # bs x 2nb x c x h x w
        x_conv2 = self.gcn2(x_conv1, adj_matrix)            # bs x 2nb x cout x h x w
        x_average = torch.mean(x_conv2, dim=(3, 4))         # bs x 2nb x cout

        # x_current: bs*nb x cout
        x_currents = x_average[:, :num_blocks].reshape([B*num_blocks, -1])
        x_fc = F.relu(self.fc1(x_currents))
        q = self.fc2(x_fc)                                  # bs*nb x na
        Q = q.view([-1, num_blocks, self.n_actions])   # bs x nb x na

        return Q


class SDFGCNQNetV2(nn.Module):
    def __init__(self, num_blocks, n_actions=8, n_hidden=64):
        super(SDFGCNQNetV2, self).__init__()
        self.n_actions = n_actions
        self.num_blocks = num_blocks
        self.adj_matrix = torch.ones([num_blocks, num_blocks]).to(device)

        self.gcn1 = GraphConvolution(2, n_hidden, False)
        self.gcn2 = GraphConvolution(4*n_hidden, 4*n_hidden, False)
        self.fc1 = nn.Linear(16*n_hidden, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, sdfs):
        # sdfs: 2 x bs x nb x h x w
        # ( current_sdfs, goal_sdfs )
        s, g = sdfs
        sdfs = torch.cat([s, g], 1)         # bs x 2nb x h x w

        B, NS, H, W = sdfs.shape
        if NS!=2*self.num_blocks:
            num_blocks = NS//2
            adj_matrix = torch.ones([num_blocks, num_blocks]).to(device)
        else:
            num_blocks = self.num_blocks
            adj_matrix = self.adj_matrix

        # sdfs_sg: bs x nb x 2 x h x w
        sdfs_sg = sdfs.view([B, 2, num_blocks, H, W]).transpose(1, 2)
        x_conv1 = self.gcn1(sdfs_sg, adj_matrix)                # bs x nb x c x h x w
        x_conv2 = self.gcn2(x_conv1, adj_matrix)                # bs x nb x cout x h x w
        x_average = torch.mean(x_conv2, dim=(3, 4))             # bs x nb x c

        x_currents = x_average.reshape([B*num_blocks, -1]) # bs*nb x c
        x_fc = F.relu(self.fc1(x_currents))
        q = self.fc2(x_fc)
        Q = q.view([-1, num_blocks, self.n_actions])       # bs x nb x na
        return Q

