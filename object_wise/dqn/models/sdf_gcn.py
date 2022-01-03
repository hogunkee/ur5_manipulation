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


class GraphConvolutionSeparateEdge(nn.Module):
    def __init__(self, in_ch, hidden_dim=64, bias=False):
        super(GraphConvolutionSeparateEdge, self).__init__()
        self.in_ch = in_ch
        self.out_ch = 4*hidden_dim

        self.conv_root = CNNBlock(in_ch, hidden_dim=hidden_dim, bias=bias)
        self.conv_block_edge = CNNBlock(in_ch, hidden_dim=hidden_dim, bias=bias)
        self.conv_goal_edge = CNNBlock(in_ch, hidden_dim=hidden_dim, bias=bias)

    def forward(self, sdfs, adj_matrix):
        # sdfs: bs x n x c x h x w
        B, N, C, Hin, Win = sdfs.shape
        '''
        NB = N//2
        adj1 = torch.zeros_like(adj_matrix)
        adj1[:, :NB, :NB] = adj_matrix[:, :NB, :NB]
        adj2 = torch.zeros_like(adj_matrix)
        adj2[:, NB:, NB:] = adj_matrix[:, NB:, NB:]
        '''

        sdfs_block = sdfs[:, :N//2]
        sdfs_block = sdfs_block.reshape([B*N//2, C, Hin, Win])
        sdfs_goal = sdfs[:, N//2:]
        sdfs_goal = sdfs_goal.reshape([B*N//2, C, Hin, Win])

        sdfs_spread = sdfs.reshape([B*N, C, Hin, Win])     # bs*n x c x h x w
        x_root = self.conv_root(sdfs_spread)               # bs*n x cout x h x w
        x_block_edge = self.conv_block_edge(sdfs_block)    # bs*n/2 x cout x h x w
        x_goal_edge = self.conv_block_edge(sdfs_goal)      # bs*n/2 x cout x h x w

        Cout, Hout, Wout = x_root.shape[-3:]
        x_root_flat = x_root.view([B, N, Cout * Hout * Wout])
        x_block_edge_flat = x_block_edge.view([B, N//2, Cout * Hout * Wout])
        x_goal_edge_flat = x_goal_edge.view([B, N//2, Cout * Hout * Wout])
        x_support_flat = torch.cat([x_block_edge_flat, x_goal_edge_flat], 1)
        x_neighbor_flat = torch.matmul(adj_matrix, x_support_flat)

        out = x_root_flat + x_neighbor_flat
        out = out.view([B, N, Cout, Hout, Wout])
        return out

    def __repr__(self):
        return self.__class__.__name__ + f'({self.in_ch} -> {self.out_ch})'


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


class SDFGCNQNetV2(nn.Module):
    # version 2: graph with different edges
    def __init__(self, num_blocks, n_actions=8, n_hidden=16):
        super(SDFGCNQNetV2, self).__init__()
        self.n_actions = n_actions
        self.num_blocks = num_blocks

        self.adj_matrix = self.generate_adj()

        self.gcn1 = GraphConvolutionSeparateEdge(1, n_hidden, False)
        self.gcn2 = GraphConvolutionSeparateEdge(4*n_hidden, 4*n_hidden, False)
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


class SDFGCNQNetV3(nn.Module):
    def __init__(self, num_blocks, n_actions=8, n_hidden=16):
        super(SDFGCNQNetV3, self).__init__()
        self.n_actions = n_actions
        self.num_blocks = num_blocks

        self.adj_matrix = self.generate_adj()

        self.gcn1 = GraphConvolution(2, n_hidden, False)
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

        block_flags = torch.zeros_like(sdfs)
        block_flags[:, :NS//2] = 1.0        # blocks as 1, goals as 0

        adj_matrix = self.adj_matrix[nsdf]

        sdfs_concat = torch.cat([sdfs.unsqueeze(2), block_flags.unsqueeze(2)], 2)   # bs x 2nb x 2 x h x w
        x_conv1 = self.gcn1(sdfs_concat, adj_matrix)        # bs x 2nb x c x h x w
        x_conv2 = self.gcn2(x_conv1, adj_matrix)            # bs x 2nb x cout x h x w
        x_average = torch.mean(x_conv2, dim=(3, 4))         # bs x 2nb x cout

        # x_current: bs*nb x cout
        x_currents = x_average[:, :self.num_blocks].reshape([B*self.num_blocks, -1])
        x_fc = F.relu(self.fc1(x_currents))
        q = self.fc2(x_fc)                                  # bs*nb x na
        Q = q.view([-1, self.num_blocks, self.n_actions])   # bs x nb x na

        return Q


