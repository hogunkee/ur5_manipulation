import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

#dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNBlock(nn.Module):
    def __init__(self, in_ch, hidden_dim=64, k=3, bias=False):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Sequential(
                nn.Conv2d(in_ch, hidden_dim, kernel_size=k, stride=1, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=k, stride=1, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=k, stride=1, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=k, stride=1, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=k, stride=1, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=k, stride=1, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                )

    def forward(self, x):
        return self.cnn(x)


class GraphConvolution(nn.Module):
    def __init__(self, in_ch, hidden_dim=64, kernel_size=3, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_ch = in_ch
        self.out_ch = hidden_dim

        self.conv_root = CNNBlock(in_ch, hidden_dim=hidden_dim, k=kernel_size, bias=bias)
        self.conv_support = CNNBlock(in_ch, hidden_dim=hidden_dim, k=kernel_size, bias=bias)

    def forward(self, sdfs, adj_matrix, N):
        # sdfs: bs*n x c x h x w
        x_root = self.conv_root(sdfs)            # bs*n x cout x h x w
        x_support = self.conv_support(sdfs)      # bs*n x cout x h x w

        Cout, Hout, Wout = x_root.shape[-3:]
        x_root_flat = x_root.view([-1, N, Cout * Hout * Wout])
        x_support_flat = x_support.view([-1, N, Cout * Hout * Wout])
        x_neighbor_flat = torch.matmul(adj_matrix, x_support_flat)

        out = x_root_flat + x_neighbor_flat
        out = out.view([-1, Cout, Hout, Wout])
        return out

    def __repr__(self):
        return self.__class__.__name__ + f'({self.in_ch} -> {self.out_ch})'


class TrackQNetV3(nn.Module):
    def __init__(self, num_blocks, n_actions=8, n_hidden=64, normalize=False, resize=True):
        super(TrackQNetV3, self).__init__()
        self.n_actions = n_actions
        self.num_blocks = num_blocks
        self.normalize = normalize
        self.n_hidden = n_hidden

        self.ws_mask = self.generate_wsmask()
        self.adj_matrix = self.generate_adj()

        self.conv1 = nn.Sequential(
                nn.Conv2d(3, n_hidden, kernel_size=6, stride=2, padding=2),
                nn.BatchNorm2d(n_hidden),
                nn.ReLU(),
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(n_hidden),
                nn.ReLU(),
                nn.Conv2d(n_hidden, n_hidden, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(n_hidden),
                nn.ReLU(),
                )
        self.gcn1 = GraphConvolution(n_hidden, n_hidden, kernel_size=5, bias=False)
        self.gcn2 = GraphConvolution(n_hidden, n_hidden, kernel_size=3, bias=False)
        self.pool1 = nn.MaxPool2d(3, stride=3, padding=1)
        self.pool2 = nn.MaxPool2d(3, stride=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, padding=1)
        self.fc1 = nn.Linear(n_hidden, 64)
        self.fc2 = nn.Linear(64, n_actions)

    def generate_wsmask(self):
        mask = np.load('../../ur5_mujoco/workspace_mask_480.npy').astype(float)
        return mask

    def generate_adj(self):
        NB = self.num_blocks
        adj_matrix = torch.zeros([NB, 2 * NB, 2 * NB])
        for nb in range(1, NB + 1):
            adj_matrix[nb - 1, :nb, :nb] = torch.ones([nb, nb])
            adj_matrix[nb - 1, NB:NB + nb, :nb] = torch.eye(nb)
            adj_matrix[nb - 1, :nb, NB:NB + nb] = torch.eye(nb)
            adj_matrix[nb - 1, NB:NB + nb, NB:NB + nb] = torch.eye(nb)
            if self.normalize:
                diag = [1/np.sqrt(nb+1)] * nb
                diag += [0] * (NB - nb)
                diag += [1/np.sqrt(2)] * nb
                diag += [0] * (NB - nb)
                d_mat = torch.Tensor(np.diag(diag))
                adj_matrix[nb-1] = torch.matmul(torch.matmul(d_mat, adj_matrix[nb-1]), d_mat)
        return adj_matrix.to(device)

    def forward(self, sdfs, nsdf):
        # sdfs: 2 x bs x nb x h x w
        # ( current_sdfs, goal_sdfs )
        s, g = sdfs
        sdfs = torch.cat([s, g], 1)         # bs x 2nb x h x w
        B, NS, H, W = sdfs.shape

        ## block flag ##
        block_flags = torch.zeros_like(sdfs)
        block_flags[:, :NS//2] = 1.0        # blocks as 1, goals as 0

        ## workspace mask ##
        ws_masks = torch.zeros_like(sdfs)
        ws_masks[:, :] = torch.Tensor(self.ws_mask)

        ## adjacency matrix ##
        adj_matrix = self.adj_matrix[nsdf]

        # bs x 2nb x 3 x h x w
        sdfs_concat = torch.cat([ sdfs.unsqueeze(2), 
                                  block_flags.unsqueeze(2),
                                  ws_masks.unsqueeze(2)
                                 ], 2)   
        sdfs_spread = sdfs_concat.view([-1, 3, H, W])
        x_conv1 = self.conv1(sdfs_spread)
        x_pool1 = self.pool1(x_conv1)
        x_gcn1 = self.gcn1(x_pool1, adj_matrix, NS)
        x_pool2 = self.pool2(x_gcn1)
        x_gcn2 = self.gcn2(x_pool2, adj_matrix, NS)
        x_pool3 = self.pool3(x_gcn2)
        x_conv2 = self.conv2(x_pool3)

        x_average = torch.mean(x_conv2, dim=(2, 3)).view([B, NS, self.n_hidden]) # bs x 2nb x cout
        x_currents = x_average[:, :self.num_blocks].reshape([B*self.num_blocks, self.n_hidden])
        x_fc1 = F.relu(self.fc1(x_currents))
        q = self.fc2(x_fc1)
        Q = q.view([B, self.num_blocks, self.n_actions])    # bs x nb x na
        return Q


