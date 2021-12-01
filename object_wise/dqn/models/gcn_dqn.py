import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class GraphConvolution(nn.Module):
    def __init__(self, in_ch, out_ch, adj_matrix, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.adj_matrix = adj_matrix

        self.conv_root = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv_support = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, sdfs):
        # sdfs: bs x n x c x h x w
        B, N, C, Hin, Win = sdfs.shape
        root_tensors = []
        support_tensors = []
        for n in range(N):
            sdf = sdfs[:, n]
            x_root = self.conv_root(sdf)       # bs x cout x h x w
            x_support = self.conv_support(sdf)
            x_root.unsqueeze(1)        # bs x 1 x cout x h x w
            x_support.unsqueeze(1)     # bs x 1 x cout x h x w
            root_tensors.append(x_root)
            support_tensors.append(x_support)

        root = torch.cat(root_tensors, axis=1)       # bs x n x cout x hout x wout
        support = torch.cat(support_tensors, axis=1) # bs x n x cout x hout x wout

        Cout, Hout, Wout = root.shape[-2:]
        root_flat = root.view([B, N, Cout * Hout * Wout])
        support_flat = support.view([B, N, Cout * Hout * Wout])
        neighbor_flat = torch.matmul(self.adj_matrix, support_flat)

        out = root_flat + neighbor_flat
        out = out.view([B, N, Cout, Hout, Wout])
        return out

    def __repr__(self):
        return self.__class__.__name__ + f'({self.in_ch} -> {self.out_ch})'


class SDfQNet(nn.Module):
    def __init__(self, n_actions, num_blocks, n_hidden=64):
        super(SDFQNet, self).__init__()
        self.n_actions = n_actions
        self.num_blocks = num_blocks

        adj_matrix = np.ones([num_blocks, num_blocks])
        self.gcn = nn.Sequential(
                GraphConvolution(2, n_hidden, adj_matrix),
                nn.ReLU(),
                GraphConvolution(n_hidden, n_hidden, adj_matrix),
                nn.ReLU(),
                GraphConvolution(n_hidden, n_hidden, adj_matrix)
                )
        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_actions)

    def forward(self, sdfs):
        # sdfs: bs x n x c x h x w
        # concat of ( current_sdfs, goal_sdfs )
        x_conv = self.gcn(sdfs)         # bs x n x c x h x w
        B, N, C, H, W = x_conv.shape
        Q = []
        for n in range(N//2):
            x = x_conv[:, n]                # bs x c x h x w
            x = torch.mean(x, dim=(2, 3))   # bs x c
            x = F.relu(self.fc1(x))
            q = self.fc2(x)                 # bs x na
            Q.append(q.unsqueeze(1))
        Q = torch.cat(Q, axis=1)            # bs x nb x na
        return Q

