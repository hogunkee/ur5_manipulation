import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

#dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FCBlock(nn.Module):
    def __init__(self, in_ch, hidden_dims=[64, 64], bias=True):
        super(FCBlock, self).__init__()
        layers = []
        pre_hd = in_ch
        for hd in hidden_dims:
            layers.append(nn.Linear(pre_hd, hd, bias=bias))
            layers.append(nn.ReLU())
            pre_hd = hd
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class GraphConvolution(nn.Module):
    def __init__(self, in_ch, hidden_dims=[8, 16, 32], bias=True):
        super(GraphConvolution, self).__init__()
        self.in_ch = in_ch
        self.out_ch = hidden_dims[-1]

        self.fc_root = FCBlock(in_ch, hidden_dims=hidden_dims, bias=bias)
        self.fc_support = FCBlock(in_ch, hidden_dims=hidden_dims, bias=bias)

    def forward(self, poses, adj_matrix):
        # poses: bs x n x c (c = 2 or 3)
        B, N, C = poses.shape
        poses_spread = poses.reshape([B*N, C])
        x_root = self.fc_root(poses_spread)
        x_support = self.fc_support(poses_spread)
        
        Cout = self.out_ch
        x_root_flat = x_root.view([B, N, Cout])
        x_support_flat = x_support.view([B, N, Cout])
        x_neighbor_flat = torch.matmul(adj_matrix, x_support_flat)

        out = x_root_flat + x_neighbor_flat
        return out

    def __repr__(self):
        return self.__class__.__name__ + f'({self.in_ch} -> {self.out_ch})'


class GraphConvolutionSeparateEdge(nn.Module):
    def __init__(self, in_ch, hidden_dims=[8, 16, 32], bias=True):
        super(GraphConvolutionSeparateEdge, self).__init__()
        self.in_ch = in_ch
        self.out_ch = hidden_dims[-1]

        self.fc_root = FCBlock(in_ch, hidden_dims=hidden_dims, bias=bias)
        self.fc_inscene = FCBlock(in_ch, hidden_dims=hidden_dims, bias=bias)
        self.fc_btwscene = FCBlock(in_ch, hidden_dims=hidden_dims, bias=bias)

    def forward(self, poses, adj_matrix):
        # poses: bs x n c c
        B, N, C = poses.shape
        adj_inscene = copy.deepcopy(adj_matrix)
        adj_inscene[:, :N//2, N//2:] = 0
        adj_inscene[:, N//2:, :N//2] = 0
        adj_btwscene = copy.deepcopy(adj_matrix)
        adj_btwscene[:, :N//2, :N//2] = 0
        adj_btwscene[:, N//2:, N//2:] = 0

        poses_spread = poses.reshape([B*N, C])
        x_root = self.fc_root(poses_spread)
        x_inscene = self.fc_inscene(poses_spread)
        x_btwscene = self.fc_btwscene(poses_spread)

        Cout = x_root.shape[-1]
        x_root_flat = x_root.view([B, N, Cout])
        x_inscene_flat = x_inscene.view([B, N, Cout])
        x_btwscene_flat = x_btwscene.view([B, N, Cout])
        x_neighbor_flat = torch.matmul(adj_inscene, x_inscene_flat) + \
                            torch.matmul(adj_btwscene, x_btwscene_flat)

        out = x_root_flat + x_neighbor_flat
        return out

    def __repr__(self):
        return self.__class__.__name__ + f'({self.in_ch} -> {self.out_ch})'


class GTQNetV0(nn.Module):
    def __init__(self, num_blocks, adj_ver=0, n_actions=8, n_hidden=8, selfloop=False, normalize=False, separate=False, bias=True):
        super(GTQNetV0, self).__init__()
        self.n_actions = n_actions
        self.num_blocks = num_blocks

        self.adj_version = adj_ver
        self.selfloop = selfloop
        self.normalize = normalize
        self.adj_matrix = self.generate_adj()

        if separate:
            graphconv = GraphConvolutionSeparateEdge
        else:
            graphconv = GraphConvolution

        self.gcn1 = graphconv(3, [n_hidden, 2*n_hidden, 4*n_hidden], bias)
        self.gcn2 = graphconv(4*n_hidden, [8*n_hidden, 8*n_hidden, 8*n_hidden], bias)
        self.fc1 = nn.Linear(8*n_hidden, 64)
        self.fc2 = nn.Linear(64, n_actions)

    def generate_adj(self):
        NB = self.num_blocks
        if self.adj_version==-1:
            adj_matrix = torch.ones([2*NB, 2*NB])
        elif self.adj_version==0:
            adj_upper = torch.cat([torch.eye(NB), torch.eye(NB)], 1)
            adj_lower = torch.cat([torch.eye(NB), torch.eye(NB)], 1)
            adj_matrix = torch.cat([adj_upper, adj_lower], 0)
        elif self.adj_version==1:
            adj_upper = torch.cat([torch.ones([NB, NB]), torch.eye(NB)], 1)
            adj_lower = torch.cat([torch.eye(NB), torch.eye(NB)], 1)
            adj_matrix = torch.cat([adj_upper, adj_lower], 0)
        elif self.adj_version==2:
            adj_upper = torch.cat([torch.ones([NB, NB]), torch.eye(NB)], 1)
            adj_lower = torch.cat([torch.eye(NB), torch.ones([NB, NB])], 1)
            adj_matrix = torch.cat([adj_upper, adj_lower], 0)
        elif self.adj_version==3:
            adj_upper = torch.cat([torch.ones([NB, NB]), torch.eye(NB)], 1)
            adj_lower = torch.cat([torch.zeros([NB, NB]), torch.eye(NB)], 1)
            adj_matrix = torch.cat([adj_upper, adj_lower], 0)
        if not self.selfloop:
            adj_matrix = adj_matrix * (1 - torch.eye(2*NB))
        if self.normalize:
            diag = torch.eye(2*NB) / (torch.diag(torch.sum(adj_matrix, 1)) + 1e-10)
            adj_matrix = torch.matmul(adj_matrix, diag)
        return adj_matrix.to(device)

    def forward(self, poses, nsdf):
        # poses: 2 x bs x nb x c (c=2)
        # (current_poses, goal_poses)
        s, g = poses
        poses = torch.cat([s, g], 1)    # bs x 2nb x c
        B, NS, C = poses.shape

        ## adj matrix ##
        adj_matrix = self.adj_matrix.repeat(B, 1, 1)

        ## block flag ##
        block_flags = torch.zeros(B, NS, 1).to(device)
        block_flags[:, :NS//2] = 1.0                        # blocks as 1, goals as 0
        poses_concat = torch.cat([poses, block_flags], 2)   # bs x 2nb x 3

        x_gcn1 = self.gcn1(poses_concat, adj_matrix)
        x_gcn2 = self.gcn2(x_gcn1, adj_matrix)              # bs x 2nb x cout

        x_currents = x_gcn2[:, :self.num_blocks].reshape([B*self.num_blocks, -1])
        x_fc = F.relu(self.fc1(x_currents))
        q = self.fc2(x_fc)                                  # bs*nb x na
        Q = q.view([-1, self.num_blocks, self.n_actions])   # bs x nb x na

        return Q


class GTQNetV1(nn.Module):
    def __init__(self, num_blocks, adj_ver=0, n_actions=8, n_hidden=8, selfloop=False, normalize=False, separate=False, bias=True):
        super(GTQNetV1, self).__init__()
        self.n_actions = n_actions
        self.num_blocks = num_blocks

        self.adj_version = adj_ver
        self.selfloop = selfloop
        self.normalize = normalize
        self.adj_matrix = self.generate_adj()

        if separate:
            graphconv = GraphConvolutionSeparateEdge
        else:
            graphconv = GraphConvolution

        self.gcn1 = graphconv(4, [n_hidden, 2*n_hidden, 4*n_hidden], bias)
        self.gcn2 = graphconv(4*n_hidden, [8*n_hidden, 8*n_hidden, 8*n_hidden], bias)
        self.fc1 = nn.Linear(8*n_hidden, 64)
        self.fc2 = nn.Linear(64, n_actions)

    def generate_adj(self):
        NB = self.num_blocks
        if self.adj_version==-1:
            adj_matrix = torch.ones([2*NB, 2*NB])
        elif self.adj_version==0:
            adj_upper = torch.cat([torch.eye(NB), torch.eye(NB)], 1)
            adj_lower = torch.cat([torch.eye(NB), torch.eye(NB)], 1)
            adj_matrix = torch.cat([adj_upper, adj_lower], 0)
        elif self.adj_version==1:
            adj_upper = torch.cat([torch.ones([NB, NB]), torch.eye(NB)], 1)
            adj_lower = torch.cat([torch.eye(NB), torch.eye(NB)], 1)
            adj_matrix = torch.cat([adj_upper, adj_lower], 0)
        elif self.adj_version==2:
            adj_upper = torch.cat([torch.ones([NB, NB]), torch.eye(NB)], 1)
            adj_lower = torch.cat([torch.eye(NB), torch.ones([NB, NB])], 1)
            adj_matrix = torch.cat([adj_upper, adj_lower], 0)
        elif self.adj_version==3:
            adj_upper = torch.cat([torch.ones([NB, NB]), torch.eye(NB)], 1)
            adj_lower = torch.cat([torch.zeros([NB, NB]), torch.eye(NB)], 1)
            adj_matrix = torch.cat([adj_upper, adj_lower], 0)
        if not self.selfloop:
            adj_matrix = adj_matrix * (1 - torch.eye(2*NB))
        if self.normalize:
            diag = torch.eye(2*NB) / (torch.diag(torch.sum(adj_matrix, 1)) + 1e-10)
            adj_matrix = torch.matmul(adj_matrix, diag)
        return adj_matrix.to(device)

    def forward(self, poses, nsdf):
        # poses: 2 x bs x nb x c (c=2)
        # (current_poses, goal_poses)
        pose_s, pose_g = poses
        B, NB, C = pose_s.shape

        ## adj matrix ##
        adj_matrix = self.adj_matrix.repeat(B, 1, 1)

        poses_concat = torch.cat([pose_s, pose_g], 2)       # bs x nb x 2*c
        x_gcn1 = self.gcn1(poses_concat, adj_matrix)
        x_gcn2 = self.gcn2(x_gcn1, adj_matrix)              # bs x nb x cout

        x_currents = x_gcn2.reshape([B*self.num_blocks, -1])
        x_fc = F.relu(self.fc1(x_currents))
        q = self.fc2(x_fc)                                  # bs*nb x na
        Q = q.view([-1, self.num_blocks, self.n_actions])   # bs x nb x na

        return Q


