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
    def __init__(self, num_blocks, n_actions=8, n_hidden=8, normalize=False, separate=False, bias=True):
        super(GTQNetV0, self).__init__()
        self.n_actions = n_actions
        self.num_blocks = num_blocks
        self.normalize = normalize

        if separate:
            graphconv = GraphConvolutionSeparateEdge
        else:
            graphconv = GraphConvolution

        self.gcn1 = graphconv(3, [n_hidden, 2*n_hidden, 4*n_hidden], bias)
        self.gcn2 = graphconv(4*n_hidden, [8*n_hidden, 8*n_hidden, 8*n_hidden], bias)
        self.fc1 = nn.Linear(8*n_hidden, 64)
        self.fc2 = nn.Linear(64, n_actions)

    def forward(self, poses):
        # poses: 2 x bs x nb x c (c=2)
        # (current_poses, goal_poses)
        s, g = poses
        poses = torch.cat([s, g], 1)    # bs x 2nb x c
        B, NS, C = poses.shape

        ## adj matrix ##
        ms = (torch.sum(s, (2, 3))!=0).type(torch.float32)
        mg = (torch.sum(g, (2, 3))!=0).type(torch.float32)
        sg_pair = torch.logical_and(ms, mg).type(torch.float32)
        A_gs = []
        for pair in sg_pair:
            A_gs.append(pair * torch.eye(NS//2).to(device))
        A_gs = torch.cat(A_gs).reshape(B, NS//2, NS//2)
        A_ss = torch.zeros([B, NS//2, NS//2]).to(device)
        A_gg = torch.zeros([B, NS//2, NS//2]).to(device)

        A1 = torch.cat((A_ss, A_gs), 2)
        A2 = torch.cat((A_gs, A_gg), 2)
        A = torch.cat((A1, A2), 1)
        adj_matrix = A * (1-torch.eye(NS)).to(device)

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
    def __init__(self, num_blocks, n_actions=8, n_hidden=8, normalize=False, separate=False, bias=True):
        super(GTQNetV1, self).__init__()
        self.n_actions = n_actions
        self.num_blocks = num_blocks
        self.normalize = normalize

        if separate:
            graphconv = GraphConvolutionSeparateEdge
        else:
            graphconv = GraphConvolution

        self.gcn1 = graphconv(3, [n_hidden, 2*n_hidden, 4*n_hidden], bias)
        self.gcn2 = graphconv(4*n_hidden, [8*n_hidden, 8*n_hidden, 8*n_hidden], bias)
        self.fc1 = nn.Linear(8*n_hidden, 64)
        self.fc2 = nn.Linear(64, n_actions)

    def forward(self, poses):
        # poses: 2 x bs x nb x c (c=2)
        # (current_poses, goal_poses)
        s, g = poses
        poses = torch.cat([s, g], 1)    # bs x 2nb x c
        B, NS, C = poses.shape

        ## adj matrix ##
        ms = (torch.sum(s, (2, 3))!=0).type(torch.float32)
        mg = (torch.sum(g, (2, 3))!=0).type(torch.float32)
        sg_pair = torch.logical_and(ms, mg).type(torch.float32)
        A_gs = []
        for pair in sg_pair:
            A_gs.append(pair * torch.eye(NS//2).to(device))
        A_gs = torch.cat(A_gs).reshape(B, NS//2, NS//2)
        A_ss = ((ms.reshape(B, NS//2, 1) + ms.reshape(B, 1, NS//2))==2).type(torch.float32)
        A_gg = ((mg.reshape(B, NS//2, 1) + mg.reshape(B, 1, NS//2))==2).type(torch.float32)

        A1 = torch.cat((A_ss, A_gs), 2)
        A2 = torch.cat((A_gs, A_gg), 2)
        A = torch.cat((A1, A2), 1)
        adj_matrix = A * (1-torch.eye(NS)).to(device)

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


class GTQNetV2(nn.Module):
    def __init__(self, num_blocks, n_actions=8, n_hidden=8, normalize=False, separate=False, bias=True):
        super(GTQNetV2, self).__init__()
        self.n_actions = n_actions
        self.num_blocks = num_blocks
        self.normalize = normalize

        if separate:
            graphconv = GraphConvolutionSeparateEdge
        else:
            graphconv = GraphConvolution

        self.gcn1 = graphconv(3, [n_hidden, 2*n_hidden, 4*n_hidden], bias)
        self.gcn2 = graphconv(4*n_hidden, [8*n_hidden, 8*n_hidden, 8*n_hidden], bias)
        self.fc1 = nn.Linear(8*n_hidden, 64)
        self.fc2 = nn.Linear(64, n_actions)

    def forward(self, poses):
        # poses: 2 x bs x nb x c (c=2)
        # (current_poses, goal_poses)
        s, g = poses
        poses = torch.cat([s, g], 1)    # bs x 2nb x c
        B, NS, C = poses.shape

        ## adj matrix ##
        ms = (torch.sum(s, (2, 3))!=0).type(torch.float32)
        mg = (torch.sum(g, (2, 3))!=0).type(torch.float32)
        sg_pair = torch.logical_and(ms, mg).type(torch.float32)
        A_gs = []
        for pair in sg_pair:
            A_gs.append(pair * torch.eye(NS//2).to(device))
        A_gs = torch.cat(A_gs).reshape(B, NS//2, NS//2)
        A_ss = ((ms.reshape(B, NS//2, 1) + ms.reshape(B, 1, NS//2))==2).type(torch.float32)
        A_gg = ((mg.reshape(B, NS//2, 1) + mg.reshape(B, 1, NS//2))==2).type(torch.float32)

        A1 = torch.cat((A_ss, A_gs), 2)
        A2 = torch.cat((torch.zeros_like(A_gs), A_gg), 2)
        A = torch.cat((A1, A2), 1)
        adj_matrix = A * (1-torch.eye(NS)).to(device)

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

