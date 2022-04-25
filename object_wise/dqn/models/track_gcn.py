import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

#dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN2LayerBlock(nn.Module):
    def __init__(self, in_ch, hidden_dims=[64, 64], pool=2, bias=True):
        super(CNN2LayerBlock, self).__init__()
        self.cnn = nn.Sequential(
                nn.Conv2d(in_ch, hidden_dims[0], kernel_size=3, stride=1, padding=1, bias=bias),
                nn.BatchNorm2d(hidden_dims[0]),
                nn.ReLU(),
                nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=3, stride=1, padding=1, bias=bias),
                nn.BatchNorm2d(hidden_dims[1]),
                nn.ReLU(),
                nn.MaxPool2d(pool, stride=pool, padding=1),
                )

    def forward(self, x):
        return self.cnn(x)

class CNN3LayerBlock(nn.Module):
    def __init__(self, in_ch, hidden_dims=[64, 64, 64], pool=2, bias=True):
        super(CNN3LayerBlock, self).__init__()
        self.cnn = nn.Sequential(
                nn.Conv2d(in_ch, hidden_dims[0], kernel_size=3, stride=1, padding=1, bias=bias),
                nn.BatchNorm2d(hidden_dims[0]),
                nn.ReLU(),
                nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=3, stride=1, padding=1, bias=bias),
                nn.BatchNorm2d(hidden_dims[1]),
                nn.ReLU(),
                nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel_size=3, stride=1, padding=1, bias=bias),
                nn.BatchNorm2d(hidden_dims[2]),
                nn.ReLU(),
                nn.MaxPool2d(pool, stride=pool, padding=1),
                )

    def forward(self, x):
        return self.cnn(x)


class GraphConvolution(nn.Module):
    def __init__(self, in_ch, hidden_dims=[8, 16, 32], pool=3, cnnblock=CNN3LayerBlock, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_ch = in_ch
        self.out_ch = hidden_dims[-1]

        self.conv_root = cnnblock(in_ch, hidden_dims=hidden_dims, pool=pool, bias=bias)
        self.conv_support = cnnblock(in_ch, hidden_dims=hidden_dims, pool=pool, bias=bias)

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
    def __init__(self, in_ch, hidden_dims=[8, 16, 32], pool=3, cnnblock=CNN3LayerBlock, bias=True):
        super(GraphConvolutionSeparateEdge, self).__init__()
        self.in_ch = in_ch
        self.out_ch = hidden_dims[-1]

        self.conv_root = cnnblock(in_ch, hidden_dims=hidden_dims, pool=pool, bias=bias)
        self.conv_inscene = cnnblock(in_ch, hidden_dims=hidden_dims, pool=pool, bias=bias)
        self.conv_btwscene = cnnblock(in_ch, hidden_dims=hidden_dims, pool=pool, bias=bias)

    def forward(self, sdfs, adj_matrix):
        # sdfs: bs x n x c x h x w
        B, N, C, Hin, Win = sdfs.shape
        adj_inscene = copy.deepcopy(adj_matrix)
        adj_inscene[:, :N//2, N//2:] = 0
        adj_inscene[:, N//2:, :N//2] = 0
        adj_btwscene = copy.deepcopy(adj_matrix)
        adj_btwscene[:, :N//2, :N//2] = 0
        adj_btwscene[:, N//2:, N//2:] = 0

        sdfs_block = sdfs[:, :N//2]
        sdfs_block = sdfs_block.reshape([B*N//2, C, Hin, Win])
        sdfs_goal = sdfs[:, N//2:]
        sdfs_goal = sdfs_goal.reshape([B*N//2, C, Hin, Win])

        sdfs_spread = sdfs.reshape([B*N, C, Hin, Win])     # bs*n x c x h x w
        x_root = self.conv_root(sdfs_spread)               # bs*n x cout x h x w
        x_inscene = self.conv_inscene(sdfs_spread)         # bs*n x cout x h x w
        x_btwscene = self.conv_btwscene(sdfs_spread)       # bs*n x cout x h x w

        Cout, Hout, Wout = x_root.shape[-3:]
        x_root_flat = x_root.view([B, N, Cout * Hout * Wout])
        x_inscene_flat = x_inscene.view([B, N, Cout * Hout * Wout])
        x_btwscene_flat = x_btwscene.view([B, N, Cout * Hout * Wout])
        x_neighbor_flat = torch.matmul(adj_inscene, x_inscene_flat) + \
                            torch.matmul(adj_btwscene, x_btwscene_flat)

        out = x_root_flat + x_neighbor_flat
        out = out.view([B, N, Cout, Hout, Wout])
        return out

    def __repr__(self):
        return self.__class__.__name__ + f'({self.in_ch} -> {self.out_ch})'


class TrackQNetV0(nn.Module):
    def __init__(self, num_blocks, n_actions=8, n_hidden=8, normalize=False, resize=True, separate=False, bias=True):
        super(TrackQNetV0, self).__init__()
        self.n_actions = n_actions
        self.num_blocks = num_blocks
        self.normalize = normalize
        self.resize = resize

        self.ws_mask = self.generate_wsmask()

        if separate:
            graphconv = GraphConvolutionSeparateEdge
        else:
            graphconv = GraphConvolution

        self.gcn1 = graphconv(3, [n_hidden, 2*n_hidden, 4*n_hidden], 3, CNN3LayerBlock, bias)
        self.gcn2 = graphconv(4*n_hidden, [8*n_hidden, 8*n_hidden, 8*n_hidden], 3, CNN3LayerBlock, bias)
        self.fc1 = nn.Linear(8*n_hidden, 64)
        self.fc2 = nn.Linear(64, n_actions)

    def generate_wsmask(self):
        if self.resize:
            mask = np.load('../../ur5_mujoco/workspace_mask.npy').astype(float)
        else:
            mask = np.load('../../ur5_mujoco/workspace_mask_480.npy').astype(float)
        return mask

    def forward(self, sdfs):
        # sdfs: 2 x bs x nb x h x w
        # ( current_sdfs, goal_sdfs )
        s, g = sdfs
        sdfs = torch.cat([s, g], 1)         # bs x 2nb x h x w
        B, NS, H, W = sdfs.shape

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
        block_flags = torch.zeros_like(sdfs)
        block_flags[:, :NS//2] = 1.0        # blocks as 1, goals as 0

        ## workspace mask ##
        ws_masks = torch.zeros_like(sdfs)
        ws_masks[:, :] = torch.Tensor(self.ws_mask)

        sdfs_concat = torch.cat([ sdfs.unsqueeze(2), 
                                  block_flags.unsqueeze(2),
                                  ws_masks.unsqueeze(2)
                                 ], 2)   # bs x 2nb x 3 x h x w
        x_conv1 = self.gcn1(sdfs_concat, adj_matrix)        # bs x 2nb x c x h x w
        x_conv2 = self.gcn2(x_conv1, adj_matrix)            # bs x 2nb x cout x h x w
        x_average = torch.mean(x_conv2, dim=(3, 4))         # bs x 2nb x cout

        # x_current: bs*nb x cout
        x_currents = x_average[:, :self.num_blocks].reshape([B*self.num_blocks, -1])
        x_fc = F.relu(self.fc1(x_currents))
        q = self.fc2(x_fc)                                  # bs*nb x na
        Q = q.view([-1, self.num_blocks, self.n_actions])   # bs x nb x na

        return Q


class TrackQNetV1(nn.Module):
    def __init__(self, num_blocks, n_actions=8, n_hidden=8, normalize=False, resize=True, separate=False, bias=True):
        super(TrackQNetV1, self).__init__()
        self.n_actions = n_actions
        self.num_blocks = num_blocks
        self.normalize = normalize
        self.resize = resize

        self.ws_mask = self.generate_wsmask()

        if separate:
            graphconv = GraphConvolutionSeparateEdge
        else:
            graphconv = GraphConvolution

        self.gcn1 = graphconv(3, [n_hidden, 2*n_hidden, 4*n_hidden], 3, CNN3LayerBlock, bias)
        self.gcn2 = graphconv(4*n_hidden, [8*n_hidden, 8*n_hidden, 8*n_hidden], 3, CNN3LayerBlock, bias)
        self.fc1 = nn.Linear(8*n_hidden, 64)
        self.fc2 = nn.Linear(64, n_actions)

    def generate_wsmask(self):
        if self.resize:
            mask = np.load('../../ur5_mujoco/workspace_mask.npy').astype(float)
        else:
            mask = np.load('../../ur5_mujoco/workspace_mask_480.npy').astype(float)
        return mask

    def forward(self, sdfs):
        # sdfs: 2 x bs x nb x h x w
        # ( current_sdfs, goal_sdfs )
        s, g = sdfs
        sdfs = torch.cat([s, g], 1)         # bs x 2nb x h x w
        B, NS, H, W = sdfs.shape

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
        block_flags = torch.zeros_like(sdfs)
        block_flags[:, :NS//2] = 1.0        # blocks as 1, goals as 0

        ## workspace mask ##
        ws_masks = torch.zeros_like(sdfs)
        ws_masks[:, :] = torch.Tensor(self.ws_mask)

        sdfs_concat = torch.cat([ sdfs.unsqueeze(2), 
                                  block_flags.unsqueeze(2),
                                  ws_masks.unsqueeze(2)
                                 ], 2)   # bs x 2nb x 3 x h x w
        x_conv1 = self.gcn1(sdfs_concat, adj_matrix)        # bs x 2nb x c x h x w
        x_conv2 = self.gcn2(x_conv1, adj_matrix)            # bs x 2nb x cout x h x w
        x_average = torch.mean(x_conv2, dim=(3, 4))         # bs x 2nb x cout

        # x_current: bs*nb x cout
        x_currents = x_average[:, :self.num_blocks].reshape([B*self.num_blocks, -1])
        x_fc = F.relu(self.fc1(x_currents))
        q = self.fc2(x_fc)                                  # bs*nb x na
        Q = q.view([-1, self.num_blocks, self.n_actions])   # bs x nb x na

        return Q


class TrackQNetV2(nn.Module):
    def __init__(self, num_blocks, n_actions=8, n_hidden=8, normalize=False, resize=True, separate=False, bias=True):
        super(TrackQNetV2, self).__init__()
        self.n_actions = n_actions
        self.num_blocks = num_blocks
        self.normalize = normalize
        self.resize = resize

        self.ws_mask = self.generate_wsmask()

        if separate:
            graphconv = GraphConvolutionSeparateEdge
        else:
            graphconv = GraphConvolution

        self.gcn1 = graphconv(3, [n_hidden, 2*n_hidden, 4*n_hidden], 3, CNN3LayerBlock, bias)
        self.gcn2 = graphconv(4*n_hidden, [8*n_hidden, 8*n_hidden, 8*n_hidden], 3, CNN3LayerBlock, bias)
        self.fc1 = nn.Linear(8*n_hidden, 64)
        self.fc2 = nn.Linear(64, n_actions)

    def generate_wsmask(self):
        if self.resize:
            mask = np.load('../../ur5_mujoco/workspace_mask.npy').astype(float)
        else:
            mask = np.load('../../ur5_mujoco/workspace_mask_480.npy').astype(float)
        return mask

    def forward(self, sdfs):
        # sdfs: 2 x bs x nb x h x w
        # ( current_sdfs, goal_sdfs )
        s, g = sdfs
        sdfs = torch.cat([s, g], 1)         # bs x 2nb x h x w
        B, NS, H, W = sdfs.shape

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
        block_flags = torch.zeros_like(sdfs)
        block_flags[:, :NS//2] = 1.0        # blocks as 1, goals as 0

        ## workspace mask ##
        ws_masks = torch.zeros_like(sdfs)
        ws_masks[:, :] = torch.Tensor(self.ws_mask)

        sdfs_concat = torch.cat([ sdfs.unsqueeze(2), 
                                  block_flags.unsqueeze(2),
                                  ws_masks.unsqueeze(2)
                                 ], 2)   # bs x 2nb x 3 x h x w
        x_conv1 = self.gcn1(sdfs_concat, adj_matrix)        # bs x 2nb x c x h x w
        x_conv2 = self.gcn2(x_conv1, adj_matrix)            # bs x 2nb x cout x h x w
        x_average = torch.mean(x_conv2, dim=(3, 4))         # bs x 2nb x cout

        # x_current: bs*nb x cout
        x_currents = x_average[:, :self.num_blocks].reshape([B*self.num_blocks, -1])
        x_fc = F.relu(self.fc1(x_currents))
        q = self.fc2(x_fc)                                  # bs*nb x na
        Q = q.view([-1, self.num_blocks, self.n_actions])   # bs x nb x na

        return Q


class TrackQNetV3(nn.Module):
    def __init__(self, num_blocks, n_actions=8, n_hidden=8, normalize=False, separate=False, bias=True):
        super(TrackQNetV3, self).__init__()
        self.n_actions = n_actions
        self.num_blocks = num_blocks
        self.normalize = normalize

        self.ws_mask = self.generate_wsmask()

        if separate:
            graphconv = GraphConvolutionSeparateEdge
        else:
            graphconv = GraphConvolution

        self.gcn1 = graphconv(3, [n_hidden, 2*n_hidden], 2, CNN2LayerBlock, bias)
        self.gcn2 = graphconv(2*n_hidden, [4*n_hidden, 8*n_hidden], 2, CNN2LayerBlock, bias)
        self.gcn3 = graphconv(8*n_hidden, [8*n_hidden, 8*n_hidden], 2, CNN2LayerBlock, bias)
        self.cnn = nn.Sequential(
                nn.Conv2d(8*n_hidden, 8*n_hidden, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(8*n_hidden),
                nn.ReLU(),
                )
        self.fc1 = nn.Linear(8*n_hidden, 64)
        self.fc2 = nn.Linear(64, n_actions)

    def generate_wsmask(self):
        mask = np.load('../../ur5_mujoco/workspace_mask.npy').astype(float)
        return mask

    def forward(self, sdfs):
        # sdfs: 2 x bs x nb x h x w
        # ( current_sdfs, goal_sdfs )
        s, g = sdfs
        sdfs = torch.cat([s, g], 1)         # bs x 2nb x h x w
        B, NS, H, W = sdfs.shape

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
        block_flags = torch.zeros_like(sdfs)
        block_flags[:, :NS//2] = 1.0        # blocks as 1, goals as 0

        ## workspace mask ##
        ws_masks = torch.zeros_like(sdfs)
        ws_masks[:, :] = torch.Tensor(self.ws_mask)

        sdfs_concat = torch.cat([ sdfs.unsqueeze(2), 
                                  block_flags.unsqueeze(2),
                                  ws_masks.unsqueeze(2)
                                 ], 2)   # bs x 2nb x 3 x h x w
        x_conv1 = self.gcn1(sdfs_concat, adj_matrix)        # bs x 2nb x c x h x w
        x_conv2 = self.gcn2(x_conv1, adj_matrix)            # bs x 2nb x cout x h x w
        x_conv3 = self.gcn3(x_conv2, adj_matrix)            # bs x 2nb x cout x h x w
        x_conv3_spread = x_conv3.reshape([B*NS, *x_conv3.shape[2:]])
        x_conv4_spread = self.cnn(x_conv3_spread)
        x_conv4 = x_conv4_spread.reshape([B, NS, *x_conv4_spread.shape[1:]])
        x_average = torch.mean(x_conv4, dim=(3, 4))         # bs x 2nb x cout

        # x_current: bs*nb x cout
        x_currents = x_average[:, :self.num_blocks].reshape([B*self.num_blocks, -1])
        x_fc = F.relu(self.fc1(x_currents))
        q = self.fc2(x_fc)                                  # bs*nb x na
        Q = q.view([-1, self.num_blocks, self.n_actions])   # bs x nb x na

        return Q

class TrackQNetV4(nn.Module):
    def __init__(self, num_blocks, n_actions=8, n_hidden=8, normalize=False, separate=False, bias=True):
        super(TrackQNetV4, self).__init__()
        self.n_actions = n_actions
        self.num_blocks = num_blocks
        self.normalize = normalize

        self.ws_mask = self.generate_wsmask()

        if separate:
            graphconv = GraphConvolutionSeparateEdge
        else:
            graphconv = GraphConvolution

        self.gcn1 = graphconv(3, [n_hidden, 2*n_hidden], 2, CNN2LayerBlock, bias)
        self.gcn2 = graphconv(2*n_hidden, [4*n_hidden, 8*n_hidden], 2, CNN2LayerBlock, bias)
        self.gcn3 = graphconv(8*n_hidden, [8*n_hidden, 8*n_hidden], 2, CNN2LayerBlock, bias)
        self.cnn = nn.Sequential(
                nn.Conv2d(8*n_hidden, 8*n_hidden, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(8*n_hidden),
                nn.ReLU(),
                )
        self.fc1 = nn.Linear(8*n_hidden, 64)
        self.fc2 = nn.Linear(64, n_actions)

    def generate_wsmask(self):
        mask = np.load('../../ur5_mujoco/workspace_mask.npy').astype(float)
        return mask

    def forward(self, sdfs):
        # sdfs: 2 x bs x nb x h x w
        # ( current_sdfs, goal_sdfs )
        s, g = sdfs
        sdfs = torch.cat([s, g], 1)         # bs x 2nb x h x w
        B, NS, H, W = sdfs.shape

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
        block_flags = torch.zeros_like(sdfs)
        block_flags[:, :NS//2] = 1.0        # blocks as 1, goals as 0

        ## workspace mask ##
        ws_masks = torch.zeros_like(sdfs)
        ws_masks[:, :] = torch.Tensor(self.ws_mask)

        sdfs_concat = torch.cat([ sdfs.unsqueeze(2), 
                                  block_flags.unsqueeze(2),
                                  ws_masks.unsqueeze(2)
                                 ], 2)   # bs x 2nb x 3 x h x w
        x_conv1 = self.gcn1(sdfs_concat, adj_matrix)        # bs x 2nb x c x h x w
        x_conv2 = self.gcn2(x_conv1, adj_matrix)            # bs x 2nb x cout x h x w
        x_conv3 = self.gcn3(x_conv2, adj_matrix)            # bs x 2nb x cout x h x w
        x_conv3_spread = x_conv3.reshape([B*NS, *x_conv3.shape[2:]])
        x_conv4_spread = self.cnn(x_conv3_spread)
        x_conv4 = x_conv4_spread.reshape([B, NS, *x_conv4_spread.shape[1:]])
        x_average = torch.mean(x_conv4, dim=(3, 4))         # bs x 2nb x cout

        # x_current: bs*nb x cout
        x_currents = x_average[:, :self.num_blocks].reshape([B*self.num_blocks, -1])
        x_fc = F.relu(self.fc1(x_currents))
        q = self.fc2(x_fc)                                  # bs*nb x na
        Q = q.view([-1, self.num_blocks, self.n_actions])   # bs x nb x na

        return Q

