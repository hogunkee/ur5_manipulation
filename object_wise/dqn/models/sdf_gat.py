import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GraphAttentionLayer(nn.Module):
    """
    https://github.com/Diego999/pyGAT/
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W) # h.shape: (B, N, in_features), Wh.shape: (B, N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = torch.zeros_like(e) #-9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = attention - torch.where(attention != 0, torch.zeros_like(attention), torch.ones_like(attention) * float('inf'))
        attention = F.softmax(attention, dim=2)
        attention = torch.where(adj > 0, attention, zero_vec)
        #attention = F.dropout(attention, self.dropout, training=self.training)
        return attention

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (B, N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.permute([0,2,1])
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.conv_root = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1) 
        self.conv_inscene = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1) 
        self.conv_goal = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1) 

    def forward(self, x, attn_inscene, attn_goal):
        # x: [BS, N, C, H, W]
        B, N, Cin, Hin, Win = x.shape
        x_spread = x.reshape([B*N, Cin, Hin, Win])

        x_root = self.conv_root(x_spread)
        x_inscene = self.conv_inscene(x_spread)
        x_goal = self.conv_goal(x_spread)

        Cout, Hout, Wout = x_root.shape[-3:]
        x_root_flat = x_root.view([B, N, -1])
        x_inscene_flat = x_root.view([B, N, -1])
        x_goal_flat = x_root.view([B, N, -1])

        x_inscene_attn = torch.matmul(attn_inscene, x_inscene_flat)
        x_goal_attn = torch.matmul(attn_goal, x_goal_flat)
        out = F.relu(x_root_flat + x_inscene_attn + x_goal_attn)
        out = out.view([B, N, Cout, Hout, Wout])
        return out

class GraphConvolutionPoolLayer(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True):
        super(GraphConvolutionPoolLayer, self).__init__()
        self.conv_root = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(3, stride=3, padding=1),
                )
        self.conv_inscene = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(3, stride=3, padding=1),
                )
        self.conv_goal = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(3, stride=3, padding=1),
                )

    def forward(self, x, attn_inscene, attn_goal):
        # x: [BS, N, C, H, W]
        B, N, Cin, Hin, Win = x.shape
        x_spread = x.reshape([B*N, Cin, Hin, Win])

        x_root = self.conv_root(x_spread)
        x_inscene = self.conv_inscene(x_spread)
        x_goal = self.conv_goal(x_spread)

        Cout, Hout, Wout = x_root.shape[-3:]
        x_root_flat = x_root.view([B, N, -1])
        x_inscene_flat = x_root.view([B, N, -1])
        x_goal_flat = x_root.view([B, N, -1])

        x_inscene_attn = torch.matmul(attn_inscene, x_inscene_flat)
        x_goal_attn = torch.matmul(attn_goal, x_goal_flat)
        out = F.relu(x_root_flat + x_inscene_attn + x_goal_attn)
        out = out.view([B, N, Cout, Hout, Wout])
        return out

class SDFGATQNet(nn.Module):
    def __init__(self, num_blocks, sdim=1, fdim=12, hdim=64, n_actions=8):
        super(SDFGATQNet, self).__init__()
        self.n_actions = n_actions
        self.nb_max = num_blocks
        self.sdim = sdim
        self.fdim = fdim
        self.n_actions = n_actions

        self.zeropad = nn.ZeroPad2d((0, num_blocks, 0, num_blocks))
        self.attn_inscene = GraphAttentionLayer(fdim, hdim)
        self.attn_goal = GraphAttentionLayer(fdim, hdim)
        self.cnn1 = nn.Sequential(
                nn.Conv2d(sdim, hdim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(hdim),
                nn.ReLU()
                )
        self.cnn2 = nn.Sequential(
                nn.Conv2d(hdim, 2*hdim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(2*hdim),
                nn.ReLU()
                )
        self.pool = nn.MaxPool2d(3, stride=3, padding=1)
        self.gconv1 = GraphConvolutionLayer(2*hdim, 4*hdim)
        self.cnn3 = nn.Sequential(
                nn.Conv2d(4*hdim, 8*hdim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(8*hdim),
                nn.ReLU()
                )
        self.cnn4 = nn.Sequential(
                nn.Conv2d(8*hdim, 8*hdim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(8*hdim),
                nn.ReLU()
                )
        self.gconv2 = GraphConvolutionLayer(8*hdim, 8*hdim)
        self.fc1 = nn.Linear(8*hdim, 16*hdim)
        self.fc2 = nn.Linear(16*hdim, n_actions)

    def generate_adj(self, nb_st, nb_g):
        NB = self.nb_max
        adj_in = []
        adj_g = []
        for ns, ng in zip(nb_st, nb_g):
            _adj_in = torch.zeros(NB, NB)
            _adj_g = torch.zeros(2*NB, 2*NB)
            _adj_in[:ns, :ns] = torch.ones(ns, ns) - torch.eye(int(ns)) # without self-loop
            _adj_g[:ns, NB:NB+ng] = 1.
            #_adj_g[NB:NB+ng, :ns] = 1. # directed graph (g->st:X)
            adj_in.append(_adj_in.unsqueeze(0))
            adj_g.append(_adj_g.unsqueeze(0))
        adj_inscene = torch.cat(adj_in, 0).to(device)
        adj_goal = torch.cat(adj_g, 0).to(device)
        return adj_inscene, adj_goal

    def forward(self, obs_st, obs_g, nb_st, nb_g):
        # sdf: [B, N, H, W]
        # feature: [B, N, F]
        # nb: [B, 1]
        feature_st, sdfs_st = obs_st
        feature_g, sdfs_g = obs_g
        B = len(nb_st)
        N = self.nb_max
        
        adj_inscene, adj_goal = self.generate_adj(nb_st, nb_g)
        attn_inscene = self.attn_inscene(feature_st, adj_inscene)
        attn_inscene = self.zeropad(attn_inscene)
        attn_goal = self.attn_goal(torch.cat([feature_st, feature_g], 1), adj_goal)
        #return attn_inscene, attn_goal

        sdfs = torch.cat([sdfs_st, sdfs_g], 1) # [B, 2N, H, W]
        sdfs_spread = sdfs.reshape([2*B*N, 1, *sdfs.shape[-2:]])
        x = self.cnn1(sdfs_spread)
        x = self.cnn2(x)
        x = self.pool(x)
        x = x.reshape([B, 2*N, *x.shape[-3:]])
        x = self.gconv1(x, attn_inscene, attn_goal)
        x = x.reshape([2*B*N, *x.shape[-3:]])
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.pool(x)
        x = x.reshape([B, 2*N, *x.shape[-3:]])
        x = self.gconv2(x, attn_inscene, attn_goal)
        x_average = torch.mean(x, dim=(3, 4))  # [B, 2N, C]

        x_current = x_average[:, :N].reshape([B*N, -1])
        x_fc = F.relu(self.fc1(x_current))
        q = self.fc2(x_fc)
        Q = q.view([-1, N, self.n_actions])
        return Q

# V1: Shape CNN structure with GCN
class SDFGATQNetV1(nn.Module):
    def __init__(self, num_blocks, sdim=1, fdim=12, hdim=16, n_actions=8):
        super(SDFGATQNetV1, self).__init__()
        self.n_actions = n_actions
        self.nb_max = num_blocks
        self.sdim = sdim
        self.fdim = fdim
        self.n_actions = n_actions

        self.zeropad = nn.ZeroPad2d((0, num_blocks, 0, num_blocks))
        self.attn_inscene = GraphAttentionLayer(fdim, hdim)
        self.attn_goal = GraphAttentionLayer(fdim, hdim)
        self.cnn1 = nn.Sequential(
                nn.Conv2d(sdim, hdim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(hdim),
                nn.ReLU()
                )
        self.cnn2 = nn.Sequential(
                nn.Conv2d(hdim, 2*hdim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(2*hdim),
                nn.ReLU()
                )
        self.gconv1 = GraphConvolutionPoolLayer(2*hdim, 4*hdim)
        self.cnn3 = nn.Sequential(
                nn.Conv2d(4*hdim, 4*hdim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(4*hdim),
                nn.ReLU()
                )
        self.cnn4 = nn.Sequential(
                nn.Conv2d(4*hdim, 8*hdim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(8*hdim),
                nn.ReLU()
                )
        self.gconv2 = GraphConvolutionPoolLayer(8*hdim, 16*hdim)
        self.fc1 = nn.Linear(16*hdim, 16*hdim)
        self.fc2 = nn.Linear(16*hdim, n_actions)

    def generate_adj(self, nb_st, nb_g):
        NB = self.nb_max
        adj_in = []
        adj_g = []
        for ns, ng in zip(nb_st, nb_g):
            _adj_in = torch.zeros(NB, NB)
            _adj_g = torch.zeros(2*NB, 2*NB)
            _adj_in[:ns, :ns] = torch.ones(ns, ns) - torch.eye(int(ns)) # without self-loop
            _adj_g[:ns, NB:NB+ng] = 1.
            #_adj_g[NB:NB+ng, :ns] = 1. # directed graph (g->st:X)
            adj_in.append(_adj_in.unsqueeze(0))
            adj_g.append(_adj_g.unsqueeze(0))
        adj_inscene = torch.cat(adj_in, 0).to(device)
        adj_goal = torch.cat(adj_g, 0).to(device)
        return adj_inscene, adj_goal

    def forward(self, obs_st, obs_g, nb_st, nb_g):
        # sdf: [B, N, H, W]
        # feature: [B, N, F]
        # nb: [B, 1]
        feature_st, sdfs_st = obs_st
        feature_g, sdfs_g = obs_g
        B = len(nb_st)
        N = self.nb_max
        
        adj_inscene, adj_goal = self.generate_adj(nb_st, nb_g)
        attn_inscene = self.attn_inscene(feature_st, adj_inscene)
        attn_inscene = self.zeropad(attn_inscene)
        attn_goal = self.attn_goal(torch.cat([feature_st, feature_g], 1), adj_goal)
        #return attn_inscene, attn_goal

        sdfs = torch.cat([sdfs_st, sdfs_g], 1) # [B, 2N, H, W]
        sdfs_spread = sdfs.reshape([2*B*N, 1, *sdfs.shape[-2:]])
        x = self.cnn1(sdfs_spread)
        x = self.cnn2(x)
        x = x.reshape([B, 2*N, *x.shape[-3:]])
        x = self.gconv1(x, attn_inscene, attn_goal)
        x = x.reshape([2*B*N, *x.shape[-3:]])
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = x.reshape([B, 2*N, *x.shape[-3:]])
        x = self.gconv2(x, attn_inscene, attn_goal)
        x_average = torch.mean(x, dim=(3, 4))  # [B, 2N, C]

        x_current = x_average[:, :N].reshape([B*N, -1])
        x_fc = F.relu(self.fc1(x_current))
        q = self.fc2(x_fc)
        Q = q.view([-1, N, self.n_actions])
        return Q

# V2: for Multi-dim Input [B, N, C, H, W]
class SDFGATQNetV2(nn.Module):
    def __init__(self, num_blocks, sdim=1, fdim=12, hdim=64, n_actions=8):
        super(SDFGATQNetV2, self).__init__()
        self.n_actions = n_actions
        self.nb_max = num_blocks
        self.sdim = sdim
        self.fdim = fdim
        self.n_actions = n_actions

        self.zeropad = nn.ZeroPad2d((0, num_blocks, 0, num_blocks))
        self.attn_inscene = GraphAttentionLayer(fdim, hdim)
        self.attn_goal = GraphAttentionLayer(fdim, hdim)
        self.cnn1 = nn.Sequential(
                nn.Conv2d(sdim, hdim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(hdim),
                nn.ReLU()
                )
        self.cnn2 = nn.Sequential(
                nn.Conv2d(hdim, 2*hdim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(2*hdim),
                nn.ReLU()
                )
        self.pool = nn.MaxPool2d(3, stride=3, padding=1)
        self.gconv1 = GraphConvolutionLayer(2*hdim, 4*hdim)
        self.cnn3 = nn.Sequential(
                nn.Conv2d(4*hdim, 8*hdim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(8*hdim),
                nn.ReLU()
                )
        self.cnn4 = nn.Sequential(
                nn.Conv2d(8*hdim, 8*hdim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(8*hdim),
                nn.ReLU()
                )
        self.gconv2 = GraphConvolutionLayer(8*hdim, 8*hdim)
        self.fc1 = nn.Linear(8*hdim, 16*hdim)
        self.fc2 = nn.Linear(16*hdim, n_actions)

    def generate_adj(self, nb_st, nb_g):
        NB = self.nb_max
        adj_in = []
        adj_g = []
        for ns, ng in zip(nb_st, nb_g):
            _adj_in = torch.zeros(NB, NB)
            _adj_g = torch.zeros(2*NB, 2*NB)
            _adj_in[:ns, :ns] = torch.ones(ns, ns) - torch.eye(int(ns)) # without self-loop
            _adj_g[:ns, NB:NB+ng] = 1.
            #_adj_g[NB:NB+ng, :ns] = 1. # directed graph (g->st:X)
            adj_in.append(_adj_in.unsqueeze(0))
            adj_g.append(_adj_g.unsqueeze(0))
        adj_inscene = torch.cat(adj_in, 0).to(device)
        adj_goal = torch.cat(adj_g, 0).to(device)
        return adj_inscene, adj_goal

    def forward(self, obs_st, obs_g, nb_st, nb_g):
        # sdf: [B, N, C, H, W]
        # feature: [B, N, F]
        # nb: [B, 1]
        feature_st, sdfs_st = obs_st
        feature_g, sdfs_g = obs_g
        B = len(nb_st)
        N = self.nb_max
        
        adj_inscene, adj_goal = self.generate_adj(nb_st, nb_g)
        attn_inscene = self.attn_inscene(feature_st, adj_inscene)
        attn_inscene = self.zeropad(attn_inscene)
        attn_goal = self.attn_goal(torch.cat([feature_st, feature_g], 1), adj_goal)
        #return attn_inscene, attn_goal

        sdfs = torch.cat([sdfs_st, sdfs_g], 1) # [B, 2N, C, H, W]
        sdfs_spread = sdfs.reshape([2*B*N, *sdfs.shape[-3:]])
        x = self.cnn1(sdfs_spread)
        x = self.cnn2(x)
        x = self.pool(x)
        x = x.reshape([B, 2*N, *x.shape[-3:]])
        x = self.gconv1(x, attn_inscene, attn_goal)
        x = x.reshape([2*B*N, *x.shape[-3:]])
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.pool(x)
        x = x.reshape([B, 2*N, *x.shape[-3:]])
        x = self.gconv2(x, attn_inscene, attn_goal)
        x_average = torch.mean(x, dim=(3, 4))  # [B, 2N, C]

        x_current = x_average[:, :N].reshape([B*N, -1])
        x_fc = F.relu(self.fc1(x_current))
        q = self.fc2(x_fc)
        Q = q.view([-1, N, self.n_actions])
        return Q

