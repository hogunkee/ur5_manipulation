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
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        #attention = F.dropout(attention, self.dropout, training=self.training)
        return attention

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True):
        self.conv_root = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1) 
        self.conv_inscene = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1) 
        self.conv_goal = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1) 

    def forward(self, x, adj_inscene, adj_goal):
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

        x_incsene_attn = torch.matmul(adj_inscene, x_inscene_flat)
        x_goal_attn = torch.matmul(adj_goal, x_goal_flat)
        out = F.relu(x_root_flat + x_inscene_attn + x_goal_attn)
        out = out.view([B, N, Cout, Hout, Wout])
        return out


class SDFGATQNet(nn.Module):
    def __init__(self, num_blocks, sdim=1, fdim=12, hdim=64, n_actions=8, n_hidden=16):
        super(SDFGATQNet, self).__init__()
        self.n_actions = n_actions
        self.num_blocks = num_blocks
        self.fdim = fdim
        self.normalize = norm

        self.zeropad = nn.ZeroPad2d((0, 0, num_blocks, num_blocks))
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

    def generate_adj(self, nb_st, nb_g):
        adj_in = []
        adj_g
        for ns, ng in zip(nb_st, nb_g):
            _adj_in = torch.zeros(self.num_blocks, self.num_blocks)
            _adj_g = torch.zeros(2*self.num_blocks, 2*self.num_blocks)
            _adj_in[:ns, :ns] = 1.
            _adj_g[:ng, self.num_blocks:self.num_blocks+ns] = 1.
            _adj_g[self.num_blocks:self.num_blocks+ng, :ns] = 1.
            adj_in.append(_adj_in.unsqueeze(0))
            adj_g.append(_adj_g.unsqueeze(0))
        adj_inscene = torch.cat(adj_in, 0).to(device)
        adj_goal = torch.cat(adj_g, 0).to(device)
        return adj_inscene, adj_goal

    def forward(self, obs_st, obs_g, nb_st, nb_g):
        # sdf: [B, N, C, H, W]
        # feature: [B, N, F]
        # nb: [B, 1]
        sdfs_st, feature_st = obs_st
        sdfs_g, feature_g = obs_g
        
        adj_inscene, adj_goal = self.generate_adj(nb_st, nb_g)
        #print(adj_inscene, adj_goal)
        attn_inscene = self.attn_inscene(feature_st, adj_inscene)
        attn_inscene = self.zeropad(attn_inscene)
        attn_goal = self.attn_goal(torch.cat([feature_st, feature_g], 1), adj_goal)
        #print(attn_inscene, attn_goal)
        return attn_inscene, attn_goal

        #x = self.cnn1

