import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class GraphConvolution(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_blocks = num_blocks
        self.weight = Parameter(torch.FloatTensor(in_ch, out_ch))
        self.root_weight = Parameter(torch.FloatTensor(in_ch, out_ch))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_ch))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.root_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x): # x: pos1, pos2, .. , goal1, goal2, ..
        nb = self.num_blocks
        adj1 = torch.cat([torch.ones([nb, nb]), torch.eye(nb)])
        adj2 = torch.cat([torch.eye(nb), torch.eye(nb)])
        adj = torch.cat([adj1, adj2], 1).type(dtype) # 2*nb x 2*nb
        support = torch.matmul(x, self.weight)
        out = torch.matmul(adj, support)
        out += torch.matmul(x, self.root_weight)
        if self.bias is not None:
            out += self.bias
        return out

    def __repr__(self):
        return self.__class__.__name__ + f'({self.in_ch} -> {self.out_ch})'


class ObjectQNet(nn.Module):
    def __init__(self, n_actions, num_blocks, n_hidden=64):
        super(ObjectQNet, self).__init__()
        self.n_actions = n_actions
        self.num_blocks = num_blocks
        self.gcn = nn.Sequential(
                GraphConvolution(2, n_hidden, num_blocks),
                nn.ReLU(),
                GraphConvolution(n_hidden, n_hidden, num_blocks),
                nn.ReLU(),
                GraphConvolution(n_hidden, n_actions, num_blocks)
                )

    def forward(self, state_goal):
        states, goals = state_goal
        features = torch.cat([states, goals], 1)
        q = self.gcn(features)
        q = q[:, :self.num_blocks]
        return q

