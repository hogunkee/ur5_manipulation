import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.weight = Parameter(torch.FloatTensor(in_ch, out_ch))
        self.root_weight = Parameter(torch.FloatTensor(in_ch, out_ch))
        if bias:
            self.bias = Parameter(torch.FloatTensor(in_ch, out_ch))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.root_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        out = torch.spmm(adj, support)
        out += torch.mm(x, self.root_weight)
        if self.bias is not None:
            out += self.bias
        return out

    def __repr__(self):
        return self.__class__.__name__ + f'({self.in_ch} -> {self.out_ch})'


class OneblockQ(nn.Module):
    def __init__(self, n_actions=8, n_hidden=64):
        super(OneblockQ, self).__init__()
        self.gcn = nn.Sequential(
                GraphConvolution(4, n_hidden),
                nn.ReLU(),
                GraphConvolution(n_hidden, n_hidden),
                nn.ReLU(),
                GraphConvolution(n_hidden, n_actions)
                )

    def forward(self, x):
        return self.gcn(x)


class ObjectQNet(nn.Module):
    def __init__(self, n_actions, num_blocks):
        super(ObjectQNet, self).__init__()
        self.n_actions = n_actions
        self.num_blocks = num_blocks
        Q_nets = []
        for nb in range(self.num_blocks):
            Q_nets.append(OneblockQ(n_actions=self.n_actions))
        self.Q_nets = nn.ModuleList(Q_nets)

    def forward(self, state_goal):
        states, goals = state_goal
        q_values = []
        bs = states.size()[0]
        nb = states.size()[1]
        for i in range(nb):
            s = states[torch.arange(bs), i] # bs x 2
            g = goals[torch.arange(bs), i] # bs x 2
            s_g = torch.cat([s, g], 1) # bs x 4
            q = self.Q_nets[i](s_g) # bs x 8
            q = q.unsqueeze(1) # bs x 1 x 8
            q_values.append(q)
        return torch.cat(q_values, 1) # bs x nb x 8

