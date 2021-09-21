import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class OneblockR(nn.Module):
    def __init__(self, n_actions=8, n_hidden=64):
        super(OneblockR, self).__init__()
        self.mlp = nn.Sequential(
                nn.Linear(4, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_actions)
                )

    def forward(self, x):
        return self.mlp(x)


class RewardNetSA(nn.Module):
    def __init__(self, n_actions, num_blocks):
        super(RewardNetSA, self).__init__()
        self.n_actions = n_actions
        self.num_blocks = num_blocks
        R_nets = []
        for nb in range(self.num_blocks):
            R_nets.append(OneblockR(n_actions=self.n_actions))
        self.R_nets = nn.ModuleList(R_nets)

    def forward(self, state_goal):
        states, goals = state_goal
        predict_rewards = []
        bs = states.size()[0]
        for i in range(self.num_blocks):
            s = states[torch.arange(bs), i] # bs x 2
            g = goals[torch.arange(bs), i] # bs x 2
            s_g = torch.cat([s, g], 1) # bs x 4
            r = self.R_nets[i](s_g) # bs x 8
            r = r.unsqueeze(1) # bs x 1 x 8
            predict_rewards.append(r)
        return torch.cat(predict_rewards, 1) # bs x nb x 8


class RewardNetSNS(nn.Module):
    def __init__(self, num_blocks):
        super(RewardNetSNS, self).__init__()
        self.num_blocks = num_blocks
        n_hidden = 64
        self.mlp = nn.Sequential(
                nn.Linear(6*num_blocks, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, 1)
                )

    def forward(self, state_goal_nextstate):
        states, goals, next_states = state_goal_nextstate
        s = states.flatten(1) # bs x nb*2
        g = goals.flatten(1) # bs x nb*2
        ns = next_states.flatten(1) # bs x nb*2
        s_g_ns = torch.cat([s, g, ns], 1) # bs x nb*6
        r = self.mlp(s_g_ns) # bs x 1
        return r
