import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class OneblockQ(nn.Module):
    def __init__(self, n_actions=8, n_hidden=64):
        super(OneblockQ, self).__init__()
        self.mlp = nn.Sequential(
                nn.Linear(4, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_actions)
                )

    def forward(self, x):
        return self.mlp(x)


class ObjectQNet(nn.Module):
    def __init__(self, n_actions, num_blocks):
        super(ObjectQNet, self).__init__()
        self.n_actions = n_actions
        self.num_blocks = num_blocks
        self.Q_nets = []
        for nb in range(self.num_blocks):
            self.Q_nets.append(OneblockQ(n_actions=self.n_actions))

    def forward(self, state_goal):
        states, goals = state_goal
        q_values = []
        for i in range(len(states)):
            s = states[i] # bs x 2
            g = goals[i] # bs x 2
            s_g = torch.cat([s, g], 1) # bs x 4
            q = self.Q_nets[i](s_g) # bs x 8
            q = q.unsqueeze(1) # bs x 1 x 8
            q_values.append(q)
        return torch.cat(q_values, 1) # bs x nb x 8
