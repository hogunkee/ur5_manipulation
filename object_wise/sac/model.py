import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

# CNN block
class CNNLayerBlock(nn.Module):
    def __init__(self, in_ch, hidden_dims=[64, 64, 64], pool=2, bias=True):
        super(CNNLayerBlock, self).__init__()
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


# GCN block
class GraphConvolution(nn.Module):
    def __init__(self, in_ch, hidden_dims=[8, 16, 32], pool=3, cnnblock=CNNLayerBlock, bias=True):
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
    def __init__(self, in_ch, hidden_dims=[8, 16, 32], pool=3, cnnblock=CNNLayerBlock, bias=True):
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

# Graph Encoder
class GraphEncoderV0(nn.Module):
    def __init__(self, max_blocks, adj_ver=0, n_hidden=8, selfloop=False, normalize=False, resize=True, separate=False, bias=True):
        super(GraphEncoderV0, self).__init__()
        self.max_blocks = max_blocks

        self.adj_version = adj_ver
        self.selfloop = selfloop
        self.normalize = normalize
        self.resize = resize

        self.ws_mask = self.generate_wsmask()
        self.adj_matrix = self.generate_adj()

        if separate:
            graphconv = GraphConvolutionSeparateEdge
        else:
            graphconv = GraphConvolution

        self.gcn1 = graphconv(3, [n_hidden, 2*n_hidden, 4*n_hidden], 3, CNNLayerBlock, bias)
        self.gcn2 = graphconv(4*n_hidden, [8*n_hidden, 8*n_hidden, 8*n_hidden], 3, CNNLayerBlock, bias)

    def generate_wsmask(self):
        if self.resize:
            mask = np.load('../../ur5_mujoco/workspace_mask.npy').astype(float)
        else:
            mask = np.load('../../ur5_mujoco/workspace_mask_480.npy').astype(float)
        return mask

    def generate_adj(self):
        NB = self.max_blocks
        adj_matrix = torch.zeros([NB, 2 * NB, 2 * NB])
        for nb in range(1, NB + 1):
            if self.adj_version==0:
                adj_matrix[nb - 1, NB:NB + nb, :nb] = torch.eye(nb)
                adj_matrix[nb - 1, :nb, NB:NB + nb] = torch.eye(nb)
            elif self.adj_version==1:
                adj_matrix[nb - 1, :nb, :nb] = torch.ones([nb, nb])
                adj_matrix[nb - 1, NB:NB + nb, :nb] = torch.eye(nb)
                adj_matrix[nb - 1, :nb, NB:NB + nb] = torch.eye(nb)
                adj_matrix[nb - 1, NB:NB + nb, NB:NB + nb] = torch.eye(nb)
            elif self.adj_version==2:
                adj_matrix[nb - 1, :nb, :nb] = torch.ones([nb, nb])
                adj_matrix[nb - 1, NB:NB + nb, :nb] = torch.eye(nb)
                adj_matrix[nb - 1, :nb, NB:NB + nb] = torch.eye(nb)
                adj_matrix[nb - 1, NB:NB + nb, NB:NB + nb] = torch.ones([nb, nb])
            elif self.adj_version==3:
                adj_matrix[nb - 1, :nb, :nb] = torch.ones([nb, nb])
                adj_matrix[nb - 1, :nb, NB:NB + nb] = torch.eye(nb)
                adj_matrix[nb - 1, NB:NB + nb, NB:NB + nb] = torch.eye(nb)
            if not self.selfloop:
                adj_matrix[nb - 1] = adj_matrix[nb - 1] * (1 - torch.eye(2*NB))
            if self.normalize:
                diag = torch.eye(2*NB) / (torch.diag(torch.sum(adj_matrix[nb - 1], 1)) + 1e-10)
                adj_matrix[nb - 1] = torch.matmul(adj_matrix[nb - 1], diag)
        return adj_matrix.to(device)

    def forward(self, sdfs, nsdf):
        # sdfs: 2 x bs x nb x h x w
        # ( current_sdfs, goal_sdfs )
        s, g = sdfs
        sdfs = torch.cat([s, g], 1)         # bs x 2nb x h x w
        B, NS, H, W = sdfs.shape

        ## adj matrix ##
        adj_matrix = self.adj_matrix[nsdf-1]

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

        # x_current: bs x nb x cout
        x_currents = x_average[:, :self.max_blocks]
        return x_currents


class GraphEncoderV1(nn.Module):
    def __init__(self, max_blocks, adj_ver=0, n_hidden=8, selfloop=False, normalize=False, resize=True, separate=False, bias=True):
        super(GraphEncoderV1, self).__init__()
        self.max_blocks = max_blocks

        self.adj_version = adj_ver
        self.selfloop = selfloop
        self.normalize = normalize
        self.resize = resize

        self.ws_mask = self.generate_wsmask()
        self.adj_matrix = self.generate_adj()

        graphconv = GraphConvolution
        self.gcn1 = graphconv(3, [n_hidden, 2*n_hidden, 4*n_hidden], 3, CNN3LayerBlock, bias)
        self.gcn2 = graphconv(4*n_hidden, [8*n_hidden, 8*n_hidden, 8*n_hidden], 3, CNN3LayerBlock, bias)

    def generate_wsmask(self):
        if self.resize:
            mask = np.load('../../ur5_mujoco/workspace_mask.npy').astype(float)
        else:
            mask = np.load('../../ur5_mujoco/workspace_mask_480.npy').astype(float)
        return mask

    def generate_adj(self):
        NB = self.max_blocks
        adj_matrix = torch.zeros([NB, NB, NB])
        for nb in range(1, NB + 1):
            if self.adj_version==0:
                adj_matrix[nb - 1, :nb, :nb] = torch.eye(nb)
            elif self.adj_version==1:
                adj_matrix[nb - 1, :nb, :nb] = torch.ones([nb, nb])
            if not self.selfloop:
                adj_matrix[nb - 1] = adj_matrix[nb - 1] * (1 - torch.eye(NB))
            if self.normalize:
                diag = torch.eye(NB) / (torch.diag(torch.sum(adj_matrix[nb - 1], 1)) + 1e-10)
                adj_matrix[nb - 1] = torch.matmul(adj_matrix[nb - 1], diag)
        return adj_matrix.to(device)

    def forward(self, sdfs, nsdf):
        # sdfs: 2 x bs x nb x h x w
        # ( current_sdfs, goal_sdfs )
        sdfs_s, sdfs_g = sdfs
        B, NB, H, W = sdfs_s.shape

        ## adj matrix ##
        adj_matrix = self.adj_matrix[nsdf-1]

        ## workspace mask ##
        ws_masks = torch.zeros_like(sdfs_s)
        ws_masks[:, :] = torch.Tensor(self.ws_mask)

        sdfs_concat = torch.cat([ sdfs_s.unsqueeze(2), 
                                  sdfs_g.unsqueeze(2),
                                  ws_masks.unsqueeze(2)
                                 ], 2)                      # bs x nb x 3 x h x w
        x_conv1 = self.gcn1(sdfs_concat, adj_matrix)        # bs x nb x c x h x w
        x_conv2 = self.gcn2(x_conv1, adj_matrix)            # bs x nb x cout x h x w
        x_currents = torch.mean(x_conv2, dim=(3, 4))         # bs x nb x cout
        return x_currents


# Q networks
class QNetwork(nn.Module):
    def __init__(self, max_blocks, ver=0, adj_ver=0, n_hidden=8, selfloop=False, normalize=False, resize=True, separate=False, bias=True):
        super(QNetwork, self).__init__()
        if ver==0:
            self.gcn = GraphEncoderV0(max_blocks, adj_ver, n_hidden, selfloop, normalize, resize, separate, bias)
        elif ver==1:
            self.gcn = GraphEncoderV1(max_blocks, adj_ver, n_hidden, selfloop, normalize, resize, separate, bias)

        self.fc1 = nn.Linear(8*n_hidden + 2, 128)
        self.fc2 = nn.Linear(128, 1)
        self.fc3 = nn.Linear(8*n_hidden + 2, 128)
        self.fc4 = nn.Linear(128, 1)
        self.apply(weights_init_)

    def forward(self, sdfs, action, nsdf, idx_sdf):
        features = self.gcn(sdfs, nsdf)                 # bs x nb x c
        selected_feature = features[:, idx_sdf]         # bs x c
        x = torch.cat([selected_feature, action], 1)    # bs x (c+2)

        x1 = F.relu(self.fc1(x))
        Q1 = self.fc2(x1)

        x2 = F.relu(self.fc3(x))
        Q2 = self.fc4(x2)
        return Q1, Q2


# Policy networks
class GaussianPolicy(nn.Module):
    def __init__(self, max_blocks, ver=0, adj_ver=0, n_hidden=8, selfloop=False, normalize=False, rezize=True, separate=False, bias=True):
        super(GaussianPolicy, self).__init__()
        if ver==0:
            self.gcn = GraphEncoderV0(max_blocks, adj_ver, n_hidden, selfloop, normalize, resize, separate, bias)
        elif ver==1:
            self.gcn = GraphEncoderV1(max_blocks, adj_ver, n_hidden, selfloop, normalize, resize, separate, bias)

        self.fc1 = nn.Linear(8*n_hidden, 128)
        self.fc2 = nn.Linear(8*n_hidden, 128)
        self.fc3 = nn.Linear(8*n_hidden, 128)
        self.fc_mean = nn.Linear(128, 2)
        self.fc_log_std = nn.Linear(128, 2)
        self.fc_dist = nn.Linear(128, 1)
        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, sdfs, nsdf):
        features = self.gcn(sdfs, nsdf)                         # bs x nb x c
        B, NB, C = features.shape
        f_spread = features.reshape([-1, features.shape[-1]])   # bs*nb x c

        x_mean = F.relu(self.fc1(f_spread))
        mean = self.fc_mean(x_mean).reshape([B, NB, 2])         # bs x nb x 2
        x_logstd = F.relu(self.fc2(f_spread))
        log_std = self.fc_log_std(x_logstd).reshape([B, NB, 2]) # bs x nb x 2
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        x_dist = F.relu(self.fc3(f_spread))
        x_dist = self.fc_dist(x_dist)                           # bs*nb x 1
        x_dist = x_dist.reshape([B, NB])                        # bs x nb 
        obj_dist = F.gumbel_softmax(x_dist, hard=True)

        #selected = torch.argmax(obj_dist, 1)
        #one_hot = torch.zeros_like(obj_dist)
        #one_hot[torch.arange(b), selected] = obj_dist[torch.arange(b), selected]
        selected = torch.argmax(obj_dist, 1)
        select_mean = torch.matmul(obj_dist.reshape(B, 1, NB), mean).squeeze(1)
        select_log_std = torch.matmul(obj_dist.reshape(B, 1, NB), log_std).squeeze(1)
        return selected, select_mean, select_log_std
    
    def sample(self, sdfs, nsdf):
        select, mean, log_std = self.forward(sdfs, nsdf)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        # print(f'mean: {mean}')
        # print(f'action: {action}')
        # print(f'action-mean: {(action-mean).abs().mean()}')
        # print('std:', std.mean(axis=0))
        return select, action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, max_blocks, ver=0, adj_ver=0, n_hidden=8, selfloop=False, normalize=False, rezize=True, separate=False, bias=True):
        super(DeterministicPolicy, self).__init__()
        if ver==0:
            self.gcn = GraphEncoderV0(max_blocks, adj_ver, n_hidden, selfloop, normalize, resize, separate, bias)
        elif ver==1:
            self.gcn = GraphEncoderV1(max_blocks, adj_ver, n_hidden, selfloop, normalize, resize, separate, bias)

        self.fc1 = nn.Linear(8*n_hidden, 128)
        self.fc2 = nn.Linear(8*n_hidden, 128)
        self.fc_mean = nn.Linear(128, 2)
        self.fc_dist = nn.Linear(128, 1)
        self.noise = torch.Tensor(2)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, sdfs, nsdf):
        features = self.gcn(sdfs, nsdf)                         # bs x nb x c
        B, NB, C = features.shape
        f_spread = features.reshape([-1, features.shape[-1]])   # bs*nb x c

        x_mean = F.relu(self.fc1(f_spread))
        mean = self.fc_mean(x_mean).reshape([B, NB, 2])         # bs x nb x 2
        #mean = self.fc_mean(x_mean)                             # bs*nb x 2
        #mean = torch.tanh(mean) * self.action_scale + self.action_bias
        #mean = mean.reshape([B, NB, 2])                         # bs x nb x 2

        x_dist = F.relu(self.fc2(f_spread))
        x_dist = self.fc_dist(x_dist)                           # bs*nb x 1
        x_dist = x_dist.reshape([B, NB])                        # bs x nb 
        obj_dist = F.gumbel_softmax(x_dist, hard=True)

        selected = torch.argmax(obj_dist, 1)
        select_mean = torch.matmul(obj_dist.reshape(B, 1, NB), mean).squeeze(1)
        return selected, select_mean

    def sample(self, sdfs, nsdf):
        select, mean = self.forward(sdfs, nsdf)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = torch.tanh(mean + noise) * self.action_scale + self.action_bias
        return select, action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
