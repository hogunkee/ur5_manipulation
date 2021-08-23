import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU()
        )
    def forward(self, x_list):
        L, B = x_list.size(0), x_list.size(1)
        x_list = x_list.reshape(-1, 64, 64, 3).permute(0, 3, 1, 2)
        embedded_x = self.net(x_list).reshape(L, B, 1024)
        return embedded_x
    
class Encoder(nn.Module):
    def __init__(self, z_dim, h_dim=200, _min_std=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * z_dim + 1024, 2 * z_dim)
        )
        self.z_dim = z_dim
        self._min_std = _min_std
        
    def forward(self, embedded_x, mu_prior=None, std_prior=None):
        B = embedded_x.size(0)
        if mu_prior is None or std_prior is None:
            mu_prior = torch.zeros([B, self.z_dim]).to('cuda')
            std_prior = torch.zeros([B, self.z_dim]).to('cuda')
        hidden = torch.cat([mu_prior, std_prior, embedded_x], dim=-1)
        hidden = self.net(hidden)
        
        mu, prestd = hidden.split(self.z_dim, dim=-1)
        std = F.softplus(prestd) + self._min_std
        z = Normal(mu, std).rsample()
        
        return mu, std, z
    
class Transition(nn.Module):
    def __init__(self, z_dim, u_dim, h_dim=200, _min_std=0.):
        super().__init__()
        self.z_dim, self.u_dim = z_dim, u_dim
        self._min_std = _min_std
        self.fc_state_action = nn.Sequential(
            nn.Linear(2 * z_dim + u_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 2 * z_dim)
        )
        
    def forward(self, z_prior, z_post, u):
        if np.random.uniform() < 0.5:
            hidden = torch.cat([z_prior, z_post, u], dim=-1)
        else:
            hidden = torch.cat([z_post, z_prior, u], dim=-1)
        hidden = self.fc_state_action(hidden)
        
        delta_mu, pre_std = hidden.split(self.z_dim, dim=-1)
        mu = z_prior + delta_mu
        std = F.softplus(pre_std) + self._min_std
        
        z = Normal(mu, std).rsample()
        
        return mu, std, z

class SingleStateTransition(nn.Module):
    def __init__(self, z_dim, u_dim, h_dim=200, _min_std=0.):
        super().__init__()
        self.z_dim, self.u_dim = z_dim, u_dim
        self._min_std = _min_std
        self.fc_state_action = nn.Sequential(
            nn.Linear(z_dim + u_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 2 * z_dim)
        )
        
    def forward(self, z, u):
        hidden = torch.cat([z, u], dim=-1)
        hidden = self.fc_state_action(hidden)
        
        delta_mu, pre_std = hidden.split(self.z_dim, dim=-1)
        mu = z + delta_mu
        std = F.softplus(pre_std) + self._min_std
        
        z_next = Normal(mu, std).rsample()
        
        return mu, std, z_next

    
class SingleStateDecoder(nn.Module):
    """
    p(o_t | s_t, h_t)
    Observation model to reconstruct image observation (3, 64, 64)
    from state and rnn hidden state
    """
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim, 1024)
        self.dc1 = nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2)
        self.dc2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.dc3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)
        self.dc4 = nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2)

    def forward(self, z):
        L, B = z.size(0), z.size(1)
        hidden = self.fc(z.reshape(-1, self.z_dim))
        hidden = hidden.view(hidden.size(0), 1024, 1, 1)
        hidden = F.relu(self.dc1(hidden))
        hidden = F.relu(self.dc2(hidden))
        hidden = F.relu(self.dc3(hidden))
        obs = self.dc4(hidden)
        return obs.permute(0, 2, 3, 1).reshape([L, B, 64, 64, 3])
    
class Decoder(nn.Module):
    """
    p(o_t | s_t, h_t)
    Observation model to reconstruct image observation (3, 64, 64)
    from state and rnn hidden state
    """
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.fc = nn.Linear(2 * z_dim, 1024)
        self.dc1 = nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2)
        self.dc2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.dc3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)
        self.dc4 = nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2)

    def forward(self, z_post, z_prior):
        L, B = z_post.size(0), z_post.size(1)
        z_post = z_post.reshape([-1, self.z_dim])
        z_prior = z_prior.reshape([-1, self.z_dim])
        z = torch.cat([z_post, z_prior], dim=1) if np.random.uniform() < 0.5 else torch.cat([z_prior, z_post], dim=1)
        hidden = self.fc(z)
        hidden = hidden.view(hidden.size(0), 1024, 1, 1)
        hidden = F.relu(self.dc1(hidden))
        hidden = F.relu(self.dc2(hidden))
        hidden = F.relu(self.dc3(hidden))
        obs = self.dc4(hidden)
        return obs.permute(0, 2, 3, 1).reshape([L, B, 64, 64, 3])
    
class Metric(nn.Module):
    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 200),
            nn.ReLU(),
            nn.Linear(200, w_dim)
        )
        
    def forward(self, z):
        return self.fc(z)
    
class Discriminator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * z_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 1)
        )
        
    def forward(self, z_query, z_key):
        h = torch.cat([z_query - z_key, z_key], dim=-1)
        return torch.sigmoid(self.fc(h))
        