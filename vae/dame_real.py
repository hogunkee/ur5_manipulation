import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.functional import mse_loss
from torch.nn.functional import binary_cross_entropy as bce_loss
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Normal

from torch.autograd import Variable
from collections import OrderedDict
from algo_new_new.networks import *
from algo_new_new.utils import *

eps = 1e-10

class DAME(object):
    def __init__(self, x_dim, z_dim, w_dim, u_dim, K, beta=1.0, lr=1e-3):
        self.modules = {
            'conv': ConvNet().to('cuda'),
            'enc': Encoder(z_dim).to('cuda'),
            'trans': SingleStateTransition(z_dim, u_dim).to('cuda'),
            'dec': SingleStateDecoder(z_dim).to('cuda'),
            'disc': Discriminator(z_dim).to('cuda'),
            'met': Metric(z_dim, w_dim).to('cuda')
        }
        self.main_param, self.disc_param = [], []
        for k in self.modules.keys():
            if k == 'disc':
                self.disc_param += [{'params': self.modules[k].parameters()}]
            else:
                self.main_param += [{'params': self.modules[k].parameters()}]
        self.optimizers = {
            'main_opt': torch.optim.Adam(self.main_param, lr=lr),
            'disc_opt': torch.optim.Adam(self.disc_param, lr=lr)
        }
        
        self.z_dim, self.w_dim, self.u_dim = z_dim, w_dim, u_dim
        self.K = K
        self.beta = beta
        
    def _sample_traj(self, x_list, u_list=None):
        L, B = x_list.size(0), x_list.size(1)
        T = 0 if u_list is None else u_list.size(0)
        L_kl = 0
        
        z_post_list = torch.zeros([L, B, self.z_dim]).to('cuda')
        z_prior_list = torch.zeros([L, B, self.z_dim]).to('cuda')
        
        embedded_x = self.modules['conv'](x_list)
        
        mu_post, std_post, z_post_list[0] = self.modules['enc'](embedded_x[0])
        z_prior_list[0] = Normal(mu_post, std_post).rsample()
        for t in range(T):
            mu_prior, std_prior, z_prior_list[t + 1] = self.modules['trans'](z_post_list[t], u_list[t])
            mu_post, std_post, z_post_list[t + 1] = self.modules['enc'](embedded_x[t + 1], mu_prior, std_prior)
            L_kl += D_kl(mu_prior, std_prior, mu_post, std_post).sum(dim=-1).clamp(min=3).mean() / T
            
        return z_prior_list, z_post_list, L_kl
    
    def _get_recons_loss(self, x_list, z_post):
        x_hat = self.modules['dec'](z_post)
        L_x = 0.5 * mse_loss(x_hat, x_list, reduction='none').sum((2, 3, 4)).mean()
        return L_x
    
    def _get_met_loss(self, z_list):
        L, B = z_list.size(0), z_list.size(1)
        z_list = z_list.reshape([-1, self.z_dim])
        w_list = self.modules['met'](z_list)
        
        L_met = 0
        for i in range(B):
            l = np.random.randint(L)
            query_i = l * B + i
            
            z_query = z_list[query_i].unsqueeze(0).repeat([L * B - 1, 1])
            z_key = torch.cat([z_list[:query_i], z_list[query_i + 1:]], dim=0)
            with torch.no_grad():
                y = self.modules['disc'](z_query, z_key).detach()
            P_i = y / (1. - y + eps)
            P_i = P_i / (torch.sum(P_i) + eps)
            
            w_query = w_list[query_i].unsqueeze(0).repeat([L * B - 1, 1])
            w_key = torch.cat([w_list[:query_i], w_list[query_i + 1:]], dim=0)
            logits = torch.sum((w_query - w_key)**2, dim=-1, keepdim=True)
            Q_i = 1. / (1. + logits)
            Q_i = Q_i / (torch.sum(Q_i) + eps)
            
            L_met += torch.mean(-P_i * torch.log(Q_i + eps)) / B
            
        return L_met
    
    def _get_disc_loss(self, z_key, Pz):
        B = z_key.size(0)
        
        v_list = Variable(torch.rand(self.K, B, self.u_dim) * 2.0 - 1.0).to('cuda')
        
        L_D = 0
        zk = z_key
        for k in range(self.K):
            with torch.no_grad():
                mu, std, zk = self.modules['trans'](zk, v_list[k])
            z_fake = torch.cat([Pz[k:], Pz[:k]], dim=0) if k > 0 else Pz
        
            y_true = self.modules['disc'](zk.detach(), z_key)
            y_fake = self.modules['disc'](z_fake, z_key)
            
            L_D += bce_loss(y_true, torch.ones_like(y_true)) + bce_loss(y_fake, torch.zeros_like(y_fake))
        L_D /= self.K
        
        return L_D
    
    def train_op(self, x_list, u_list, Px, Pu, train_metric=True, train_disc=True):
        for k in self.modules.keys():
            self.modules[k].train()
        x_list = preprocess_obs(x_list)
        Px = preprocess_obs(Px)
        
        x_list = torch.Tensor(x_list).to('cuda')
        u_list = torch.Tensor(u_list).to('cuda')
        Px = torch.Tensor(Px).to('cuda')
        Pu = torch.Tensor(Pu).to('cuda')
        
        L, B = x_list.size(0), x_list.size(1)
        
        z_prior, z_post, L_kl = self._sample_traj(x_list, u_list)
        L_x = self._get_recons_loss(x_list[1:], z_post[1:])
        
        L_met = self._get_met_loss(z_post) if train_metric else torch.zeros(1).to('cuda')
        
        L_main = self.beta * L_kl + L_x + L_met
        
        self.optimizers['main_opt'].zero_grad()
        L_main.backward()
#         clip_grad_norm_(self.main_param, 1000)
        self.optimizers['main_opt'].step()
        
        if train_disc:
            with torch.no_grad():
                _, Pz, _ = self._sample_traj(Px, Pu)
            l = np.random.randint(L)
            L_D = self._get_disc_loss(z_post[l].detach(), Pz[-1].detach())

            self.optimizers['disc_opt'].zero_grad()
            L_D.backward()
            self.optimizers['disc_opt'].step()
        
        
        Losses = OrderedDict([
            ('Lmain', L_main.item()), ('Lx', L_x.item()), ('Lkl', L_kl.item()),
            ('L_met', L_met.item()), ('LD', L_D.item())
        ])
        return Losses
    
    def test(self, x_list, u_list, Px, Pu):
        for k in self.modules.keys():
            self.modules[k].eval()
        x_list = preprocess_obs(x_list, test=True)
        Px = preprocess_obs(Px, test=True)
        
        x_list = torch.Tensor(x_list).to('cuda')
        u_list = torch.Tensor(u_list).to('cuda')
        Px = torch.Tensor(Px).to('cuda')
        Pu = torch.Tensor(Pu).to('cuda')
        
        L, B = x_list.size(0), x_list.size(1)
        
        with torch.no_grad():
            z_prior, z_post, L_kl = self._sample_traj(x_list, u_list)
            L_x = self._get_recons_loss(x_list[1:], z_post[1:])
        
            L_met = self._get_met_loss(z_post)
        
            L_main = self.beta * L_kl + L_x + L_met
        
            _, Pz, _ = self._sample_traj(Px, Pu)
            l = np.random.randint(L)
            L_D = self._get_disc_loss(z_post[l].detach(), Pz[-1].detach())
        
        
        Losses = OrderedDict([
            ('Lmain', L_main.item()), ('Lx', L_x.item()), ('Lkl', L_kl.item()),
            ('L_met', L_met.item()), ('LD', L_D.item())
        ])
        return Losses
    
    def predict(self, x_list, u_list, start=4):
        for k in self.modules.keys():
            self.modules[k].eval()
        x_list = preprocess_obs(x_list, test=True)
        
        x_list = torch.Tensor(x_list).to('cuda')
        u_list = torch.Tensor(u_list).to('cuda')
        L, T = x_list.size(0), u_list.size(0)
        
        x_list = x_list.reshape(L, -1, 64, 64, 3)
        u_list = u_list.reshape(T, -1, self.u_dim)
        B = x_list.size(1)
        
        z_list = torch.zeros([L, B, self.z_dim]).to('cuda')
        z_list_ref = torch.zeros([L, B, self.z_dim]).to('cuda')
        
        with torch.no_grad():
            embedded_x = self.modules['conv'](x_list)
        
            z_post, _, _ = self.modules['enc'](embedded_x[0])
            z_list[0], z_list_ref[0] = z_post, z_post
            for t in range(T):
                z_list[t + 1], _, _ = self.modules['trans'](z_list[t], u_list[t])
                
                z_list_ref[t + 1], std_prior, _ = self.modules['trans'](z_post, u_list[t])
                z_post, _, _ = self.modules['enc'](embedded_x[t + 1], z_list_ref[t + 1], std_prior)
        
        x_hat = (0.5 + self.modules['dec'](z_list)).clamp(min=0, max=1)
        x_ref = (0.5 + self.modules['dec'](z_list_ref)).clamp(min=0, max=1)
        
        return x_hat.cpu().detach().numpy(), x_ref.cpu().detach().numpy()
    
    def save(self, name):
        model_state_dict = dict()
        for k in self.modules.keys():
            model_state_dict[k] = self.modules[k].state_dict()
        for k in self.optimizers.keys():
            model_state_dict[k] = self.optimizers[k].state_dict()
        torch.save(model_state_dict, name)
        print('=' * 20)
        print('model saved at %s' % name)
        print('=' * 20)

    def load(self, name):
        model_state_dict = torch.load(name)
        for k in self.modules.keys():
            self.modules[k].load_state_dict(model_state_dict[k])
        for k in self.optimizers.keys():
            self.optimizers[k].load_state_dict(model_state_dict[k])
        print('=' * 20)
        print('model loaded from %s' % name)
        print('=' * 20)