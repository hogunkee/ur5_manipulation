import torch
import torch.nn as nn
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

def preprocess_obs(obs, bit_depth=5, test=False):
    """
    Reduces the bit depth of image for the ease of training
    and convert to [-0.5, 0.5]
    In addition, add uniform random noise same as original implementation
    """
    obs = obs.astype(np.float32)
    p = 1. if test else np.random.uniform(0.9, 1.0)
    obs *= p
    reduced_obs = np.floor(obs / 2 ** (8 - bit_depth))
    normalized_obs = reduced_obs / 2**bit_depth - 0.5
    if not test:
        normalized_obs += np.random.uniform(0.0, 1.0 / 2**bit_depth, normalized_obs.shape)
    return normalized_obs

def D_kl(mu1, std1, mu2, std2):
    return kl_divergence(Normal(mu1, std1), Normal(mu2, std2))

def BCE_loss(x, x_hat):
    return torch.mean(torch.sum(
        -x * torch.log(torch.clamp(x_hat, eps, 1.0)) - (1.0 - x) * torch.log(torch.clamp(1. - x_hat, eps, 1.0)), dim=-1
    ))

def L2_loss(x, x_hat):
    return torch.mean(torch.sum((x - x_hat)**2, dim=-1))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0.)
    if type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        
class DataLoader(object):
    def __init__(self, data_pth, num_test_epi, L, batch_size):
        self.batch_size = batch_size
        self.L = L
        with open(data_pth, 'rb') as f:
            self.data = pickle.load(f, encoding='latin1')
        num_epi = len(self.data['x_list'])
        self.u_dim = self.data['a_list'].shape[-1]
        self.data['x_list'] = np.concatenate([self.data['x_list'][:, :1], self.data['x_list']], axis=1)
        self.data['a_list'] = np.concatenate([np.zeros([num_epi, 1, self.u_dim]), self.data['a_list']], axis=1)
        
        idxs = np.arange(len(self.data['x_list']))
        np.random.shuffle(idxs)
        self.data['x_list'] = self.data['x_list'][idxs].astype(np.uint8)
        self.data['a_list'] = self.data['a_list'][idxs]
        self.num_train_epi = len(self.data['x_list']) - num_test_epi
        
        self.L_max = self.data['x_list'].shape[1]
        
        self.train_idxs = np.arange(self.num_train_epi)
        np.random.shuffle(self.train_idxs)
        self.max_sample = len(self.train_idxs) // batch_size
        self.num_sample = 0
        
        
        
    def reset(self):
        self.num_sample = 0
        np.random.shuffle(self.train_idxs)
    
    def sample_batch(self):
        if self.num_sample == self.max_sample:
            self.reset()
        batch_idx = self.train_idxs[self.num_sample * self.batch_size:(self.num_sample + 1) * self.batch_size]
        t_0 = np.random.randint(self.L_max - self.L)
        x_batch = self.data['x_list'][batch_idx, t_0:t_0 + self.L + 1]
        a_batch = self.data['a_list'][batch_idx, t_0:t_0 + self.L]
        self.num_sample += 1
        
        x_batch[self.batch_size // 2:] = x_batch[self.batch_size // 2:, :, :, ::-1, :]
        a_batch[self.batch_size // 2:, :, 0] = -a_batch[self.batch_size // 2:, :, 0]
        x_batch, a_batch = np.transpose(x_batch, (1, 0, 2, 3, 4)), np.transpose(a_batch, (1, 0, 2))
        
        return x_batch, a_batch
    
    def sample_px(self):
        stack = np.random.randint(3, self.L_max)
        batch_epi = np.random.choice(self.train_idxs, self.batch_size, replace=False)
        batch_t = np.random.choice(self.L_max - stack + 1, self.batch_size)
        x_batch = [self.data['x_list'][(batch_epi, batch_t + i)] for i in range(stack)]
        u_batch = [self.data['a_list'][(batch_epi, batch_t + i)] for i in range(stack - 1)]
        x_batch = np.stack(x_batch, axis=0)
        u_batch = np.stack(u_batch, axis=0)
        return x_batch, u_batch
    
    def get_test_set(self):
        x_test, a_test = [], []
        for l in range(self.L_max - self.L):
            x_test.append(self.data['x_list'][self.num_train_epi:, l:l + self.L + 1])
            a_test.append(self.data['a_list'][self.num_train_epi:, l:l + self.L])
        x_test = np.concatenate(x_test)
        a_test = np.concatenate(a_test)
        
        return np.transpose(x_test, (1, 0, 2, 3, 4)), np.transpose(a_test, (1, 0, 2))
    
def print_info(info_dict, headline):
    print('=' * 20 + headline + '=' * 20)
    for k in info_dict.keys():
        print(k, end=' ')
        print('%.4f' % info_dict[k], end='\t')
    print()
    
def show_traj(x_list, length=6):
    L = x_list.shape[0]
    step_size = (L - 1) // 5
    plt.figure(figsize=[5*length, 5])
    for l in range(length):
        plt.subplot(1, 6, 1 + l)
        plt.imshow(x_list[l * step_size])
    plt.show()
    plt.close()