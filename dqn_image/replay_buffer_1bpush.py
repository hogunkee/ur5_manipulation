import numpy as np
import torch
import torch.nn as nn

class ReplayBuffer(object):
    def __init__(self, im_dim, state_dim, max_size=int(5e5), dim_action=1, dim_reward=1):
        self.max_size = max_size
        self.ptr = 0 
        self.size = 0

        self.state_im = np.zeros([max_size] + list(im_dim))
        self.next_state_im = np.zeros([max_size] + list(im_dim))
        self.state = np.zeros([max_size, state_dim])
        self.next_state = np.zeros([max_size, state_dim])
        self.action = np.zeros((max_size, dim_action))
        self.reward = np.zeros((max_size, dim_reward))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done, goal=None):
        self.state_im[self.ptr] = state[0]
        self.next_state_im[self.ptr] = next_state[0]
        self.state[self.ptr] = state[1]
        self.next_state[self.ptr] = next_state[1]
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        data_batch = [
                    torch.FloatTensor(self.state_im[ind]).to(self.device),
                    torch.FloatTensor(self.state[ind]).to(self.device),
                    torch.FloatTensor(self.next_state_im[ind]).to(self.device),
                    torch.FloatTensor(self.next_state[ind]).to(self.device),
                    torch.FloatTensor(self.action[ind]).to(self.device),
                    torch.FloatTensor(self.reward[ind]).to(self.device),
                    torch.FloatTensor(self.not_done[ind]).to(self.device),
                    ]
        return data_batch

