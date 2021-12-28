import numpy as np
import torch
import torch.nn as nn

class ReplayBuffer(object):
    def __init__(self, state_dim, goal_dim, max_size=int(5e5), dim_reward=1, \
            state_im_dim=None, goal_im_dim=None):
        self.max_size = max_size
        self.ptr = 0 
        self.size = 0
        self.save_img = not (state_im_dim is None)
        dim_action = 2

        self.numblocks = np.zeros((max_size, 1))
        self.next_numblocks = np.zeros((max_size, 1))
        self.state = np.zeros([max_size] + list(state_dim))
        self.next_state = np.zeros([max_size] + list(state_dim))
        self.action = np.zeros((max_size, dim_action))
        self.reward = np.zeros((max_size, dim_reward))
        self.not_done = np.zeros((max_size, 1))
        self.goal = np.zeros([max_size] + list(goal_dim))

        if self.save_img:
            self.state_im = np.zeros([max_size] + list(state_im_dim))
            self.next_state_im = np.zeros([max_size] + list(state_im_dim))
            self.goal_im = np.zeros([max_size] + list(goal_im_dim))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done, goal):
        if self.save_img:
            self.state_im[self.ptr] = state[1]
            self.next_state_im[self.ptr] = next_state[1]
            self.goal_im[self.ptr] = goal[1]
            state = state[0]
            next_state = next_state[0]
            goal = goal[0]

        n_blocks = len(state)
        next_n_blocks = len(next_state)
        self.numblocks[self.ptr] = n_blocks
        self.next_numblocks[self.ptr] = next_n_blocks
        self.state[self.ptr][:n_blocks] = state
        self.next_state[self.ptr][:n_blocks] = next_state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.goal[self.ptr][:n_blocks] = goal

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        data_batch = [
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.goal[ind]).to(self.device),
            torch.LongTensor([self.numblocks[ind]]).to(self.device),
            torch.LongTensor([self.next_numblocks[ind]]).to(self.device),
        ]
        if self.save_img:
            data_bath.append(torch.FloatTensor(self.state_im[ind]).to(self.device))
            data_bath.append(torch.FloatTensor(self.next_state_im[ind]).to(self.device))
            data_bath.append(torch.FloatTensor(self.goal_im[ind]).to(self.device))
        return data_batch


class PER(object):
    def __init__(self, state_dim, goal_dim, max_size=int(5e5), dim_reward=1, \
            state_im_dim=None, goal_im_dim=None):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.save_img = not (state_im_dim is None)
        dim_action = 2

        self.tree = np.zeros(2 * max_size - 1)
        self.e = 0.01
        self.a = 0.6
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001

        self.numblocks = np.zeros((max_size, 1))
        self.next_numblocks = np.zeros((max_size, 1))
        self.state = np.zeros([max_size] + list(state_dim))
        self.next_state = np.zeros([max_size] + list(state_dim))
        self.action = np.zeros((max_size, dim_action))
        self.reward = np.zeros((max_size, dim_reward))
        self.not_done = np.zeros((max_size, 1))
        self.goal = np.zeros([max_size] + list(goal_dim))

        if self.save_img:
            self.state_im = np.zeros([max_size] + list(state_im_dim))
            self.next_state_im = np.zeros([max_size] + list(state_im_dim))
            self.goal_im = np.zeros([max_size] + list(goal_im_dim))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## tree functions ##
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def update_tree(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get_tree(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.max_size + 1
        return (idx, self.tree[idx], data_idx)

    ## replay buffer functions ##
    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, state, action, next_state, reward, done, goal=None):
        p = self._get_priority(error)
        idx = self.ptr + self.max_size - 1

        if self.save_img:
            self.state_im[self.ptr] = state[1]
            self.next_state_im[self.ptr] = next_state[1]
            self.goal_im[self.ptr] = goal[1]
            state = state[0]
            next_state = next_state[0]
            goal = goal[0]

        n_blocks = len(state)
        next_n_blocks = len(next_state)
        self.numblocks[self.ptr] = n_blocks
        self.next_numblocks[self.ptr] = next_n_blocks
        self.state[self.ptr][:n_blocks] = state
        self.next_state[self.ptr][:n_blocks] = next_state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.goal[self.ptr][:n_blocks] = goal

        self.update_tree(idx, p)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, n):
        priorities = []
        data_idxs = []
        idxs = []
        segment = self.total() / n
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i+1)
            s = np.random.uniform(a, b)
            (idx, p, data_idx) = self.get_tree(s)
            priorities.append(p)
            data_idxs.append(data_idx)
            idxs.append(idx)

        ind = np.array(data_idxs)
        data_batch = [
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.goal[ind]).to(self.device),
            torch.LongTensor([self.numblocks[ind]]).to(self.device),
            torch.LongTensor([self.next_numblocks[ind]]).to(self.device),
        ]
        if self.save_img:
            data_bath.append(torch.FloatTensor(self.state_im[ind]).to(self.device))
            data_bath.append(torch.FloatTensor(self.next_state_im[ind]).to(self.device))
            data_bath.append(torch.FloatTensor(self.goal_im[ind]).to(self.device))

        sampling_probabilities = priorities / self.total()
        is_weight = np.power(self.size * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return data_batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.update_tree(idx, p)
