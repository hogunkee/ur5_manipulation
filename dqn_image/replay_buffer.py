import numpy as np
import torch
import torch.nn as nn

class ReplayBuffer(object):
    def __init__(self, state_im_dim, goal_im_dim, state_gp_dim, save_goal=False, save_gripper=True, max_size=int(5e5), dim_reward=1):
        self.max_size = max_size
        self.save_goal = save_goal
        self.save_gripper = save_gripper
        self.ptr = 0 
        self.size = 0
        dim_action = 3

        self.state_im = np.zeros([max_size] + list(state_im_dim))
        self.next_state_im = np.zeros([max_size] + list(state_im_dim))
        if self.save_gripper:
            self.state_gp = np.zeros([max_size, state_gp_dim])
            self.next_state_gp = np.zeros([max_size, state_gp_dim])
            dim_action = 1
        self.action = np.zeros((max_size, dim_action))
        self.reward = np.zeros((max_size, dim_reward))
        self.not_done = np.zeros((max_size, 1))
        if self.save_goal:
            self.goal_im = np.zeros([max_size] + list(goal_im_dim))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done, goal=None):
        self.state_im[self.ptr] = state[0]
        self.next_state_im[self.ptr] = next_state[0]
        if self.save_gripper:
            self.state_gp[self.ptr] = state[1]
            self.next_state_gp[self.ptr] = next_state[1]
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        if self.save_goal:
            self.goal_im[self.ptr] = goal

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        if self.save_gripper:
            data_batch = [
                        torch.FloatTensor(self.state_im[ind]).to(self.device),
                        torch.FloatTensor(self.state_gp[ind]).to(self.device),
                        torch.FloatTensor(self.next_state_im[ind]).to(self.device),
                        torch.FloatTensor(self.next_state_gp[ind]).to(self.device),
                        torch.FloatTensor(self.action[ind]).to(self.device),
                        torch.FloatTensor(self.reward[ind]).to(self.device),
                        torch.FloatTensor(self.not_done[ind]).to(self.device),
            ]
        else:
            data_batch = [
                torch.FloatTensor(self.state_im[ind]).to(self.device),
                torch.FloatTensor(self.next_state_im[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device),
            ]
        if self.save_goal:
            data_batch.append(torch.FloatTensor(self.goal_im[ind]).to(self.device))
        return data_batch


class PER(object):
    def __init__(self, state_im_dim, goal_im_dim, state_gp_dim, save_goal=False, save_gripper=True, max_size=int(5e5), dim_reward=1):
        self.max_size = max_size
        self.save_goal = save_goal
        self.save_gripper = save_gripper
        self.ptr = 0
        self.size = 0
        dim_action = 3

        self.tree = np.zeros(2 * max_size - 1)
        self.e = 0.01
        self.a = 0.6
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001

        self.state_im = np.zeros([max_size] + list(state_im_dim))
        self.next_state_im = np.zeros([max_size] + list(state_im_dim))
        if self.save_gripper:
            self.state_gp = np.zeros([max_size, state_gp_dim])
            self.next_state_gp = np.zeros([max_size, state_gp_dim])
            dim_action = 1
        self.action = np.zeros((max_size, dim_action))
        self.reward = np.zeros((max_size, dim_reward))
        self.not_done = np.zeros((max_size, 1))
        if self.save_goal:
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

        self.state_im[self.ptr] = state[0]
        self.next_state_im[self.ptr] = next_state[0]
        if self.save_gripper:
            self.state_gp[self.ptr] = state[1]
            self.next_state_gp[self.ptr] = next_state[1]
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        if self.save_goal:
            self.goal_im[self.ptr] = goal

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
        if self.save_gripper:
            data_batch = [
                        torch.FloatTensor(self.state_im[ind]).to(self.device),
                        torch.FloatTensor(self.state_gp[ind]).to(self.device),
                        torch.FloatTensor(self.next_state_im[ind]).to(self.device),
                        torch.FloatTensor(self.next_state_gp[ind]).to(self.device),
                        torch.FloatTensor(self.action[ind]).to(self.device),
                        torch.FloatTensor(self.reward[ind]).to(self.device),
                        torch.FloatTensor(self.not_done[ind]).to(self.device),
            ]
        else:
            data_batch = [
                torch.FloatTensor(self.state_im[ind]).to(self.device),
                torch.FloatTensor(self.next_state_im[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device),
            ]
        if self.save_goal:
            data_batch.append(torch.FloatTensor(self.goal_im[ind]).to(self.device))

        sampling_probabilities = priorities / self.total()
        is_weight = np.power(self.size * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return data_batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.update_tree(idx, p)
