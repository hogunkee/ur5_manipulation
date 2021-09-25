import os
import numpy as np

import torch
from torch.utils.data import Dataset

def load_urdata(datapath):
    s_list = [f for f in os.listdir(datapath) if f.startswith('state_')]
    ns_list = [f for f in os.listdir(datapath) if f.startswith('nextstate_')]
    a_list = [f for f in os.listdir(datapath) if f.startswith('action_')]
    r_list = [f for f in os.listdir(datapath) if f.startswith('reward_')]
    d_list = [f for f in os.listdir(datapath) if f.startswith('done_')]
    f_list = [f for f in os.listdir(datapath) if f.startswith('frame_')]
    nf_list = [f for f in os.listdir(datapath) if f.startswith('nextframe_')]

    def load_numpy(fname_list):
        fpath_list = [os.path.join(datapath, f) for f in fname_list]
        data_list = []
        for f in fpath_list:
            data = np.load(f)
            data_list.append(data)
        return np.concatenate(data_list)

    states = load_numpy(s_list)
    next_states = load_numpy(ns_list)
    actions = load_numpy(a_list)
    rewards = load_numpy(r_list)
    # dones = load_numpy(d_list)
    # frames = load_numpy(f_list)
    # next_frames = load_numpy(nf_list)
    return states, next_states, actions, rewards


class SADataset(Dataset):
    def __init__(self, datapath):
        super().__init__()
        s_data, ns_data, a_data, r_data = load_urdata(datapath)
        self.s = torch.tensor(s_data)
        self.a = torch.tensor(a_data)
        self.r = torch.tensor(r_data)

    def __getitem__(self, index):
        s = self.s[index]
        a = self.a[index]
        r = self.r[index]
        return [s, a], r

    def __len__(self):
        return len(self.r)


class SNSDataset(Dataset):
    def __init__(self, datapath):
        super().__init__()
        s_data, ns_data, a_data, r_data = load_urdata(datapath)
        self.s = torch.tensor(s_data)
        self.ns = torch.tensor(ns_data)
        self.r = torch.tensor(r_data)

    def __getitem__(self, index):
        s = self.s[index]
        ns = self.ns[index]
        r = self.r[index]
        return [s, ns], r

    def __len__(self):
        return len(self.r)
