import os
import numpy as np

import torch
from torch.utils.data import Dataset

def load_urdata(datapath):
    fname_list = [f for f in os.listdir(datapath) if 'state_' in f]
    fpath_list = [os.path.join(datapath, f) for f in fname_list]
    data_list = []
    for f in fpath_list:
        data = np.load(f)
        data_list.append(data)
    return np.concatenate(data_list) # ne x 50 x 3 x 96 x 96

class URDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.x = torch.tensor(data)
        indices = np.array([np.arange(self.x.shape[1])] * self.x.shape[0])
        self.y = torch.tensor(indices)

    def __getitem__(self, index):
        x = self.x[index] / 255
        y = self.y[index]
        return x, y

    def __len__(self):
        return len(self.x)


class CustomDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = torch.tensor(data)
        #self.x = x_tensor
        #self.y = y_tensor

    def __getitem__(self, index):
        x, y = self._sample_pairs(index)
        x_tensor = torch.tensor(x)
        y_tensor = torch.tensor(y)
        return x_tensor, y_tensor

    def __len__(self):
        return len(self.data)

    def _sample_pairs(self, index, num_pairs=16):
        idx = np.random.random_integers(0, self.data.shape[1]-1, 2*num_pairs)
        distance = np.abs(idx[0] - idx[1])

        data1 = self.data[index][idx[0]]
        data2 = self.data[index][idx[1]]
        data_concat = np.concatenate([data1, data2]).reshape(2, num_pairs, 3, 96, 96).transpose([1, 0, 2, 3, 4])
        return data_concat, distance


