from torch.utils.data import DataLoader
from data_loader import *

learning_rate = 1e-5
batch_size = 64
num_blocks = 1
data_path = 'data/%dblock'%num_blocks
dataset = SADataset(data_path)
dataloader = DataLoader(dataset, batch_size=batch_size)

for ne in range(num_epochs):
    for i, batch in enumerate(dataloader):
        pass
