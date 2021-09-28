import datetime
from torch.utils.data import DataLoader
from data_loader import *
from models.reward_net import RewardNetSA as RNet

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


learning_rate = 1e-4
batch_size = 256
num_blocks = 1
num_epochs = 100
data_path = 'data/%dblock'%num_blocks

dataset = SADataset(data_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

rnet = RNet(8, num_blocks).type(dtype)
rnet.train()
optimizer = torch.optim.Adam(rnet.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

for ne in range(num_epochs):
    running_loss = 0.
    for step, batch in enumerate(dataloader):
        [state_goal, action], reward = batch
        s = state_goal[:, 0].type(dtype)
        g = state_goal[:, 1].type(dtype)
        r = reward.type(dtype)
        r_hat = rnet([s, g])[torch.arange(batch_size), action[:, 0], action[:, 1]]

        optimizer.zero_grad()
        loss = criterion(r_hat, r)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"[{ne+1}/{num_epochs}]")
    print(f"loss: {running_loss/len(dataloader)}")

now = datetime.datetime.now()
savename = "PRSA_%s" % (now.strftime("%m%d_%H%M"))
torch.save(rnet.state_dict(), 'results/models/%s.pth' % savename)