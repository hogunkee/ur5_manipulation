import datetime
from torch.utils.data import DataLoader
from data_loader import *
from models.reward_net import RewardNetSNS as RNet

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


learning_rate = 2e-4
batch_size = 64
num_blocks = 2
num_epochs = 50 #100
data_path = 'data/%dblock'%num_blocks

dataset = SNSDataset(data_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

rnet = RNet(num_blocks).type(dtype)
rnet.train()
optimizer = torch.optim.Adam(rnet.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

for ne in range(num_epochs):
    running_loss = 0.
    for step, batch in enumerate(dataloader):
        [state_goal, next_state_goal], reward = batch
        s = state_goal[:, 0].type(dtype)
        g = state_goal[:, 1].type(dtype)
        ns = next_state_goal[:, 0].type(dtype)
        r = reward.type(dtype)
        r_hat = rnet([s, g, ns])

        optimizer.zero_grad()
        loss = criterion(r_hat, r)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"[{ne+1}/{num_epochs}]")
    print(f"loss: {running_loss/len(dataloader)}")

now = datetime.datetime.now()
savename = "PRSNS_%s" % (now.strftime("%m%d_%H%M"))
torch.save(rnet.state_dict(), 'results/models/%s.pth' % savename)
