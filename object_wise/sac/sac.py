import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy

class SAC(object):
    def __init__(self, max_blocks, args):
        self.target_res = 96
        self.max_blocks = max_blocks
        self.gamma = 0.99 #args.gamma
        self.tau = 0.005 #args.tau
        self.alpha = 0.1 #args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.critic = QNetwork(max_blocks, args.ver, args.adj_ver, args.n_hidden, args.selfloop, \
                args.normalize, args.resize, args.separate, args.bias).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(max_blocks, args.ver, args.adj_ver, args.n_hidden, args.selfloop, \
                args.normalize, args.resize, args.separate, args.bias).to(device=self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(2).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(max_blocks, args.ver, args.adj_ver, args.n_hidden, args.selfloop, \
                    args.normalize, args.resize, args.separate, args.bias).to(device=self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(max_blocks, args.ver, args.adj_ver, args.n_hidden, args.selfloop, \
                    args.normalize, args.resize, args.separate, args.bias).to(device=self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def pad_sdf(self, sdf):
        nsdf = len(sdf)
        padded = np.zeros([self.max_blocks, self.target_res, self.target_res])
        if nsdf > self.max_blocks:
            padded[:] = sdf[:self.max_blocks]
        elif nsdf > 0:
            padded[:nsdf] = sdf
        return padded

    def random_action(self, sdfs):
        sidx = np.random.choice(np.where(np.sum(sdfs[0], (1,2))!=0)[0])
        #nsdf = sdfs[0].shape[0]
        #sidx = np.random.randint(nsdf)
        a = np.random.uniform(-1, 1, 2)
        action = a * self.policy.action_scale.cpu().numpy() + self.policy.action_bias.cpu().numpy()
        return (sidx, *action)

    def select_action(self, sdfs, evaluate=False):
        nsdf = sdfs[0].shape[0]
        nsdf = torch.LongTensor([nsdf]).to(self.device)
        s = self.pad_sdf(sdfs[0])
        s = torch.FloatTensor(s).to(self.device).unsqueeze(0)
        g = self.pad_sdf(sdfs[1])
        g = torch.FloatTensor(g).to(self.device).unsqueeze(0)

        if evaluate is False:
            sidx, displacement, _, _ = self.policy.sample([s, g], nsdf)
        else:
            sidx, _, _, displacement = self.policy.sample([s, g], nsdf)
        displacement = displacement.detach().cpu().numpy()[0]
        return (sidx, *displacement)

    def process_action(self, action):
        theta = np.arctan2(action[0], action[1])
        theta = int((theta/np.pi)%2 * 4)
        return theta

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        minibatch = memory.sample(batch_size)
        state = minibatch[0]
        next_state = minibatch[1]
        rewards = minibatch[3]
        actions = minibatch[2]
        actions = (actions[:, 0].type(torch.long), actions[:, 1:])
        not_done = minibatch[4]
        goal = minibatch[5]
        next_goal = minibatch[6]
        nsdf = minibatch[7].squeeze()
        next_nsdf = minibatch[8].squeeze()
        batch_size = state.size()[0]

        state_goal = [state, goal]
        next_state_goal = [next_state, next_goal]

        with torch.no_grad():
            next_sidx, next_displacement, next_log_pi, _ = self.policy.sample(next_state_goal, next_nsdf)
            next_action = (next_sidx, next_displacement)
            #next_action, next_log_pi, _ = self.policy.sample(next_state_goal, next_nsdf)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_goal, next_action, next_nsdf)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_log_pi
            next_q_value = rewards + not_done * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_goal, actions, nsdf)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        sidx, displacement, log_pi, _ = self.policy.sample(state_goal, nsdf)
        pi = (sidx, displacement)

        qf1_pi, qf2_pi = self.critic(state_goal, pi, nsdf)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, actor_path=None, critic_path=None):
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

