import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../ur5_mujoco'))
from pushpixel_env import *

import torch
import torch.nn as nn
import argparse

import datetime
from models.mlp import QNet
from matplotlib import pyplot as plt

dtype = torch.FloatTensor 
crop_min = 19
crop_max = 78

def combine_batch(minibatch, data):
    combined = []
    for i in range(len(minibatch)):
        combined.append(torch.cat([minibatch[i], data[i].unsqueeze(0)]))
    return combined

class ReplayBuffer(object):
    def __init__(self, dim_state, max_size=int(5e5)):
        self.max_size = max_size
        self.ptr = 0 
        self.size = 0
        dim_action = 3

        self.state = np.zeros((max_size, dim_state))
        self.next_state = np.zeros((max_size, dim_state))
        self.action = np.zeros((max_size, dim_action))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.next_state[self.ptr] = next_state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

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
        ]
        return data_batch

def learning(env, 
        savename,
        n_actions=8,
        learning_rate=1e-4, 
        batch_size=64, 
        buff_size=1e4, 
        total_steps=1e6,
        learn_start=1e4,
        update_freq=100,
        log_freq=1e3,
        double=True
        ):

    state_dim = 4*env.num_blocks
    Q = QNet(n_actions, state_dim).type(dtype)
    Q_target = QNet(n_actions, state_dim).type(dtype)
    Q_target.load_state_dict(Q.state_dict())
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(Q.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(state_dim, max_size=int(buff_size))

    def get_action(q_net, state, epsilon):
        if np.random.random() < epsilon:
            action = [np.random.randint(crop_min,crop_max), np.random.randint(crop_min,crop_max), np.random.randint(env.num_bins)]
        else:
            state_tensor = torch.tensor(state).type(dtype)
            q_raw = q_net(state_tensor).detach().cpu().numpy()
            q_pixel = (q_raw[:2] + 1/2) * env.env.camera_width
            px, py = np.clip(q_pixel, crop_min, crop_max).astype(int)
            theta = np.argmax(q_raw[2:])
            action = [px, py, theta]
        return action

    def calculate_loss_origin(minibatch, gamma=0.99):
        rewards = minibatch[3]
        not_done = minibatch[4]

        state = minibatch[0]
        next_state = minibatch[1]

        next_q = Q_target(next_state)[:, 2:]
        next_q_max = next_q[torch.arange(batch_size), next_q.max(1)[1]]
        y_target = rewards + gamma * not_done * next_q_max
        #print('y target:', y_target)
        q_values = Q(state)[:, 2:]
        actions = minibatch[4]
        max_q = q_values[torch.arange(batch_size), actions.type(torch.long)]
        #print('q:', max_q)
        loss = criterion(y_target, max_q)

        return loss

    def calculate_loss_double(minibatch, gamma=0.99):
        rewards = minibatch[3]
        not_done = minibatch[4]

        state = minibatch[0]
        next_state = minibatch[1]

        next_q = Q(next_state)[:, 2:].detach()
        _, a_prime = next_q.max(1)

        q_target_next = Q_target(next_state)[:, 2:].detach()
        q_target_s_a_prime = q_target_next.gather(1, a_prime.unsqueeze(1))
        q_target_s_a_prime = q_target_s_a_prime.squeeze()

        next_q_max = not_done * q_target_s_a_prime
        y_target = rewards + gamma * not_done * next_q_max
        #print('y target:', y_target)
        q_values = Q(state)[:, 2:]
        actions = minibatch[4]
        max_q = q_values[torch.arange(batch_size), actions.type(torch.long)]
        #max_q = q_values.gather(1, action.unsqueeze(1))
        #max_q = max_q.squeeze()
        #print('q:', max_q)
        loss = criterion(y_target, max_q)

        return loss

    if double:
        calculate_loss = calculate_loss_double
    else:
        calculate_loss = calculate_loss_origin

    log_returns = []
    log_loss = []
    log_eplen = []
    log_epsilon = []
    log_out = []
    log_success = []
    log_collisions = []
    log_minibatchloss = []

    log_mean_returns = []
    log_mean_loss = []
    log_mean_eplen = []
    log_mean_out = []
    log_mean_success = []
    log_mean_collisions = []

    if not os.path.exists("results/graph/"):
        os.makedirs("results/graph/")
    if not os.path.exists("results/models/"):
        os.makedirs("results/models/")
    if not os.path.exists("results/board/"):
        os.makedirs("results/board/")

    #plt.ion()
    plt.show(block=False)
    plt.rc('axes', labelsize=6)
    plt.rc('font', size=6)
    f, axes = plt.subplots(3, 2)
    f.set_figheight(9) #15
    f.set_figwidth(12) #10

    axes[0][0].set_title('Loss')
    axes[1][0].set_title('Episode Return')
    axes[2][0].set_title('Episode Length')
    axes[0][1].set_title('Success Rate')
    axes[1][1].set_title('Out of Range')
    axes[2][1].set_title('Num Collisions')

    lr_decay = 0.98
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)

    epsilon = 1.0
    start_epsilon = 0.5
    min_epsilon = 0.01
    epsilon_decay = 0.98
    episode_reward = 0.0
    max_rewards = -1000
    ep_len = 0
    ne = 0
    t_step = 0
    num_collisions = 0

    state = env.reset()

    while t_step<total_steps:
        action = get_action(Q, state, epsilon)
        next_state, reward, done, info = env.step(action)
        episode_reward += reward
        epsilon = max(0.999*epsilon, min_epsilon)

        replay_buffer.add(state, action, next_state, reward, done)

        if t_step<learn_start:
            if done:
                state = env.reset()
                episode_reward = 0
            else:
                state = next_state
            learn_start -= 1
            if learn_start==0:
                epsilon = start_epsilon
            continue

        data = [
                torch.FloatTensor(state).type(dtype),
                torch.FloatTensor(next_state).type(dtype),
                torch.FloatTensor(action).type(dtype),
                torch.FloatTensor([reward]).type(dtype),
                torch.FloatTensor([1 - done]).type(dtype),
                ]
        minibatch = replay_buffer.sample(batch_size-1)
        combined_minibatch = combine_batch(minibatch, data)
        loss = calculate_loss(combined_minibatch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        log_minibatchloss.append(loss.data.detach().cpu().numpy())

        state = next_state
        t_step += 1
        ep_len += 1
        num_collisions += int(info['collision'])

        if t_step%update_freq:
            Q_target.load_state_dict(Q.state_dict())
            lr_scheduler.step()
            epsilon = max(epsilon_decay * epsilon, min_epsilon)

        if done:
            ne += 1
            log_returns.append(episode_reward)
            log_loss.append(np.mean(log_minibatchloss))
            log_eplen.append(ep_len)
            log_epsilon.append(epsilon)
            log_out.append(int(info['out_of_range']))
            log_success.append(int(info['success']))
            log_collisions.append(num_collisions)
            log_mean_returns.append(np.mean(log_returns[-log_freq:]))
            log_mean_loss.append(np.mean(log_loss[-log_freq:]))
            log_mean_eplen.append(np.mean(log_eplen[-log_freq:]))
            log_mean_out.append(np.mean(log_out[-log_freq:]))
            log_mean_success.append(np.mean(log_success[-log_freq:]))
            log_mean_collisions.append(np.mean(log_collisions[-log_freq:]))

            if ne%log_freq==0:
                print()
                print("{} episodes. ({}/{} steps)".format(ne, t_step, total_steps))
                print("Mean loss: {0:.6f}".format(log_mean_loss[-1]))
                print("Mean reward: {0:.2f}".format(log_mean_returns[-1]))
                # print("Ep reward: {}".format(log_returns[-1]))
                print("Ep length: {}".format(log_mean_eplen[-1]))
                print("Epsilon: {}".format(epsilon))

                axes[0][0].plot(log_loss, color='#ff7f00', linewidth=0.5)
                axes[1][0].plot(log_returns, color='#60c7ff', linewidth=0.5)
                axes[2][0].plot(log_eplen, color='#83dcb7', linewidth=0.5)
                axes[2][1].plot(log_collisions, color='#ff33cc', linewidth=0.5)

                axes[0][0].plot(log_mean_loss, color='red')
                axes[1][0].plot(log_mean_returns, color='blue')
                axes[2][0].plot(log_mean_eplen, color='green')
                axes[0][1].plot(log_mean_success, color='red')
                axes[1][1].plot(log_mean_out, color='black')
                axes[2][1].plot(log_mean_collisions, color='#663399')

                f.canvas.draw()
                plt.savefig('results/graph/%s.png'%savename)

                numpy_log = np.array([
                    log_returns, #0
                    log_loss, #1
                    log_eplen, #2
                    log_epsilon, #3
                    log_success, #4
                    log_collisions, #5
                    log_mean_returns, #6
                    log_mean_loss, #7
                    log_mean_eplen, #8
                    log_mean_success, #9
                    log_mean_collisions, #10
                    log_mean_out #11
                    ])
                np.save('results/board/%s' %savename, numpy_log)
                
                if np.mean(log_returns[-log_freq:])>max_rewards:
                    max_rewards = np.mean(log_returns[-log_freq:])
                    torch.save(Q.state_dict(), 'results/models/%s.pth'%savename)
                    print("Max rewards! saving the model.")

            episode_reward = 0
            log_minibatchloss = []
            state = env.reset()
            ep_len = 0
            num_collisions = 0


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--num_blocks", default=1, type=int)
    parser.add_argument("--dist", default=0.05, type=float)
    parser.add_argument("--max_steps", default=20, type=int)
    parser.add_argument("--camera_height", default=96, type=int)
    parser.add_argument("--camera_width", default=96, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--bs", default=64, type=int)
    parser.add_argument("--buff_size", default=1e4, type=int)
    parser.add_argument("--total_steps", default=2e5, type=int)
    parser.add_argument("--learn_start", default=2000, type=int)
    parser.add_argument("--update_freq", default=500, type=int)
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--reward", default="binary", type=str)
    args = parser.parse_args()

    # env configuration #
    render = args.render
    task = 2
    num_blocks = args.num_blocks
    mov_dist = args.dist
    max_steps = args.max_steps
    camera_height = args.camera_height
    camera_width = args.camera_width
    reward_type = args.reward

    env = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
            control_freq=5, data_format='NCHW', xml_ver=0)
    env = pushpixel_env(env, num_blocks=num_blocks, mov_dist=mov_dist, max_steps=max_steps, task=task,\
                        reward_type = reward_type)

    # learning configuration #
    learning_rate = args.lr
    batch_size = args.bs 
    buff_size = args.buff_size
    total_steps = int(args.total_steps)
    learn_start = args.learn_start
    update_freq = args.update_freq
    log_freq = args.log_freq
    double = True #args.double

    now = datetime.datetime.now()
    savename = "DQN_%s"%(now.strftime("%m%d_%H%M"))
    learning(env, savename, 8, learning_rate, batch_size, buff_size, total_steps, \
            learn_start, update_freq, log_freq, double)
