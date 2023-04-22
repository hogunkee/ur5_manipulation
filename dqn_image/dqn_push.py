import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../ur5_mujoco'))
from push_discrete_env import *

import torch
import torch.nn as nn
import argparse

import datetime
from models.dqn import QNet
from replay_buffer_1bpush import ReplayBuffer
from matplotlib import pyplot as plt

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def learning(env, 
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

    Q = QNet(n_actions).type(dtype)
    Q_target = QNet(n_actions).type(dtype)
    Q_target.load_state_dict(Q.state_dict())
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(Q.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer([3, env.env.camera_height, env.env.camera_width], 6, \
                                max_size=int(buff_size))

    def get_action(q_net, state, epsilon):
        if np.random.random() < epsilon:
            action = np.random.randint(n_actions)
        else:
            state_im = torch.tensor([state[0]]).type(dtype)
            state = torch.tensor([state[1]]).type(dtype)
            q_value = q_net(state_im, state)
            action = np.argmax(q_value.detach().cpu().numpy())
        return action

    def calculate_loss_origin(minibatch, gamma=0.99):
        rewards = minibatch[5]
        not_done = minibatch[6]

        state_im = minibatch[0]
        state = minibatch[1]
        next_state_im = minibatch[2]
        next_state = minibatch[3]

        next_q = Q_target(next_state_im, next_state)
        next_q_max = next_q[torch.arange(batch_size), next_q.max(1)[1]]
        y_target = rewards + gamma * not_done * next_q_max
        #print('y target:', y_target)
        q_values = Q(state_im, state)
        actions = minibatch[4]
        max_q = q_values[torch.arange(batch_size), actions.type(torch.long)]
        #print('q:', max_q)
        loss = criterion(y_target, max_q)
        return loss

    def calculate_loss_double(minibatch, gamma=0.99):
        rewards = minibatch[5]
        not_done = minibatch[6]

        state_im = minibatch[0]
        state = minibatch[1]
        next_state_im = minibatch[2]
        next_state = minibatch[3]

        next_q = Q(next_state_im, next_state).detach()
        _, a_prime = next_q.max(1)

        q_target_next = Q_target(next_state_im, next_state).detach()
        q_target_s_a_prime = q_target_next.gather(1, a_prime.unsqueeze(1))
        q_target_s_a_prime = q_target_s_a_prime.squeeze()

        next_q_max = not_done * q_target_s_a_prime
        y_target = rewards + gamma * not_done * next_q_max
        #print('y target:', y_target)
        q_values = Q(state_im, state)
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
    episode_reward = 0
    log_minibatchloss = []

    log_mean_returns = []
    log_mean_loss = []
    log_mean_eplen = []

    done = False
    t_step = 0
    ridx = 0
    epsilon = 1.0
    min_epsilon = 0.01
    state = env.reset()
    print('reset.')
    ne = 0
    ep_len = 0
    max_rewards = -1000
    now = datetime.datetime.now()
    if not os.path.exists("results/graph/"):
        os.makedirs("results/graph/")
    if not os.path.exists("results/models/"):
        os.makedirs("results/models/")

    #plt.ion()
    plt.show(block=False)
    plt.rc('axes', labelsize=8)
    plt.rc('axes', labelsize=6)
    plt.rc('font', size=6)
    f, axes = plt.subplots(4, 1)
    f.set_figheight(10) #15
    f.set_figwidth(6) #10

    axes[0].set_title('Loss')
    axes[1].set_title('Episode Return')
    axes[2].set_title('Episode length')
    axes[3].set_title('Epsilon')


    while t_step<total_steps:
        action = get_action(Q, state, epsilon)
        next_state, reward, done, info = env.step(action)

        epsilon = max(0.999*epsilon, min_epsilon)

        episode_reward += reward

        replay_buffer.add(state, action, next_state, reward, done)

        if t_step<learn_start:
            if done:
                state = env.reset()
                print('reset.')
                episode_reward = 0
            else:
                state = next_state
            learn_start -= 1
            continue

        minibatch = replay_buffer.sample(batch_size)
        loss = calculate_loss(minibatch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_minibatchloss.append(loss.data.detach().cpu().numpy())
        if (t_step+1)%update_freq:
            Q_target.load_state_dict(Q.state_dict())

        t_step += 1
        ep_len += 1
        state = next_state

        if done:
            ne += 1
            log_returns.append(episode_reward)
            log_loss.append(np.mean(log_minibatchloss))
            log_eplen.append(ep_len)
            log_epsilon.append(epsilon)
            log_mean_returns.append(np.mean(log_returns[-log_freq:]))
            log_mean_loss.append(np.mean(log_loss[-log_freq:]))
            log_mean_eplen.append(np.mean(log_eplen[-log_freq:]))

            if ne%log_freq==0:
                print()
                print("{} episodes. ({}/{} steps)".format(ne, t_step, total_steps))
                print("Mean loss: {0:.6f}".format(log_mean_loss[-1]))
                print("Mean reward: {0:.2f}".format(log_mean_returns[-1]))
                #print("Ep reward: {}".format(log_returns[-1]))
                print("Ep length: {}".format(log_mean_eplen[-1]))
                print("Epsilon: {}".format(epsilon))

                axes[0].plot(log_loss, color='#ff7f00', linewidth=0.5)
                axes[1].plot(log_returns, color='#60c7ff', linewidth=0.5)
                axes[2].plot(log_eplen, color='#83dcb7', linewidth=0.5)
                axes[3].plot(log_epsilon, color='black')

                axes[0].plot(log_mean_loss, color='red')
                axes[1].plot(log_mean_returns, color='blue')
                axes[2].plot(log_mean_eplen, color='green')

                savename = "DQN_%s"%(now.strftime("%m%d_%H%M"))
                f.canvas.draw()
                plt.pause(0.001)
                plt.savefig('results/graph/%s.png'%savename)
                #plt.close()
                
                if np.mean(log_returns[-log_freq:])>max_rewards:
                    max_rewards = np.mean(log_returns[-log_freq:])
                    torch.save(Q.state_dict(), 'results/models/%s.pth'%savename)
                    print("Max rewards! saving the model.")

            episode_reward = 0
            log_minibatchloss = []
            state = env.reset()
            print('reset.')
            ep_len = 0


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--dist", default=0.04, type=float)
    parser.add_argument("--max_steps", default=100, type=int)
    parser.add_argument("--camera_height", default=128, type=int)
    parser.add_argument("--camera_width", default=128, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--bs", default=64, type=int)
    parser.add_argument("--buff_size", default=1e4, type=int)
    parser.add_argument("--total_steps", default=2e4, type=int)
    parser.add_argument("--learn_start", default=2e3, type=int)
    parser.add_argument("--update_freq", default=1000, type=int)
    parser.add_argument("--log_freq", default=10, type=int)
    parser.add_argument("--double", action="store_true")
    args = parser.parse_args()

    # env configuration #
    render = args.render
    mov_dist = args.dist
    max_steps = args.max_steps
    camera_height = args.camera_height
    camera_width = args.camera_width

    env = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
            control_freq=5, data_format='NCHW', xml_ver='1bpush')
    env = pushdiscrete_env(env, mov_dist=mov_dist, max_steps=max_steps)

    # learning configuration #
    learning_rate = args.lr
    batch_size = args.bs 
    buff_size = args.buff_size
    total_steps = int(args.total_steps)
    learn_start = args.learn_start
    update_freq = args.update_freq
    log_freq = args.log_freq
    double = True #args.double

    learning(env, 8, learning_rate, batch_size, buff_size, total_steps, \
            learn_start, update_freq, log_freq, double)
