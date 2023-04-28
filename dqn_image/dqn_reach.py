import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../ur5_mujoco'))
from reach_discrete_env import *

import torch
import torch.nn as nn
import argparse
import wandb

import datetime
from replay_buffer_1bpush import ReplayBuffer
from matplotlib import pyplot as plt

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def smoothing_log_same(log_data, log_freq):
    return np.concatenate([np.array([np.nan] * (log_freq-1)), np.convolve(log_data, np.ones(log_freq), 'valid') / log_freq])

def combine_batch(minibatch, data):
    try:
        combined = []
        if minibatch is None:
            for i in range(len(data)):
                combined.append(data[i].unsqueeze(0))
        else:
            for i in range(len(minibatch)):
                combined.append(torch.cat([minibatch[i], data[i].unsqueeze(0)]))
    except:
        print(i)
        print(data[i].shape)
        print(minibatch[i].shape)
        print(data[i])
        print(minibatch[i])
    return combined

def learning(env, 
             n_actions=8, 
             learning_rate=1e-4, 
             batch_size=64, 
             buff_size=1e4, 
             total_episodes=1e4,
             learn_start=1e4,
             update_freq=100,
             log_freq=100,
             double=True,
             wandb_off=False,
             model_type='both',
             ):

    print('='*40)
    print('{} learing starts.'.format(savename))
    print('='*40)

    if model_type=='image':
        from models.dqn_imageonly import QNet
    elif model_type=='feature':
        from models.dqn_feature import QNet
    else:
        from models.dqn import QNet

    Q = QNet(n_actions, 4).type(dtype)
    Q_target = QNet(n_actions, 4).type(dtype)
    Q_target.load_state_dict(Q.state_dict())
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(Q.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer([3, env.env.camera_height, env.env.camera_width], 4, \
                                max_size=int(buff_size))

    model_parameters = filter(lambda p: p.requires_grad, Q.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("# of params: %d"%params)

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
    log_success = []

    epsilon = 1.0
    start_epsilon = 0.5
    min_epsilon = 0.1
    epsilon_decay = 0.98
    max_success = 0.0
    st = time.time()

    count_steps = 0
    for ne in range(total_episodes):
        state = env.reset()
        episode_reward = 0.
        log_minibatchloss = []

        for t_step in range(env.max_steps):
            count_steps += 1
            action = get_action(Q, state, epsilon)

            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            replay_buffer.add(state, action, next_state, reward, done)

            if replay_buffer.size < learn_start:
                if done:
                    break
                else:
                    state = next_state
                    continue
            elif replay_buffer.size == learn_start:
                epsilon = start_epsilon
                count_steps = 0
                print("Training starts.")
                break

            ## sample from replay buff & update networks ##
            data = [
                    torch.FloatTensor(state[0]).cuda(),
                    torch.FloatTensor(state[1]).cuda(),
                    torch.FloatTensor(next_state[0]).cuda(),
                    torch.FloatTensor(next_state[1]).cuda(),
                    torch.FloatTensor([action]).cuda(),
                    torch.FloatTensor([reward]).cuda(),
                    torch.FloatTensor([1 - done]).cuda(),
                    ]
            minibatch = replay_buffer.sample(batch_size-1)
            combined_minibatch = combine_batch(minibatch, data)
            loss = calculate_loss(combined_minibatch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            log_minibatchloss.append(loss.data.detach().cpu().numpy())

            if done:
                break
            else:
                state = next_state

        if replay_buffer.size <= learn_start:
            continue

        ep_len = env.step_count
        log_returns.append(episode_reward)
        log_loss.append(np.mean(log_minibatchloss))
        log_eplen.append(ep_len)
        log_epsilon.append(epsilon)
        log_success.append(int(info['success']))

        eplog = {
                'Reward': episode_reward,
                'Loss': np.mean(log_minibatchloss),
                'EP Len': ep_len,
                'Epsilon': epsilon,
                'Success Rate': int(info['success']),
                }
        if not wandb_off:
            wandb.log(eplog, count_steps)

        if (ne+1) % log_freq == 0:
            log_mean_returns = smoothing_log_same(log_returns, log_freq)
            log_mean_loss = smoothing_log_same(log_loss, log_freq)
            log_mean_eplen = smoothing_log_same(log_eplen, log_freq)
            log_mean_success = smoothing_log_same(log_success, log_freq)

            et = time.time()
            now = datetime.datetime.now().strftime("%m/%d %H:%M")
            interval = str(datetime.timedelta(0, int(et-st)))
            st = et
            print(f"{now}({interval}) / ep{ne+1} ({count_steps} steps)", end=" / ")
            print(f"SR:{log_mean_success[-1]:.2f}", end="")
            print("/ Reward:{0:.2f}".format(log_mean_returns[-1]), end="")
            print(" / Loss:{0:.5f}".format(log_mean_loss[-1]), end="")
            print(" / Eplen:{0:.1f}".format(log_mean_eplen[-1]), end="")

            log_list = [
                    log_returns,  # 0
                    log_loss,  # 1
                    log_eplen,  # 2
                    log_epsilon,  # 3
                    log_success,  # 4
                    ]
            numpy_log = np.array(log_list, dtype=object)
            np.save('results/board/%s' %savename, numpy_log)

            if log_mean_success[-1] > max_success:
                max_success = log_mean_success[-1]
                torch.save(Q.state_dict(), 'results/models/%s.pth' % savename)
                print(" <- Highest SR. Saving the model.")
            else:
                print("")

        if ne % update_freq == 0:
            Q_target.load_state_dict(Q.state_dict())
            epsilon = max(epsilon_decay * epsilon, min_epsilon)

    print('Training finished.')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--dist", default=0.03, type=float)
    parser.add_argument("--max_steps", default=50, type=int)
    parser.add_argument("--camera_height", default=128, type=int)
    parser.add_argument("--camera_width", default=128, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--bs", default=64, type=int)
    parser.add_argument("--buff_size", default=2e3, type=int)
    parser.add_argument("--total_episodes", default=2e4, type=int)
    parser.add_argument("--learn_start", default=1e3, type=int)
    parser.add_argument("--update_freq", default=1000, type=int)
    parser.add_argument("--log_freq", default=50, type=int)
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--wandb_off", action="store_true")
    parser.add_argument("--model", default="both") # both / image / feature 
    args = parser.parse_args()

    # env configuration #
    render = args.render
    gpu = args.gpu
    mov_dist = args.dist
    max_steps = args.max_steps
    camera_height = args.camera_height
    camera_width = args.camera_width

    env = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
            control_freq=5, data_format='NCHW', xml_ver='1bpush', gpu=gpu)
    env = reachdiscrete_env(env, mov_dist=mov_dist, max_steps=max_steps)

    # learning configuration #
    learning_rate = args.lr
    batch_size = args.bs 
    buff_size = args.buff_size
    total_episodes = int(args.total_episodes)
    learn_start = args.learn_start
    update_freq = args.update_freq
    log_freq = args.log_freq
    double = True #args.double
    model_type = args.model

    # wandb model name #
    now = datetime.datetime.now()
    if model_type=='image':
        savename = "DQN_I_%s" % (now.strftime("%m%d_%H%M"))
        print("Training DQN-Image Only...")
        print("="*60)
    elif model_type=='feature':
        savename = "DQN_F_%s" % (now.strftime("%m%d_%H%M"))
        print("Training DQN-Feature Only...")
        print("="*60)
    else:
        savename = "DQN_%s" % (now.strftime("%m%d_%H%M"))
        print("Training DQN-Image+Feature...")
        print("="*60)
    log_name = savename
    if not os.path.exists("results/board/"):
        ok.makedirs("results/board/")
    if not os.path.exists("results/models/"):
        ok.makedirs("results/models/")
    if not os.path.exists("results/config/"):
        os.makedirs("results/config/")

    wandb_off = args.wandb_off
    if not wandb_off:
        wandb.init(project="DQN-Reach")
        wandb.run.name = log_name
        wandb.config.update(args)
        wandb.run.save()

    learning(env, 8, learning_rate, batch_size, buff_size, total_episodes, \
            learn_start, update_freq, log_freq, double, wandb_off=wandb_off, \
            model_type=model_type)
