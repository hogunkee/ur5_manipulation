import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../ur5_mujoco'))
from pushpixel_env import *

import torch
import torch.nn as nn
import argparse
import json

import datetime


from replay_buffer import ReplayBuffer, PER
from matplotlib import pyplot as plt

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

crop_min = 19 #11 #13
crop_max = 78 #54 #52

def combine_batch(minibatch, data):
    combined = []
    for i in range(len(minibatch)):
        combined.append(torch.cat([minibatch[i], data[i].unsqueeze(0)]))
    return combined

def get_action(env, fc_qnet, state, epsilon, pre_action=None, with_q=False):
    if np.random.random() < epsilon:
        action = [np.random.randint(16,48), np.random.randint(15,49), np.random.randint(env.num_bins)]
        # action = [np.random.randint(env.env.camera_height), np.random.randint(env.env.camera_width), np.random.randint(env.num_bins)]
        if with_q:
            if env.task == 0:
                state_im = torch.tensor([state[0]]).type(dtype)
                q_value = fc_qnet(state_im, True)
                q_raw = q_value[0].detach().cpu().numpy()
                q = np.zeros_like(q_raw)
                q[:, crop_min:crop_max, crop_min:crop_max] = q_raw[:, crop_min:crop_max, crop_min:crop_max]
            else:
                state_im = torch.tensor([state[0]]).type(dtype)
                goal_im = torch.tensor([state[1]]).type(dtype)
                state_goal = torch.cat((state_im, goal_im), 1)
                q_value = fc_qnet(state_goal, True)
                q_raw = q_value[0].detach().cpu().numpy()
                q = np.zeros_like(q_raw)
                q[:, crop_min:crop_max, crop_min:crop_max] = q_raw[:, crop_min:crop_max, crop_min:crop_max]
    else:
        if env.task==0:
            state_im = torch.tensor([state[0]]).type(dtype)
            q_value = fc_qnet(state_im, True)
            q_raw = q_value[0].detach().cpu().numpy()
            q = np.zeros_like(q_raw)
            q[:, crop_min:crop_max, crop_min:crop_max] = q_raw[:, crop_min:crop_max, crop_min:crop_max]
            # avoid redundant motion #
            if pre_action is not None:
                q[pre_action[2], pre_action[0], pre_action[1]] = q.min()
            # image coordinate #
            aidx_x = q.max(0).max(1).argmax()
            aidx_y = q.max(0).max(0).argmax()
            aidx_th = q.argmax(0)[aidx_x, aidx_y]
            action = [aidx_x, aidx_y, aidx_th]
        else:
            state_im = torch.tensor([state[0]]).type(dtype)
            goal_im = torch.tensor([state[1]]).type(dtype)
            state_goal = torch.cat((state_im, goal_im), 1)
            q_value = fc_qnet(state_goal, True)
            q_raw = q_value[0].detach().cpu().numpy()
            q = np.zeros_like(q_raw)
            q[:, crop_min:crop_max, crop_min:crop_max] = q_raw[:, crop_min:crop_max, crop_min:crop_max]
            # avoid redundant motion #
            if pre_action is not None:
                q[pre_action[2], pre_action[0], pre_action[1]] = q.min()
            # image coordinate #
            aidx_x = q.max(0).max(1).argmax()
            aidx_y = q.max(0).max(0).argmax()
            aidx_th = q.argmax(0)[aidx_x, aidx_y]
            action = [aidx_x, aidx_y, aidx_th]

    if with_q:
        return action, q
    else:
        return action


def evaluate(env, n_actions=8, model_path='', num_trials=10, visualize_q=False):
    FCQ = FC_QNet(n_actions, env.task).type(dtype)
    print('Loading trained model: {}'.format(model_path))
    FCQ.load_state_dict(torch.load(model_path))

    ne = 0
    ep_len = 0
    episode_reward = 0
    log_returns = []
    log_eplen = []
    log_success = []

    state = env.reset()
    pre_action = None
    if visualize_q:
        plt.show()
        fig = plt.figure()
        if env.task==1:
            ax0 = fig.add_subplot(131)
            ax1 = fig.add_subplot(132)
            ax2 = fig.add_subplot(133)
        else:
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

        s0 = deepcopy(state[0]).transpose([1,2,0])
        if env.task==1:
            s1 = deepcopy(state[1]).transpose([1,2,0])
            im0 = ax0.imshow(s1)
        im = ax1.imshow(s0)
        im2 = ax2.imshow(np.zeros_like(s0))
        plt.show(block=False)
        fig.canvas.draw()
        fig.canvas.draw()

    while ne < num_trials:
        action, q_map = get_action(env, FCQ, state, epsilon=0.0, pre_action=pre_action, with_q=True)
        if visualize_q:
            s0 = deepcopy(state[0]).transpose([1, 2, 0])
            if env.task == 1:
                s1 = deepcopy(state[1]).transpose([1, 2, 0])
                im0 = ax0.imshow(s1)
            s0[action[0], action[1]] = [1, 0, 0]
            # q_map = q_map[0]
            q_map = q_map.transpose([1,2,0]).max(2)
            # q_map[action[0], action[1]] = 1.5
            im = ax1.imshow(s0)
            im2 = ax2.imshow(q_map/q_map.max())
            fig.canvas.draw()

        next_state, reward, done, info = env.step(action)
        episode_reward += reward

        ep_len += 1
        state = next_state
        pre_action = action

        if done:
            ne += 1
            log_returns.append(episode_reward)
            log_eplen.append(ep_len)
            log_success.append(int(info['success']))

            print()
            print("{} episodes.".format(ne))
            print("Ep reward: {}".format(log_returns[-1]))
            print("Ep length: {}".format(log_eplen[-1]))
            print("Success rate: {}% ({}/{})".format(100*np.mean(log_success), np.sum(log_success), len(log_success)))

            state = env.reset()
            pre_action = None
            ep_len = 0
            episode_reward = 0
    print()
    print("="*80)
    print("Evaluation Done.")
    # print("Rewards: {}".format(log_returns))
    print("Mean reward: {0:.2f}".format(np.mean(log_returns)))
    print("Mean episode length: {}".format(np.mean(log_eplen)))
    print("Success rate: {}".format(100*np.mean(log_success)))


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
        double=True,
        per=True,
        her=True,
        visualize_q=False,
        goal_type='circle'
        ):

    FCQ = FC_QNet(n_actions, env.task).type(dtype)
    FCQ_target = FC_QNet(n_actions, env.task).type(dtype)
    FCQ_target.load_state_dict(FCQ.state_dict())

    criterion = nn.SmoothL1Loss(reduction=None).type(dtype)
    # criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(FCQ.parameters(), lr=learning_rate, momentum=0.9, weight_decay=2e-5)
    # optimizer = torch.optim.Adam(FCQ.parameters(), lr=learning_rate)

    if per:
        replay_buffer = PER([3, env.env.camera_height, env.env.camera_width], 1, \
                save_goal=(env.task==1), save_gripper=False, max_size=int(buff_size))
    else:
        replay_buffer = ReplayBuffer([3, env.env.camera_height, env.env.camera_width], 1, \
                 save_goal=(env.task == 1), save_gripper=False, max_size=int(buff_size))

    model_parameters = filter(lambda p: p.requires_grad, FCQ.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("# of params: %d"%params)

    def calculate_loss_pixel(minibatch, gamma=0.5):
        actions = minibatch[2].type(torch.long)
        rewards = minibatch[3]
        not_done = minibatch[4]

        state_im = minibatch[0]
        next_state_im = minibatch[1]
        if env.task==0:
            state = state_im
            next_state = next_state_im
        else:
            goal_im = minibatch[5]
            state = torch.cat((state_im, goal_im), 1)
            next_state = torch.cat((next_state_im, goal_im), 1)

        next_q = FCQ_target(next_state, True)
        next_q_max = next_q[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]].max(1, True)[0]
        y_target = rewards + gamma * not_done * next_q_max

        q_values = FCQ(state, True)
        pred = q_values[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]
        pred = pred.view(-1, 1)

        loss = criterion(y_target, pred)
        error = torch.abs(pred - y_target)
        return loss, error

    def calculate_loss_double(minibatch, gamma=0.5):
        actions = minibatch[2].type(torch.long)
        rewards = minibatch[3]
        not_done = minibatch[4]

        state_im = minibatch[0]
        next_state_im = minibatch[1]
        if env.task==0:
            state = state_im
            next_state = next_state_im
        else:
            goal_im = minibatch[5]
            state = torch.cat((state_im, goal_im), 1)
            next_state = torch.cat((next_state_im, goal_im), 1)

        def get_a_prime():
            next_q = FCQ(next_state, True)
            next_q_chosen = next_q[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]]
            _, a_prime = next_q_chosen.max(1, True)
            return a_prime
        a_prime = get_a_prime()

        next_q_target = FCQ_target(next_state, True)
        next_q_target_chosen = next_q_target[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]]
        q_target_s_a_prime = next_q_target_chosen.gather(1, a_prime)
        y_target = rewards + gamma * not_done * q_target_s_a_prime

        q_values = FCQ(state, True)
        pred = q_values[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]
        pred = pred.view(-1, 1)

        loss = criterion(y_target, pred)
        error = torch.abs(pred - y_target)
        return loss, error

    def sample_her_transitions(info, next_state, num_samples=1):
        _info = deepcopy(info)
        move_threshold = 0.005
        range_x = env.block_range_x
        range_y = env.block_range_y
        success_threshold = env.threshold

        pre_poses = info['pre_poses']
        poses = info['poses']
        pos_diff = np.linalg.norm(poses - pre_poses, axis=1)
        if np.linalg.norm(poses - pre_poses) < move_threshold:
            return

        if goal_type=='circle':
            goal_image = deepcopy(env.background_img)
            for i in range(env.num_blocks):
                if pos_diff[i] < move_threshold:
                    continue
                ## 1. archived goal ##
                direction = poses[i] - pre_poses[i]
                direction /= np.linalg.norm(direction)
                archived_goal = pre_poses[i] + np.random.uniform(0.1, 0.2) * direction
                ## clipping goal pose ##
                x, y = archived_goal
                x = np.max((x, range_x[0]))
                x = np.min((x, range_x[1]))
                y = np.max((y, range_y[0]))
                y = np.min((y, range_y[1]))
                archived_goal = np.array([x, y])
                _info['goals'][i] = archived_goal
            ## generate goal image ##
            for i in range(env.num_blocks):
                cv2.circle(goal_image, env.pos2pixel(*_info['goals'][i]), 1, env.colors[i], -1)
            goal_image = np.transpose(goal_image, [2, 0, 1])

        elif goal_type=='block':
            for i in range(env.num_blocks):
                if pos_diff[i] < move_threshold:
                    continue
                x, y = poses[i]
                _info['goals'][i] = np.array([x, y])
            goal_image = deepcopy(next_state[0])

        ## recompute reward  ##
        reward_recompute, done_recompute = env.get_reward(_info)

        return reward_recompute, goal_image, done_recompute


    if double:
        calculate_loss = calculate_loss_double
    else:
        calculate_loss = calculate_loss_pixel #calculate_loss_origin

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

    epsilon = 0.5 #1.0
    start_epsilon = 0.5
    min_epsilon = 0.1
    epsilon_decay = 0.98
    episode_reward = 0.0
    max_rewards = -1000
    ep_len = 0
    ne = 0
    t_step = 0
    num_collisions = 0

    state = env.reset()
    pre_action = None

    if visualize_q:
        fig = plt.figure()
        if env.task==1:
            ax0 = fig.add_subplot(131)
            ax1 = fig.add_subplot(132)
            ax2 = fig.add_subplot(133)
        else:
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

        s0 = deepcopy(state[0]).transpose([1,2,0])
        if env.task==1:
            s1 = deepcopy(state[1]).transpose([1, 2, 0])
            im0 = ax0.imshow(s1)
        im = ax1.imshow(s0)
        im2 = ax2.imshow(np.zeros_like(s0))
        plt.show(block=False)
        fig.canvas.draw()
        fig.canvas.draw()

    while t_step < total_steps:
        action, q_map = get_action(env, FCQ, state, epsilon=epsilon, pre_action=pre_action, with_q=True)
        if visualize_q:
            s0 = deepcopy(state[0]).transpose([1, 2, 0])
            if env.task == 1:
                s1 = deepcopy(state[1]).transpose([1, 2, 0])
                im0 = ax0.imshow(s1)
            s0[action[0], action[1]] = [1, 0, 0]
            # q_map = q_map[0]
            q_map = q_map.transpose([1,2,0]).max(2)
            im = ax1.imshow(s0)
            im2 = ax2.imshow(q_map/q_map.max())
            print('min_q:', q_map.min(), '/ max_q:', q_map.max())
            fig.canvas.draw()

        next_state, reward, done, info = env.step(action)
        episode_reward += reward

        ## save transition to the replay buffer ##
        if per:
            if env.task == 0:
                state_im = torch.tensor([state[0]]).type(dtype)
                q_value = FCQ(state_im, True)[0].data
                q_target = FCQ_target(state_im, True)[0].data

                old_val = q_value[action[2], action[0], action[1]]
                if done:
                    target_val = reward
                else:
                    gamma = 0.5
                    target_val = reward + gamma * torch.max(q_target)
                error = abs(old_val - target_val).data.detach().cpu().numpy()
                replay_buffer.add(error, [state[0], 0.0], action, [next_state[0], 0.0], reward, done)
            else:
                state_im = torch.tensor([state[0]]).type(dtype)
                goal_im = torch.tensor([state[1]]).type(dtype)
                state_goal = torch.cat((state_im, goal_im), 1)
                q_value = FCQ(state_goal, True)[0].data
                q_target = FCQ_target(state_goal, True)[0].data

                old_val = q_value[action[2], action[0], action[1]]
                if done:
                    target_val = reward
                else:
                    gamma = 0.5
                    target_val = reward + gamma * torch.max(q_target)
                error = abs(old_val - target_val).data.detach().cpu().numpy()
                replay_buffer.add(error, [state[0], 0.0], action, [next_state[0], 0.0], reward, done, state[1])

        else:
            if env.task==0:
                replay_buffer.add([state[0], 0.0], action, [next_state[0], 0.0], reward, done)
            else:
                replay_buffer.add([state[0], 0.0], action, [next_state[0], 0.0], reward, done, state[1])
        ## HER ##
        if her and not done and env.task==1:
            her_sample = sample_her_transitions(info, next_state, num_samples=1)
            if her_sample is None:
                pass
            else:
                reward_re, goal_image, done_re = her_sample
                if per:
                    state_im = torch.tensor([state[0]]).type(dtype)
                    goal_im = torch.tensor([goal_image]).type(dtype) # replaced goal
                    state_goal = torch.cat((state_im, goal_im), 1)
                    q_value = FCQ(state_goal, True)[0].data
                    q_target = FCQ_target(state_goal, True)[0].data

                    old_val = q_value[action[2], action[0], action[1]]
                    if done_re: # replaced done & reward
                        target_val = reward_re
                    else:
                        gamma = 0.5
                        target_val = reward_re + gamma * torch.max(q_target)
                    error = abs(old_val - target_val).data.detach().cpu().numpy()
                    replay_buffer.add(error, [state[0], 0.0], action, [next_state[0], 0.0], reward_re, done_re, goal_image)
                else:
                    replay_buffer.add([state[0], 0.0], action, [next_state[0], 0.0], reward_re, done_re, goal_image)

        if t_step < learn_start:
            if done:
                state = env.reset()
                pre_action = None
                episode_reward = 0
            else:
                state = next_state
                pre_action = action
            learn_start -= 1
            if learn_start==0:
                epsilon = start_epsilon
            continue

        ## sample from replay buff & update networks ##
        data = [
                torch.FloatTensor(state[0]).type(dtype),
                torch.FloatTensor(next_state[0]).type(dtype),
                torch.FloatTensor(action).type(dtype),
                torch.FloatTensor([reward]).type(dtype),
                torch.FloatTensor([1 - done]).type(dtype),
                ]
        if task==1:
            data.append(torch.FloatTensor(state[1]).type(dtype))
        if per:
            minibatch, idxs, is_weights = replay_buffer.sample(batch_size-1)
            combined_minibatch = combine_batch(minibatch, data)
            loss, error = calculate_loss(combined_minibatch)
            errors = error.data.detach().cpu().numpy()[:-1]
            # update priority
            for i in range(batch_size-1):
                idx = idxs[i]
                replay_buffer.update(idx, errors[i])
        else:
            minibatch = replay_buffer.sample(batch_size-1)
            combined_minibatch = combine_batch(minibatch, data)
            loss, _ = calculate_loss(combined_minibatch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        log_minibatchloss.append(loss.data.detach().cpu().numpy())

        state = next_state
        pre_action = action
        ep_len += 1
        t_step += 1
        num_collisions += int(info['collision'])

        if t_step % update_freq == 0:
            FCQ_target.load_state_dict(FCQ.state_dict())
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

            if ne % log_freq == 0:
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
                # plt.pause(0.001)
                plt.savefig('results/graph/%s.png' % savename)
                # plt.close()

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

                if np.mean(log_returns[-log_freq:]) > max_rewards:
                    max_rewards = np.mean(log_returns[-log_freq:])
                    torch.save(FCQ.state_dict(), 'results/models/%s.pth' % savename)
                    print("Max rewards! saving the model.")

            episode_reward = 0.
            log_minibatchloss = []
            state = env.reset()
            pre_action = None
            ep_len = 0
            num_collisions = 0

    print('Training finished.')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--task", default=1, type=int)
    parser.add_argument("--num_blocks", default=1, type=int)
    parser.add_argument("--dist", default=0.08, type=float)
    parser.add_argument("--max_steps", default=20, type=int)
    parser.add_argument("--camera_height", default=96, type=int)
    parser.add_argument("--camera_width", default=96, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--bs", default=8, type=int)
    parser.add_argument("--buff_size", default=1e4, type=float)
    parser.add_argument("--total_steps", default=2e5, type=float)
    parser.add_argument("--learn_start", default=2e3, type=float)
    parser.add_argument("--update_freq", default=500, type=int)
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--per", action="store_true")
    parser.add_argument("--her", action="store_true")
    parser.add_argument("--reward", default="binary", type=str)
    parser.add_argument("--goal", default="circle", type=str)
    parser.add_argument("--fcn_ver", default=1, type=int)
    ## Evaluate ##
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--model_path", default="FCDQN_reach_0412_1714.pth", type=str)
    parser.add_argument("--num_trials", default=50, type=int)
    parser.add_argument("--show_q", action="store_true")
    args = parser.parse_args()

    # env configuration #
    render = args.render
    task = args.task
    num_blocks = args.num_blocks
    mov_dist = args.dist
    max_steps = args.max_steps
    camera_height = args.camera_height
    camera_width = args.camera_width
    reward_type = args.reward
    goal_type = args.goal

    # evaluate configuration #
    evaluation = args.evaluate
    model_path = os.path.join("results/models/", args.model_path)
    num_trials = args.num_trials
    visualize_q = args.show_q
    if visualize_q:
        render = True

    now = datetime.datetime.now()
    if task == 0:
        savename = "FCDQN_reach_%s" % (now.strftime("%m%d_%H%M"))
    elif task == 1:
        savename = "FCDQN_%s" % (now.strftime("%m%d_%H%M"))
    if not evaluation:
        if not os.path.exists("results/config/"):
            os.makedirs("results/config/")
        with open("results/config/%s.json" % savename, 'w') as cf:
            json.dump(args.__dict__, cf, indent=2)

    env = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
            control_freq=5, data_format='NCHW', xml_ver=0)
    env = pushpixel_env(env, num_blocks=num_blocks, mov_dist=mov_dist, max_steps=max_steps, \
            task=task, reward_type=reward_type, goal_type=goal_type)

    # learning configuration #
    learning_rate = args.lr
    batch_size = args.bs 
    buff_size = int(args.buff_size)
    total_steps = int(args.total_steps)
    learn_start = int(args.learn_start)
    update_freq = args.update_freq
    log_freq = args.log_freq
    double = args.double
    per = args.per
    her = args.her
    fcn_ver = args.fcn_ver
    if fcn_ver==1:
        from models.fcn import FC_QNet
    elif fcn_ver==2:
        from models.fcn_upsample import FC_QNet
    elif fcn_ver==3:
        from models.fcn_v3 import FC_QNet
    else:
        exit()

    if evaluation:
        evaluate(env=env, n_actions=8, model_path=model_path, num_trials=num_trials, \
                 visualize_q=visualize_q)
    else:
        learning(env=env, savename=savename, n_actions=8, learning_rate=learning_rate, \
                batch_size=batch_size, buff_size=buff_size, total_steps=total_steps, \
                learn_start=learn_start, update_freq=update_freq, log_freq=log_freq, \
                double=double, her=her, per=per, visualize_q=visualize_q, goal_type=goal_type)
