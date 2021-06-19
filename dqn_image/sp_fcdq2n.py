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

crop_min = 9 #19 #11 #13
crop_max = 88 #78 #54 #52

def smoothing_log(log_data, log_freq):
    return np.convolve(log_data, np.ones(log_freq), 'valid') / log_freq

def combine_batch(minibatch, data):
    combined = []
    for i in range(len(minibatch)):
        combined.append(torch.cat([minibatch[i], data[i].unsqueeze(0)]))
    return combined

def get_action(env, fc_qnet, state, epsilon, pre_action=None, with_q=False, sample='sum', masking=''):
    if np.random.random() < epsilon:
        action = [np.random.randint(crop_min,crop_max), np.random.randint(crop_min,crop_max), np.random.randint(env.num_bins)]
        # action = [np.random.randint(env.env.camera_height), np.random.randint(env.env.camera_width), np.random.randint(env.num_bins)]
        if with_q:
            state_im = torch.tensor([state[0]]).type(dtype)
            goal_im = torch.tensor([state[1]]).type(dtype)
            state_goal = torch.cat((state_im, goal_im), 1)
            q_value = fc_qnet(state_goal, True)
            q_raw = q_value[0][0].detach().cpu().numpy()
            q = np.zeros_like(q_raw)
            q[:, crop_min:crop_max, crop_min:crop_max] = q_raw[:, crop_min:crop_max, crop_min:crop_max]

            q_next_raw = q_value[0][1].detach().cpu().numpy()
    else:
        state_im = torch.tensor([state[0]]).type(dtype)
        goal_im = torch.tensor([state[1]]).type(dtype)
        state_goal = torch.cat((state_im, goal_im), 1)
        q_value = fc_qnet(state_goal, True) # 1 x 2 x nb x 8 x 96 x 96
        q_raw = q_value[0][0].detach().cpu().numpy() # q_raw: nb x 8 x 96 x 96
        q_next_raw = q_value[0][1].detach().cpu().numpy() # q_next_raw: nb x 8 x 96 x 96

        q = np.zeros_like(q_raw[0])
        # summation of Q-values
        if sample=='sum':
            for o in range(env.num_blocks):
                q[:, crop_min:crop_max, crop_min:crop_max] += q_raw[o, :, crop_min:crop_max, crop_min:crop_max]
        # sampling with object-wise q_max
        elif sample=='choice':
            prob = []
            for o in range(env.num_blocks):
                prob.append(np.max([q_raw[o].max(), 0.1]))
            prob /= np.sum(prob)
            selected_obj = np.random.choice(env.num_blocks, 1, p=prob)[0]
            q[:, crop_min:crop_max, crop_min:crop_max] += q_raw[selected_obj, :, crop_min:crop_max, crop_min:crop_max]
        # sampling from uniform distribution
        elif sample=='uniform':
            prob = [1./env.num_blocks] * env.num_blocks
            selected_obj = np.random.choice(env.num_blocks, 1, p=prob)[0]
            q[:, crop_min:crop_max, crop_min:crop_max] += q_raw[selected_obj, :, crop_min:crop_max, crop_min:crop_max]
        # select maximum q
        elif sample=='max':
            q[:, crop_min:crop_max, crop_min:crop_max] += q_raw.max(0)[:, crop_min:crop_max, crop_min:crop_max]

        # constraints #
        mask = np.ones_like(q)
        if 'q' in masking:
            for o in range(env.num_blocks):
                if selected_obj==o:
                    continue
                mask[q_raw[o]<0] = 0.0
        if 'mean' in masking:
            for o in range(env.num_blocks):
                if selected_obj==o:
                    continue
                mask[(q_next_raw[o] - q_raw[o].mean())<0] = 0.0
        if 'max' in masking:
            for o in range(env.num_blocks):
                if selected_obj==o:
                    continue
                mask[(q_next_raw[o] - q_raw[o].max())<0] = 0.0
        q = np.multiply(q, mask)

        # avoid redundant motion #
        if pre_action is not None:
            q[pre_action[2], pre_action[0], pre_action[1]] = q.min()
        # image coordinate #
        aidx_x = q.max(0).max(1).argmax()
        aidx_y = q.max(0).max(0).argmax()
        aidx_th = q.argmax(0)[aidx_x, aidx_y]
        action = [aidx_x, aidx_y, aidx_th]


    if with_q:
        return action, q, q_raw
    else:
        return action


def evaluate(env, n_blocks=3, in_channel=6, model_path='', num_trials=10, visualize_q=False, sampling='choice', masking='max q'):
    FCQ = FC_Q2Net(8, in_channel, n_blocks).type(dtype)
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
        if sampling == 'sum':
            ax0 = fig.add_subplot(131)
            ax1 = fig.add_subplot(132)
            ax2 = fig.add_subplot(133)
        else:
            ax0 = fig.add_subplot(231)
            ax1 = fig.add_subplot(232)
            ax2 = fig.add_subplot(233)
            ax3 = fig.add_subplot(234)
            ax4 = fig.add_subplot(235)
            ax5 = fig.add_subplot(236)

        s0 = deepcopy(state[0]).transpose([1,2,0])
        if env.goal_type == 'pixel':
            s1 = np.zeros([env.env.camera_height, env.env.camera_width, 3])
            s1[:, :, :n_blocks] = state[1].transpose([1, 2, 0])
        else:
            s1 = deepcopy(state[1]).transpose([1, 2, 0])
        ax0.imshow(s1)
        ax1.imshow(s0)
        ax2.imshow(np.zeros_like(s0))
        ax3.imshow(np.zeros_like(s0))
        ax4.imshow(np.zeros_like(s0))
        ax5.imshow(np.zeros_like(s0))
        plt.show(block=False)
        fig.canvas.draw()
        fig.canvas.draw()

    while ne < num_trials:
        action, q_map, q_raw = get_action(env, FCQ, state, epsilon=0.0, pre_action=pre_action, with_q=True, sample=sampling, masking=masking)
        if visualize_q:
            s0 = deepcopy(state[0]).transpose([1, 2, 0])
            if env.goal_type == 'pixel':
                s1 = np.zeros([env.env.camera_height, env.env.camera_width, 3])
                s1[:, :, :n_blocks] = state[1].transpose([1, 2, 0])
            else:
                s1 = deepcopy(state[1]).transpose([1, 2, 0])
            ax0.imshow(s1)
            s0[action[0], action[1]] = [1, 0, 0]
            # q_map = q_map[0]
            q_map = q_map.transpose([1,2,0]).max(2)
            # print(q_map.max())
            # print(q_map.min())
            # q_map[action[0], action[1]] = 1.5
            ax1.imshow(s0)
            ax2.imshow(q_map, vmax=1.8, vmin=-0.2)
            if sampling != 'sum':
                q0 = q_raw[0].transpose([1,2,0]).max(2)
                q1 = q_raw[1].transpose([1, 2, 0]).max(2)
                ax3.imshow(q0, vmax=1.8, vmin=-0.2)
                ax4.imshow(q1, vmax=1.8, vmin=-0.2)
                if num_blocks==3:
                    q2 = q_raw[2].transpose([1, 2, 0]).max(2)
                    ax5.imshow(q2)
            fig.canvas.draw()

        next_state, rewards, done, info = env.step(action)
        episode_reward += np.sum(rewards)

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
        n_blocks=3,
        in_channel=6,
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
        next_v=True,
        visualize_q=False,
        goal_type='circle',
        sampling='uniform',
        masking=''
        ):

    FCQ = FC_Q2Net(8, in_channel, n_blocks).type(dtype)
    FCQ_target = FC_Q2Net(8, in_channel, n_blocks).type(dtype)
    FCQ_target.load_state_dict(FCQ.state_dict())

    criterion = nn.SmoothL1Loss(reduction=None).type(dtype)
    # criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(FCQ.parameters(), lr=learning_rate, momentum=0.9, weight_decay=2e-5)
    # optimizer = torch.optim.Adam(FCQ.parameters(), lr=learning_rate)

    if per:
        if goal_type=='pixel':
            goal_ch = n_blocks
        else:
            goal_ch = 3
        replay_buffer = PER([3, env.env.camera_height, env.env.camera_width], \
                    [goal_ch, env.env.camera_height, env.env.camera_width], 1, \
                    save_goal=True, save_gripper=False, max_size=int(buff_size),\
                    dim_reward=n_blocks)
    else:
        replay_buffer = ReplayBuffer([3, env.env.camera_height, env.env.camera_width], 1, \
                 save_goal=True, save_gripper=False, max_size=int(buff_size),\
                 dim_reward=n_blocks)

    model_parameters = filter(lambda p: p.requires_grad, FCQ.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("# of params: %d"%params)

    def calculate_loss_pixel(minibatch, gamma=0.5):
        state_im = minibatch[0]
        next_state_im = minibatch[1]
        actions = minibatch[2].type(torch.long)
        rewards = minibatch[3]
        not_done = minibatch[4]
        goal_im = minibatch[5]

        state = torch.cat((state_im, goal_im), 1)
        next_state = torch.cat((next_state_im, goal_im), 1)

        next_q = FCQ_target(next_state, True)
        q_values = FCQ(state, True)

        loss = []
        error = []
        for o in range(n_blocks):
            next_q_max = next_q[torch.arange(batch_size), 0, o, :, actions[:, 0], actions[:, 1]].max(1, True)[0]
            y_target = rewards[:, o].unsqueeze(1) + gamma * not_done * next_q_max

            pred = q_values[torch.arange(batch_size), 0, o, actions[:, 2], actions[:, 0], actions[:, 1]]
            pred = pred.view(-1, 1)

            loss.append(criterion(y_target, pred))
            error.append(torch.abs(pred - y_target))

        loss = torch.sum(torch.stack(loss))
        error = torch.sum(torch.stack(error), dim=0).view(-1)
        return loss, error

    def calculate_loss_double(minibatch, gamma=0.5):
        state_im = minibatch[0]
        next_state_im = minibatch[1]
        actions = minibatch[2].type(torch.long)
        rewards = minibatch[3]
        not_done = minibatch[4]
        goal_im = minibatch[5]

        state = torch.cat((state_im, goal_im), 1)
        next_state = torch.cat((next_state_im, goal_im), 1)

        next_q_target = FCQ_target(next_state, True)
        q_values = FCQ(state, True)
        next_q = FCQ(next_state, True)

        def get_a_prime(obj):
            next_q_chosen = next_q[torch.arange(batch_size), 0, obj, :, actions[:, 0], actions[:, 1]]
            _, a_prime = next_q_chosen.max(1, True)
            return a_prime

        loss = []
        error = []
        for o in range(n_blocks):
            a_prime = get_a_prime(o)
            next_q_target_chosen = next_q_target[torch.arange(batch_size), 0, o, :, actions[:, 0], actions[:, 1]]
            q_target_s_a_prime = next_q_target_chosen.gather(1, a_prime)
            y_target = rewards[:, o].unsqueeze(1) + gamma * not_done * q_target_s_a_prime

            pred = q_values[torch.arange(batch_size), 0, o, actions[:, 2], actions[:, 0], actions[:, 1]]
            pred = pred.view(-1, 1)

            loss.append(criterion(y_target, pred))
            error.append(torch.abs(pred - y_target))

        loss = torch.sum(torch.stack(loss))
        error = torch.sum(torch.stack(error), dim=0).view(-1)
        return loss, error

    def calculate_loss_next_v(minibatch, gamma=0.5):
        state_im = minibatch[0]
        next_state_im = minibatch[1]
        actions = minibatch[2].type(torch.long)
        rewards = minibatch[3]
        not_done = minibatch[4]
        goal_im = minibatch[5]

        state = torch.cat((state_im, goal_im), 1)
        next_state = torch.cat((next_state_im, goal_im), 1)

        next_q = FCQ_target(next_state, True) # bs x 2 x nb x 8 x h x w
        q_values = FCQ(state, True)

        loss = []
        error = []
        for o in range(n_blocks):
            y_target = next_q[torch.arange(batch_size), 0, o].mean([1,2,3])
            pred = q_values[torch.arange(batch_size), 1, o, actions[:, 2], actions[:, 0], actions[:, 1]]

            loss.append(criterion(y_target, pred))
            error.append(torch.abs(pred - y_target))

        loss = torch.sum(torch.stack(loss))
        error = torch.sum(torch.stack(error), dim=0)
        return loss, error

    def calculate_loss_next_q(minibatch, gamma=0.5):
        state_im = minibatch[0]
        next_state_im = minibatch[1]
        actions = minibatch[2].type(torch.long)
        rewards = minibatch[3]
        not_done = minibatch[4]
        goal_im = minibatch[5]

        state = torch.cat((state_im, goal_im), 1)
        next_state = torch.cat((next_state_im, goal_im), 1)

        next_q = FCQ_target(next_state, True) # bs x 2 x nb x 8 x h x w
        q_values = FCQ(state, True)

        loss = []
        error = []
        for o in range(n_blocks):
            y_target = next_q[torch.arange(batch_size), 0, o].max(1)[0].max(1)[0].max(1)[0]
            pred = q_values[torch.arange(batch_size), 1, o, actions[:, 2], actions[:, 0], actions[:, 1]]

            loss.append(criterion(y_target, pred))
            error.append(torch.abs(pred - y_target))

        loss = torch.sum(torch.stack(loss))
        error = torch.sum(torch.stack(error), dim=0)
        return loss, error

    def sample_her_transitions(info, next_state):
        _info = deepcopy(info)
        move_threshold = 0.005
        range_x = env.block_range_x
        range_y = env.block_range_y

        pre_poses = info['pre_poses']
        poses = info['poses']
        pos_diff = np.linalg.norm(poses - pre_poses, axis=1)
        if np.linalg.norm(poses - pre_poses) < move_threshold:
            return []

        if goal_type=='circle':
            goal_image = deepcopy(env.background_img)
            for i in range(n_blocks):
                if pos_diff[i] < move_threshold:
                    continue
                ## 1. archived goal ##
                archived_goal = poses[i]

                ## clipping goal pose ##
                x, y = archived_goal
                x = np.max((x, range_x[0]))
                x = np.min((x, range_x[1]))
                y = np.max((y, range_y[0]))
                y = np.min((y, range_y[1]))
                archived_goal = np.array([x, y])
                _info['goals'][i] = archived_goal
            _info['goal_flags'] = np.linalg.norm(_info['goals'] - _info['poses'], axis=1) < env.threshold
            ## generate goal image ##
            for i in range(n_blocks):
                if env.hide_goal and _info['goal_flags'][i]:
                    continue
                cv2.circle(goal_image, env.pos2pixel(*_info['goals'][i]), 1, env.colors[i], -1)
            goal_image = np.transpose(goal_image, [2, 0, 1])

        elif goal_type=='pixel':
            for i in range(n_blocks):
                if pos_diff[i] < move_threshold:
                    continue
                ## 1. archived goal ##
                archived_goal = poses[i]
                ## clipping goal pose ##
                x, y = archived_goal
                x = np.max((x, range_x[0]))
                x = np.min((x, range_x[1]))
                y = np.max((y, range_y[0]))
                y = np.min((y, range_y[1]))
                archived_goal = np.array([x, y])
                _info['goals'][i] = archived_goal
            _info['goal_flags'] = np.linalg.norm(_info['goals'] - _info['poses'], axis=1) < env.threshold
            ## generate goal image ##
            goal_ims = []
            for i in range(n_blocks):
                zero_array = np.zeros([env.env.camera_height, env.env.camera_width])
                if not (env.hide_goal and _info['goal_flags'][i]):
                    cv2.circle(zero_array, env.pos2pixel(*_info['goals'][i]), 1, 1, -1)
                goal_ims.append(zero_array)
            goal_image = np.concatenate(goal_ims)
            goal_image = goal_image.reshape([n_blocks, env.env.camera_height, env.env.camera_width])

        elif goal_type=='block':
            for i in range(n_blocks):
                if pos_diff[i] < move_threshold:
                    continue
                x, y = poses[i]
                _info['goals'][i] = np.array([x, y])
            goal_image = deepcopy(next_state[0])

        ## recompute reward  ##
        reward_recompute, done_recompute = env.get_reward(_info)

        return [[reward_recompute, goal_image, done_recompute]]


    def sample_ig_transitions(info, next_state, num_samples=1):
        move_threshold = 0.005
        range_x = env.block_range_x
        range_y = env.block_range_y
        success_threshold = env.threshold

        pre_poses = info['pre_poses']
        poses = info['poses']
        pos_diff = np.linalg.norm(poses - pre_poses, axis=1)
        if np.linalg.norm(poses - pre_poses) < move_threshold:
            return []

        transitions = []
        for s in range(num_samples):
            _info = deepcopy(info)
            if goal_type=='circle':
                goal_image = deepcopy(env.background_img)
                for i in range(n_blocks):
                    if pos_diff[i] < move_threshold:
                        continue
                    ## 1. archived goal ##
                    gx = np.random.uniform(*range_x)
                    gy = np.random.uniform(*range_y)
                    archived_goal = np.array([gx, gy])

                    _info['goals'][i] = archived_goal
                _info['goal_flags'] = np.linalg.norm(_info['goals'] - _info['poses'], axis=1) < env.threshold
                ## generate goal image ##
                for i in range(n_blocks):
                    if env.hide_goal and _info['goal_flags'][i]:
                        continue
                    cv2.circle(goal_image, env.pos2pixel(*_info['goals'][i]), 1, env.colors[i], -1)
                goal_image = np.transpose(goal_image, [2, 0, 1])

            elif goal_type=='pixel':
                for i in range(n_blocks):
                    if pos_diff[i] < move_threshold:
                        continue
                    ## 1. archived goal ##
                    gx = np.random.uniform(*range_x)
                    gy = np.random.uniform(*range_y)
                    archived_goal = np.array([gx, gy])

                    _info['goals'][i] = archived_goal
                _info['goal_flags'] = np.linalg.norm(_info['goals'] - _info['poses'], axis=1) < env.threshold
                ## generate goal image ##
                goal_ims = []
                for i in range(n_blocks):
                    zero_array = np.zeros([env.env.camera_height, env.env.camera_width])
                    if not (env.hide_goal and _info['goal_flags'][i]):
                        cv2.circle(zero_array, env.pos2pixel(*_info['goals'][i]), 1, 1, -1)
                    goal_ims.append(zero_array)
                goal_image = np.concatenate(goal_ims)
                goal_image = goal_image.reshape([n_blocks, env.env.camera_height, env.env.camera_width])

            elif goal_type=='block':
                pass

            ## recompute reward  ##
            reward_recompute, done_recompute = env.get_reward(_info)
            transitions.append([reward_recompute, goal_image, done_recompute])

        return transitions


    if double:
        calculate_loss = calculate_loss_double
    else:
        calculate_loss = calculate_loss_pixel #calculate_loss_origin

    if next_v:
        calculate_loss_next = calculate_loss_next_v
    else:
        calculate_loss_next = calculate_loss_next_q

    log_returns = []
    log_loss = []
    log_loss_q = []
    log_loss_next = []
    log_eplen = []
    log_epsilon = []
    log_out = []
    log_success = []
    log_collisions = []
    log_minibatchloss = []
    log_minibatchloss_q = []
    log_minibatchloss_next = []

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
    max_success = 0.0
    ep_len = 0
    ne = 0
    t_step = 0
    num_collisions = 0

    state = env.reset()
    pre_action = None

    if visualize_q:
        fig = plt.figure()
        ax0 = fig.add_subplot(131)
        ax1 = fig.add_subplot(132)
        ax2 = fig.add_subplot(133)

        s0 = deepcopy(state[0]).transpose([1,2,0])
        if env.goal_type=='pixel':
            s1 = np.zeros([env.env.camera_height, env.env.camera_width, 3])
            s1[:,:,:n_blocks] = state[1].transpose([1,2,0])
        else:
            s1 = deepcopy(state[1]).transpose([1, 2, 0])
        im0 = ax0.imshow(s1)
        im = ax1.imshow(s0)
        im2 = ax2.imshow(np.zeros_like(s0))
        plt.show(block=False)
        fig.canvas.draw()
        fig.canvas.draw()

    while t_step < total_steps:
        action, q_map, _ = get_action(env, FCQ, state, epsilon=epsilon, pre_action=pre_action, with_q=True, sample=sampling, masking=masking)
        if visualize_q:
            s0 = deepcopy(state[0]).transpose([1, 2, 0])
            if env.goal_type == 'pixel':
                s1 = np.zeros([env.env.camera_height, env.env.camera_width, 3])
                s1[:, :, :n_blocks] = state[1].transpose([1, 2, 0])
            else:
                s1 = deepcopy(state[1]).transpose([1, 2, 0])
            im0 = ax0.imshow(s1)
            s0[action[0], action[1]] = [1, 0, 0]
            # q_map = q_map[0]
            q_map = q_map.transpose([1,2,0]).max(2)
            im = ax1.imshow(s0)
            im2 = ax2.imshow(q_map/q_map.max())
            print('min_q:', q_map.min(), '/ max_q:', q_map.max())
            fig.canvas.draw()

        next_state, rewards, done, info = env.step(action)
        episode_reward += np.sum(rewards)

        ## save transition to the replay buffer ##
        if per:
            state_im = torch.tensor([state[0]]).type(dtype)
            goal_im = torch.tensor([state[1]]).type(dtype)
            state_goal = torch.cat((state_im, goal_im), 1)
            next_state_im = torch.tensor([next_state[0]]).type(dtype)
            next_state_goal = torch.cat((next_state_im, goal_im), 1)
            q_value = FCQ(state_goal, True)[0].data
            next_q_value = FCQ(next_state_goal, True)[0].data
            next_q_target = FCQ_target(next_state_goal, True)[0].data

            error = 0.0
            # Bellman error
            for o in range(n_blocks):
                if done:
                    target_val = rewards[o]
                else:
                    gamma = 0.5
                    if double:
                        next_q_chosen = next_q_value[0, o, :, action[0], action[1]]
                        _, a_prime = next_q_chosen.max(0, True)
                        q_target_s_a_prime = next_q_target[0, o, a_prime, action[0], action[1]]
                        target_val = rewards[o] + gamma * q_target_s_a_prime
                    else:
                        target_val = rewards[o] + gamma * torch.max(next_q_target[0, o])
                old_val = q_value[0, o, action[2], action[0], action[1]]
                error += abs(old_val - target_val).data.detach().cpu().numpy()
            # Next q error
            for o in range(n_blocks):
                if next_v:
                    target_next = next_q_value[0, o].mean()
                else:
                    target_next = next_q_value[0, o].max()
                pred_next = q_value[1, o, action[2], action[0], action[1]]
                error += abs(pred_next - target_next).data.detach().cpu().numpy()
            replay_buffer.add(error, [state[0], 0.0], action, [next_state[0], 0.0], rewards, done, state[1])

        else:
            replay_buffer.add([state[0], 0.0], action, [next_state[0], 0.0], rewards, done, state[1])
        ## HER ##
        if her and not done:
            her_sample = sample_her_transitions(info, next_state)
            ig_samples = sample_ig_transitions(info, next_state, num_samples=3)
            samples = her_sample + ig_samples
            for sample in samples:
                rewards_re, goal_image, done_re = sample
                if per:
                    goal_im_re = torch.tensor([goal_image]).type(dtype) # replaced goal
                    state_goal = torch.cat((state_im, goal_im_re), 1)
                    next_state_goal = torch.cat((next_state_im, goal_im_re), 1)

                    q_value = FCQ(state_goal, True)[0].data
                    next_q_value = FCQ(next_state_goal, True)[0].data
                    next_q_target = FCQ_target(next_state_goal, True)[0].data

                    error = 0.0
                    # Bellman error
                    for o in range(n_blocks):
                        if done_re: # replaced done & reward
                            target_val = rewards_re[o]
                        else:
                            gamma = 0.5
                            if double:
                                next_q_chosen = next_q_value[0, o, :, action[0], action[1]]
                                _, a_prime = next_q_chosen.max(0, True)
                                q_target_s_a_prime = next_q_target[0, o, a_prime, action[0], action[1]]
                                target_val = rewards_re[o] + gamma * q_target_s_a_prime
                            else:
                                target_val = rewards_re[o] + gamma * torch.max(next_q_target[0, o])
                        old_val = q_value[0, o, action[2], action[0], action[1]]
                        error += abs(old_val - target_val).data.detach().cpu().numpy()
                    # Next q error
                    for o in range(n_blocks):
                        if next_v:
                            target_next = next_q_value[0, o].mean()
                        else:
                            target_next = next_q_value[0, o].max()
                        pred_next = q_value[1, o, action[2], action[0], action[1]]
                        error += abs(pred_next - target_next).data.detach().cpu().numpy()
                    replay_buffer.add(error, [state[0], 0.0], action, [next_state[0], 0.0], rewards_re, done_re, goal_image)
                else:
                    replay_buffer.add([state[0], 0.0], action, [next_state[0], 0.0], rewards_re, done_re, goal_image)

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
                torch.FloatTensor(np.array(rewards)).type(dtype),
                torch.FloatTensor([1 - done]).type(dtype),
                torch.FloatTensor(state[1]).type(dtype)
                ]
        if per:
            minibatch, idxs, is_weights = replay_buffer.sample(batch_size-1)
            combined_minibatch = combine_batch(minibatch, data)
            loss_q, error_q = calculate_loss(combined_minibatch)
            loss_next, error_next = calculate_loss_next(combined_minibatch)
            loss = loss_q + loss_next
            error = error_q + error_next
            errors = error.data.detach().cpu().numpy()[:-1]
            # update priority
            for i in range(batch_size-1):
                idx = idxs[i]
                replay_buffer.update(idx, errors[i])
        else:
            minibatch = replay_buffer.sample(batch_size-1)
            combined_minibatch = combine_batch(minibatch, data)
            loss_q, _ = calculate_loss(combined_minibatch)
            loss_next, _ = calculate_loss_next(combined_minibatch)
            loss = loss_q + loss_next

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        log_minibatchloss.append(loss.data.detach().cpu().numpy())
        log_minibatchloss_q.append(loss_q.data.detach().cpu().numpy())
        log_minibatchloss_next.append(loss_next.data.detach().cpu().numpy())

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
            log_loss_q.append(np.mean(log_minibatchloss_q))
            log_loss_next.append(np.mean(log_minibatchloss_next))
            log_eplen.append(ep_len)
            log_epsilon.append(epsilon)
            log_out.append(int(info['out_of_range']))
            log_success.append(int(info['success']))
            log_collisions.append(num_collisions)

            if ne % log_freq == 0:
                log_mean_returns = smoothing_log(log_returns, log_freq)
                log_mean_loss = smoothing_log(log_loss, log_freq)
                log_mean_loss_q = smoothing_log(log_loss_q, log_freq)
                log_mean_loss_next = smoothing_log(log_loss_next, log_freq)
                log_mean_eplen = smoothing_log(log_eplen, log_freq)
                log_mean_out = smoothing_log(log_out, log_freq)
                log_mean_success = smoothing_log(log_success, log_freq)
                log_mean_collisions = smoothing_log(log_collisions, log_freq)

                print()
                print("{} episodes. ({}/{} steps)".format(ne, t_step, total_steps))
                print("Success rate: {0:.2f}".format(log_mean_success[-1]))
                print("Mean reward: {0:.2f}".format(log_mean_returns[-1]))
                print("Mean loss: {0:.6f}".format(log_mean_loss[-1]))
                # print("Ep reward: {}".format(log_returns[-1]))
                print("Ep length: {}".format(log_mean_eplen[-1]))
                print("Epsilon: {}".format(epsilon))

                axes[0][0].plot(log_loss, color='#ff7f00', linewidth=0.5)
                axes[1][0].plot(log_returns, color='#60c7ff', linewidth=0.5)
                axes[2][0].plot(log_eplen, color='#83dcb7', linewidth=0.5)
                axes[2][1].plot(log_collisions, color='#ff33cc', linewidth=0.5)

                axes[0][0].plot(log_mean_loss, color='red')
                axes[0][0].plot(log_mean_loss_q, color='#36cfbf')
                axes[0][0].plot(log_mean_loss_next, color='#7dcf36')
                axes[1][0].plot(log_mean_returns, color='blue')
                axes[2][0].plot(log_mean_eplen, color='green')
                axes[0][1].plot(log_mean_success, color='red')
                axes[1][1].plot(log_mean_out, color='black')
                axes[2][1].plot(log_mean_collisions, color='#663399')

                #f.canvas.draw()
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
                    log_out, #6
                    log_loss_q, #7
                    log_loss_next, #8
                    ])
                np.save('results/board/%s' %savename, numpy_log)

                if log_mean_success[-1] > max_success:
                    max_success = log_mean_success[-1]
                    torch.save(FCQ.state_dict(), 'results/models/%s.pth' % savename)
                    print("Max performance! saving the model.")

            episode_reward = 0.
            log_minibatchloss = []
            log_minibatchloss_q = []
            log_minibatchloss_next = []
            state = env.reset()
            pre_action = None
            ep_len = 0
            num_collisions = 0

    print('Training finished.')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--num_blocks", default=1, type=int)
    parser.add_argument("--dist", default=0.08, type=float)
    parser.add_argument("--max_steps", default=30, type=int)
    parser.add_argument("--camera_height", default=96, type=int)
    parser.add_argument("--camera_width", default=96, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--bs", default=4, type=int)
    parser.add_argument("--buff_size", default=1e3, type=float)
    parser.add_argument("--total_steps", default=2e5, type=float)
    parser.add_argument("--learn_start", default=2e3, type=float)
    parser.add_argument("--update_freq", default=500, type=int)
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--per", action="store_true")
    parser.add_argument("--her", action="store_true")
    parser.add_argument("--reward", default="binary", type=str)
    parser.add_argument("--goal", default="pixel", type=str)
    parser.add_argument("--fcn_ver", default=1, type=int)
    parser.add_argument("--sampling", default="uniform", type=str)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--masking", default="", type=str)
    parser.add_argument("--next_v", action="store_true")
    ## Evaluate ##
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--model_path", default="SP_####_####.pth", type=str)
    parser.add_argument("--num_trials", default=100, type=int)
    parser.add_argument("--show_q", action="store_true")
    args = parser.parse_args()

    # env configuration #
    render = args.render
    num_blocks = args.num_blocks
    mov_dist = args.dist
    max_steps = args.max_steps
    camera_height = args.camera_height
    camera_width = args.camera_width
    reward_type = args.reward
    goal_type = args.goal

    # nn structure
    half = args.half
    if half:
        from models.seperate_fcn import FC_Q2Net_half as FC_Q2Net
    else:
        from models.seperate_fcn import FC_Q2Net

    # evaluate configuration #
    evaluation = args.evaluate
    model_path = os.path.join("results/models/", args.model_path)
    num_trials = args.num_trials
    visualize_q = args.show_q
    if visualize_q:
        render = True

    now = datetime.datetime.now()
    savename = "Q2_%s" % (now.strftime("%m%d_%H%M"))
    if not evaluation:
        if not os.path.exists("results/config/"):
            os.makedirs("results/config/")
        with open("results/config/%s.json" % savename, 'w') as cf:
            json.dump(args.__dict__, cf, indent=2)

    env = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
            control_freq=5, data_format='NCHW', xml_ver=0)
    env = pushpixel_env(env, num_blocks=num_blocks, mov_dist=mov_dist, max_steps=max_steps, \
            task=1, reward_type=reward_type, goal_type=goal_type, seperate=True)

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
    sampling = args.sampling  # 'sum' / 'choice' / 'max'
    masking = args.masking
    next_v = args.next_v

    if goal_type=="pixel":
        in_channel = 3 + num_blocks
    else:
        in_channel = 6
            
    if evaluation:
        evaluate(env=env, n_blocks=num_blocks, in_channel=in_channel, model_path=model_path, \
                num_trials=num_trials, visualize_q=visualize_q, sampling=sampling, masking=masking)
    else:
        learning(env=env, savename=savename, n_blocks=num_blocks, in_channel=in_channel, \
                learning_rate=learning_rate, batch_size=batch_size, buff_size=buff_size, \
                total_steps=total_steps, learn_start=learn_start, update_freq=update_freq, \
                log_freq=log_freq, double=double, her=her, per=per, visualize_q=visualize_q, \
                goal_type=goal_type, sampling=sampling, next_v=next_v, masking=masking)
