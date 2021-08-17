import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../ur5_mujoco'))
from pushpixel_env import *
from utils import *

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


def get_action(env, fc_qnet, state, epsilon, pre_action=None, with_q=False, sample='sum', masking=''):
    if np.random.random() < epsilon:
        action = [np.random.randint(crop_min,crop_max), np.random.randint(crop_min,crop_max), np.random.randint(env.num_bins)]
        # action = [np.random.randint(env.env.camera_height), np.random.randint(env.env.camera_width), np.random.randint(env.num_bins)]
        if with_q:
            state_im = torch.tensor([state[0]]).type(dtype)
            goal_im = torch.tensor([state[1]]).type(dtype)
            state_goal = torch.cat((state_im, goal_im), 1)
            q_value = fc_qnet(state_goal)
            q_raw = q_value[0][0].detach().cpu().numpy()
            q = np.zeros_like(q_raw)
            q[:, crop_min:crop_max, crop_min:crop_max] = q_raw[:, crop_min:crop_max, crop_min:crop_max]

            q_next_raw = q_value[0][1].detach().cpu().numpy()
            mask = np.ones_like(q)
    else:
        state_im = torch.tensor([state[0]]).type(dtype)
        goal_im = torch.tensor([state[1]]).type(dtype)
        state_goal = torch.cat((state_im, goal_im), 1)
        q_value = fc_qnet(state_goal) # 1 x 2 x nb x 8 x 96 x 96
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
            idx1 = q_raw.max(2).max(2).argmax(1)
            idx2 = q_raw.max(1).max(2).argmax(1)
            idx3 = q_raw.max(1).max(1).argmax(1)
            max_q = [q_raw[o, idx1[o], idx2[o], idx3[o]] for o in range(env.num_blocks)]
            selected_obj = np.argmax(max_q)
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
        return action, q, q_raw, q_next_raw, mask
    else:
        return action


def evaluate( env, 
        n_blocks=3,
        in_channel=6,
        n_channel=12,
        model_path='',
        num_trials=10,
        visualize_q=False,
        sampling='choice',
        masking='max q'
        ):
    FCQ = FC_Q2Net(8, in_channel, n_blocks, n_channel).type(dtype)
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
        plt.rc('font', size=7)
        fig, ax = plt.subplots(3,4)
        fig.tight_layout()
        fig.set_size_inches(10,7)
        s0 = deepcopy(state[0]).transpose([1,2,0])
        if env.goal_type == 'pixel':
            s1 = np.zeros([env.env.camera_height, env.env.camera_width, 3])
            s1[:, :, :n_blocks] = state[1].transpose([1, 2, 0])
        else:
            s1 = deepcopy(state[1]).transpose([1, 2, 0])
        ax[0,0].imshow(s1)
        ax[0,0].set_title('Goal')
        ax[0,1].imshow(s0)
        ax[0,1].set_title('State')
        ax[0,2].imshow(np.zeros_like(s0))
        ax[0,2].set_title('Q')
        ax[0,3].imshow(np.zeros_like(s0))
        ax[0,3].set_title('Mask')
        ax[1,0].imshow(np.zeros_like(s0))
        ax[1,1].imshow(np.zeros_like(s0))
        ax[1,2].imshow(np.zeros_like(s0))
        ax[1,3].imshow(np.ones_like(s0))
        ax[2,0].imshow(np.zeros_like(s0))
        ax[2,1].imshow(np.zeros_like(s0))
        ax[2,2].imshow(np.zeros_like(s0))
        ax[2,3].imshow(np.ones_like(s0))
        for o in range(n_blocks):
            ax[1,o].set_title("Q_%d" %(o+1))
            ax[2,o].set_title("mask_%d" %(o+1))

        plt.show(block=False)
        fig.canvas.draw()
        fig.canvas.draw()

    while ne < num_trials:
        action, q_map, q_raw, q_next_raw, mask = get_action(env, FCQ, state, epsilon=0.0, pre_action=pre_action, with_q=True, sample=sampling, masking=masking)
        if visualize_q:
            s0 = deepcopy(state[0]).transpose([1, 2, 0])
            if env.goal_type == 'pixel':
                s1 = np.zeros([env.env.camera_height, env.env.camera_width, 3])
                s1[:, :, :n_blocks] = state[1].transpose([1, 2, 0])
            else:
                s1 = deepcopy(state[1]).transpose([1, 2, 0])
            ax[0,0].imshow(s1)
            s0[action[0], action[1]] = [1, 0, 0]
            # q_map = q_map[0]
            q_map = q_map.transpose([1,2,0]).max(2)
            # print(q_map.max())
            # print(q_map.min())
            # q_map[action[0], action[1]] = 1.5
            ax[0,1].imshow(s0)
            ax[0,2].imshow(q_map, vmax=1.8, vmin=-0.2)
            ## visualize Q ##
            for o in range(n_blocks):
                q = q_raw[o].transpose([1,2,0]).max(2)
                ax[1, o].imshow(q, vmax=1.8, vmin=-0.2)
            ## visualize mask ##
            ax[0,3].imshow(mask.any(0), vmax=1.1, vmin=-0.2)
            if masking!='':
                for o in range(n_blocks):
                    m = np.ones_like(mask)
                    if 'q' in masking:
                        m[q_raw[o] < 0] = 0.0
                    if 'mean' in masking:
                        m[(q_next_raw[o] - q_raw[o].mean()) < 0] = 0.0
                    if 'max' in masking:
                        m[(q_next_raw[o] - q_raw[o].max()) < 0] = 0.0
                    ax[2, o].imshow(m.any(0), vmax=1.1, vmin=-0.2)
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
        n_channel=12,
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
        masking='',
        continue_learning=False,
        ):

    FCQ = FC_Q2Net(8, in_channel, n_blocks, n_channel).type(dtype)
    FCQ_target = FC_Q2Net(8, in_channel, n_blocks, n_channel).type(dtype)
    FCQ_target.load_state_dict(FCQ.state_dict())

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
        replay_buffer = ReplayBuffer([3, env.env.camera_height, env.env.camera_width], 
                    [goal_ch, env.env.camera_height, env.env.camera_width], 1, \
                    save_goal=True, save_gripper=False, max_size=int(buff_size),\
                    dim_reward=n_blocks)

    model_parameters = filter(lambda p: p.requires_grad, FCQ.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("# of params: %d"%params)


    if double:
        calculate_loss = calculate_loss_double_constrained
    else:
        calculate_loss = calculate_loss_constrained

    if next_v:
        calculate_loss_next = calculate_loss_next_v
    else:
        calculate_loss_next = calculate_loss_next_q

    if continue_learning:
        numpy_log = np.load(model_path.replace('models/', 'board/').replace('.pth', '.npy'), allow_pickle=True)
        log_returns = list(numpy_log[0])
        log_loss = list(numpy_log[1])
        log_eplen = list(numpy_log[2])
        log_epsilon = list(numpy_log[3])
        log_success = list(numpy_log[4])
        log_collisions = list(numpy_log[5])
        log_out = list(numpy_log[6])
        log_success_block = list(numpy_log[7])
        log_loss_q = list(numpy_log[8])
        log_loss_next = list(numpy_log[9])
        log_mean_success_block = [np.mean(sb[:-log_freq]) for sb in log_success_block]
    else:
        log_returns = []
        log_loss = []
        log_loss_q = []
        log_loss_next = []
        log_eplen = []
        log_epsilon = []
        log_out = []
        log_success = []
        log_collisions = []
        log_success_block = [[], [], []]
        log_mean_success_block = [[], [], []]
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
    f, axes = plt.subplots(3, 3) # 3,2
    f.set_figheight(12) #9
    f.set_figwidth(20) #12

    axes[0][0].set_title('Block 1 success')  # 1
    axes[0][0].set_ylim([0, 1])
    axes[0][1].set_title('Block 2 success')  # 2
    axes[0][1].set_ylim([0, 1])
    axes[0][2].set_title('Block 3 success')  # 3
    axes[0][2].set_ylim([0, 1])
    axes[1][0].set_title('Success Rate')  # 4
    axes[1][0].set_ylim([0, 1])
    axes[1][1].set_title('Episode Return')  # 5
    axes[1][2].set_title('Loss')  # 6
    axes[2][0].set_title('Episode Length')  # 7
    axes[2][1].set_title('Out of Range')  # 8
    axes[2][1].set_ylim([0, 1])
    axes[2][2].set_title('Num Collisions')  # 9

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
        action, q_map, q_raw, q_next_raw, mask = get_action(env, FCQ, state, epsilon=epsilon, pre_action=pre_action, with_q=True, sample=sampling, masking=masking)
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
            trajectories = []
            replay_tensors = []

            traj_tensor = [
                torch.FloatTensor(state[0]).type(dtype),
                torch.FloatTensor(next_state[0]).type(dtype),
                torch.FloatTensor(action).type(dtype),
                torch.FloatTensor(np.array(rewards)).type(dtype),
                torch.FloatTensor([1 - done]).type(dtype),
                torch.FloatTensor(state[1]).type(dtype),
            ]
            replay_tensors.append(traj_tensor)
            trajectories.append([[state[0], 0.0], action, [next_state[0], 0.0], np.array(rewards), done, state[1]])

            if her and not done:
                her_sample = sample_her_transitions(env, info, next_state)
                ig_samples = sample_ig_transitions(env, info, next_state, num_samples=3)
                samples = her_sample + ig_samples
                for sample in samples:
                    rewards_re, goal_image, done_re, block_success_re = sample
                    state_re = [state[0], goal_image]
                    action_re = deepcopy(action)

                    traj_tensor = [
                        torch.FloatTensor(state_re[0]).type(dtype),
                        torch.FloatTensor(next_state[0]).type(dtype),
                        torch.FloatTensor(action_re).type(dtype),
                        torch.FloatTensor(np.array(rewards)).type(dtype),
                        torch.FloatTensor([1 - done_re]).type(dtype),
                        torch.FloatTensor(state_re[1]).type(dtype),
                    ]
                    replay_tensors.append(traj_tensor)
                    trajectories.append([[state_re[0], 0.0], action_re, [next_state[0], 0.0], np.array(rewards_re), done_re, state_re[1]])

            minibatch = None
            for data in replay_tensors:
                minibatch = combine_batch(minibatch, data)
            _, error_q = calculate_loss(minibatch, FCQ, FCQ_target, num_blocks)
            _, error_next = calculate_loss_next(minibatch, FCQ, FCQ_target, num_blocks)
            error = error_q + error_next
            error = error.data.detach().cpu().numpy()

            for i, traj in enumerate(trajectories):
                replay_buffer.add(error[i], *traj)
            # state_im = torch.tensor([state[0]]).type(dtype)
            # goal_im = torch.tensor([state[1]]).type(dtype)
            # next_state_im = torch.tensor([next_state[0]]).type(dtype)
            # action_tensor = torch.tensor([action]).type(dtype)
            # rewards_tensor = torch.tensor([rewards]).type(dtype)
            #
            # batch = [state_im, next_state_im, action_tensor, rewards_tensor, 1-int(done), goal_im]
            # _, error_q = calculate_loss(batch, FCQ, FCQ_target, num_blocks)
            # _, error_next = calculate_loss_next(batch, FCQ, FCQ_target, num_blocks)
            # error = error_q + error_next
            # error = error.data.detach().cpu().numpy()
            # replay_buffer.add(error, [state[0], 0.0], action, [next_state[0], 0.0], rewards, done, state[1])

        else:
            trajectories = []
            replay_tensors = []

            traj_tensor = [
                torch.FloatTensor(state[0]).type(dtype),
                torch.FloatTensor(next_state[0]).type(dtype),
                torch.FloatTensor(action).type(dtype),
                torch.FloatTensor(np.array(rewards)).type(dtype),
                torch.FloatTensor([1 - done]).type(dtype),
                torch.FloatTensor(state[1]).type(dtype),
            ]
            replay_tensors.append(traj_tensor)
            trajectories.append([[state[0], 0.0], action, [next_state[0], 0.0], np.array(rewards), done, state[1]])

            ## HER ##
            if her and not done:
                her_sample = sample_her_transitions(env, info, next_state)
                ig_samples = sample_ig_transitions(env, info, next_state, num_samples=3)
                samples = her_sample + ig_samples
                for sample in samples:
                    rewards_re, goal_image, done_re, block_success_re = sample
                    state_re = [state[0], goal_image]
                    action_re = deepcopy(action)
                    trajectories.append([[state_re[0], 0.0], action_re, [next_state[0], 0.0], np.array(rewards_re), done_re, state_re[1]])

            for traj in trajectories:
                replay_buffer.add(*traj)

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
        online_data = replay_tensors
        if per:
            minibatch, idxs, is_weights = replay_buffer.sample(batch_size - len(online_data))
            for data in online_data:
                minibatch = combine_batch(minibatch, data)
            loss_q, error_q = calculate_loss(minibatch, FCQ, FCQ_target, num_blocks)
            loss_next, error_next = calculate_loss_next(minibatch, FCQ, FCQ_target, num_blocks)
            loss = loss_q + loss_next
            error = error_q + error_next
            errors = error.data.detach().cpu().numpy()[:-len(online_data)]
            # update priority
            for i in range(batch_size-len(online_data)):
                idx = idxs[i]
                replay_buffer.update(idx, errors[i])
        else:
            minibatch = replay_buffer.sample(batch_size-len(online_data))
            for data in online_data:
                minibatch = combine_batch(minibatch, data)
            loss_q, _ = calculate_loss(minibatch, FCQ, FCQ_target, num_blocks)
            loss_next, _ = calculate_loss_next(minibatch, FCQ, FCQ_target, num_blocks)
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
            for o in range(3):
                log_success_block[o].append(int(info['block_success'][o]))

            if ne % log_freq == 0:
                log_mean_returns = smoothing_log(log_returns, log_freq)
                log_mean_loss = smoothing_log(log_loss, log_freq)
                log_mean_loss_q = smoothing_log(log_loss_q, log_freq)
                log_mean_loss_next = smoothing_log(log_loss_next, log_freq)
                log_mean_eplen = smoothing_log(log_eplen, log_freq)
                log_mean_out = smoothing_log(log_out, log_freq)
                log_mean_success = smoothing_log(log_success, log_freq)
                log_mean_collisions = smoothing_log(log_collisions, log_freq)
                for o in range(3):
                    log_mean_success_block[o] = smoothing_log(log_success_block[o], log_freq)

                print()
                print("{} episodes. ({}/{} steps)".format(ne, t_step, total_steps))
                print("Success rate: {0:.2f}".format(log_mean_success[-1]))
                print("Mean reward: {0:.2f}".format(log_mean_returns[-1]))
                print("Mean loss: {0:.6f}".format(log_mean_loss[-1]))
                # print("Ep reward: {}".format(log_returns[-1]))
                print("Ep length: {}".format(log_mean_eplen[-1]))
                print("Epsilon: {}".format(epsilon))

                axes[1][2].plot(log_loss, color='#ff7f00', linewidth=0.5)  # 3->6
                axes[1][1].plot(log_returns, color='#60c7ff', linewidth=0.5)  # 5
                axes[2][0].plot(log_eplen, color='#83dcb7', linewidth=0.5)  # 7
                axes[2][2].plot(log_collisions, color='#ff33cc', linewidth=0.5)  # 8->9

                for o in range(3):
                    axes[0][o].plot(log_mean_success_block[o], color='red')  # 1,2,3

                axes[1][2].plot(log_mean_loss, color='red')  # 3->6
                axes[1][2].plot(log_mean_loss_q, color='#36cfbf')
                axes[1][2].plot(log_mean_loss_next, color='#7dcf36')
                axes[1][1].plot(log_mean_returns, color='blue')  # 5
                axes[2][0].plot(log_mean_eplen, color='green')  # 7
                axes[1][0].plot(log_mean_success, color='red')  # 4
                axes[2][1].plot(log_mean_out, color='black')  # 6->8
                axes[2][2].plot(log_mean_collisions, color='#663399')  # 8->9

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
                    log_success_block, #7
                    log_loss_q, #8
                    log_loss_next, #9
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

            if ne % update_freq == 0:
                FCQ_target.load_state_dict(FCQ.state_dict())
                lr_scheduler.step()
                epsilon = max(epsilon_decay * epsilon, min_epsilon)

    print('Training finished.')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--num_blocks", default=3, type=int)
    parser.add_argument("--dist", default=0.08, type=float)
    parser.add_argument("--max_steps", default=30, type=int)
    parser.add_argument("--camera_height", default=96, type=int)
    parser.add_argument("--camera_width", default=96, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--bs", default=8, type=int)
    parser.add_argument("--buff_size", default=1e3, type=float)
    parser.add_argument("--total_steps", default=2e5, type=float)
    parser.add_argument("--learn_start", default=1e3, type=float)
    parser.add_argument("--update_freq", default=100, type=int)
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument("--double", action="store_false")
    parser.add_argument("--per", action="store_false")
    parser.add_argument("--her", action="store_false")
    parser.add_argument("--reward", default="binary", type=str)
    parser.add_argument("--goal", default="block", type=str)
    #parser.add_argument("--fcn_ver", default=1, type=int)
    parser.add_argument("--sampling", default="uniform", type=str)
    parser.add_argument("--small", action="store_true") # default: False
    parser.add_argument("--masking", default="", type=str)
    parser.add_argument("--next_v", action="store_true") # default: next_q
    parser.add_argument("--continue_learning", action="store_true")
    ## Evaluate ##
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--model_path", default="####_####", type=str)
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
    resnet = True
    small = args.small
    if resnet:
        from models.fcn_resnet import FC_Q2_ResNet as FC_Q2Net
        if small:
            hidden_channels = 8
        else:
            hidden_channels = 12
    #if half:
    #    from models.seperate_fcn import FC_Q2Net_half as FC_Q2Net
    #else:
    #    from models.seperate_fcn import FC_Q2Net

    # evaluate configuration #
    evaluation = args.evaluate
    model_path = os.path.join("results/models/Q2_%s.pth"%args.model_path)
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
    #fcn_ver = args.fcn_ver
    sampling = args.sampling  # 'sum' / 'choice' / 'max'
    masking = args.masking
    next_v = args.next_v
    continue_learning = args.continue_learning

    if goal_type=="pixel":
        in_channel = 3 + num_blocks
    else:
        in_channel = 6
            
    if evaluation:
        evaluate(env=env,
                n_blocks=num_blocks,
                in_channel=in_channel,
                n_channel=hidden_channels,
                model_path=model_path,
                num_trials=num_trials,
                visualize_q=visualize_q,
                sampling=sampling,
                masking=masking
                )
    else:
        learning(env=env,
                savename=savename,
                n_blocks=num_blocks,
                in_channel=in_channel,
                n_channel=hidden_channels,
                learning_rate=learning_rate,
                batch_size=batch_size,
                buff_size=buff_size,
                total_steps=total_steps,
                learn_start=learn_start,
                update_freq=update_freq,
                log_freq=log_freq,
                double=double,
                her=her,
                per=per,
                visualize_q=visualize_q,
                goal_type=goal_type,
                sampling=sampling,
                next_v=next_v,
                masking=masking,
                continue_learning=continue_learning
                )
