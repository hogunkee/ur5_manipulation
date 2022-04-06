import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../../ur5_mujoco'))
sys.path.append(os.path.join(FILE_PATH, '..'))
from object_env import *
from training_utils import *
from skimage import color
from PIL import Image

import torch
import torch.nn as nn
import argparse
import json

import copy
import time
import datetime
import random
import pylab

from models.discriminator import Discriminator
from sdf_module import SDFModule
from replay_buffer import ReplayBuffer, PER
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import wandb

#dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def norm_npy(array):
    positive = array - array.min()
    return positive / positive.max()

def pad_flag(flag, nmax):
    nsdf = len(flag)
    padded = np.zeros(nmax)
    if nsdf > nmax:
        padded[:] = flag[:nmax]
    elif nsdf > 0:
        padded[:nsdf] = flag 
    return padded

def pad_sdf(sdf, nmax, res=96):
    nsdf = len(sdf)
    padded = np.zeros([nmax, res, res])
    if nsdf > nmax:
        padded[:] = sdf[:nmax]
    elif nsdf > 0:
        padded[:nsdf] = sdf
    return padded

def get_action(env, max_blocks, qnet, depth, sdf_raw, sdfs, goal_flags, epsilon, with_q=False, sdf_action=False, target_res=96):
    if np.random.random() < epsilon:
        #print('Random action')
        obj = np.random.randint(len(sdf_raw))
        theta = np.random.randint(env.num_bins)
        if with_q:
            nsdf = sdfs[0].shape[0]
            s = pad_sdf(sdfs[0], max_blocks, target_res)
            s = torch.FloatTensor(s).to(device).unsqueeze(0)
            g = pad_sdf(sdfs[1], max_blocks, target_res)
            g = torch.FloatTensor(g).to(device).unsqueeze(0)
            nsdf = torch.LongTensor([nsdf]).to(device)
            goalflag = torch.FloatTensor(goal_flags).to(device).unsqueeze(0)
            q_value = qnet([s, g], nsdf, goalflag)
            q = q_value[0][:nsdf].detach().cpu().numpy()
    else:
        nsdf = sdfs[0].shape[0]
        s = pad_sdf(sdfs[0], max_blocks, target_res)
        empty_mask = (np.sum(s, (1,2))==0)[:nsdf]
        s = torch.FloatTensor(s).to(device).unsqueeze(0)
        g = pad_sdf(sdfs[1], max_blocks, target_res)
        g = torch.FloatTensor(g).to(device).unsqueeze(0)
        nsdf = torch.LongTensor([nsdf]).to(device)
        goalflag = torch.FloatTensor(goal_flags).to(device).unsqueeze(0)
        q_value = qnet([s, g], nsdf, goalflag)
        q = q_value[0][:nsdf].detach().cpu().numpy()
        q[empty_mask] = q.min()

        obj = q.max(1).argmax()
        theta = q.max(0).argmax()

    action = [obj, theta]
    sdf_target = sdf_raw[obj]
    cx, cy = env.get_center_from_sdf(sdf_target, depth)

    mask = None
    if sdf_action:
        masks = []
        for s in sdf_raw:
            m = copy.deepcopy(s)
            m[m<0] = 0
            m[m>0] = 1
            masks.append(m)
        mask = np.sum(masks, 0)

    if with_q:
        return action, [cx, cy, theta], mask, q
    else:
        return action, [cx, cy, theta], mask

def learning(env, 
        savename,
        sdf_module,
        n_actions=8,
        n_hidden=16,
        learning_rate=1e-4, 
        batch_size=64, 
        buff_size=1e4, 
        total_episodes=1e4,
        learn_start=1e4,
        update_freq=100,
        log_freq=1e3,
        double=True,
        per=True,
        her=True,
        visualize_q=False,
        pretrain=False,
        continue_learning=False,
        model_path='',
        clip_sdf=False,
        sdf_action=False,
        graph_normalize=False,
        max_blocks=5,
        oracle_matching=False,
        round_sdf=False,
        ):

    print('='*30)
    print('{} learing starts.'.format(savename))
    print('='*30)
    qnet = QNet(max_blocks, env.num_blocks, n_actions, n_hidden=n_hidden, normalize=graph_normalize, resize=sdf_module.resize).to(device)
    if pretrain:
        qnet.load_state_dict(torch.load(model_path))
        print('Loading pre-trained model: {}'.format(model_path))
    elif continue_learning:
        qnet.load_state_dict(torch.load(model_path))
        print('Loading trained model: {}'.format(model_path))
    qnet_target = QNet(max_blocks, env.num_blocks, n_actions, n_hidden=n_hidden, normalize=graph_normalize).to(device)
    qnet_target.load_state_dict(qnet.state_dict())
    dnet = Discriminator(max_blocks).to(device)

    #optimizer = torch.optim.SGD(qnet.parameters(), lr=learning_rate, momentum=0.9, weight_decay=2e-5)
    optimizer = torch.optim.Adam(qnet.parameters(), lr=learning_rate)
    disc_optimizer = torch.optim.Adam(dnet.parameters(), lr=2e-5)

    if sdf_module.resize:
        sdf_res = 96
    else:
        sdf_res = 480

    if per:
        replay_buffer = PER([max_blocks, sdf_res, sdf_res], [max_blocks, sdf_res, sdf_res], max_size=int(buff_size), save_goal_flag=True)
    else:
        replay_buffer = ReplayBuffer([max_blocks, sdf_res, sdf_res], [max_blocks, sdf_res, sdf_res], max_size=int(buff_size), save_goal_flag=True)

    model_parameters = filter(lambda p: p.requires_grad, qnet.parameters())
    D_model_parameters = filter(lambda p: p.requires_grad, dnet.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    params += sum([np.prod(p.size()) for p in D_model_parameters])
    print("# of params: %d"%params)

    if double:
        calculate_loss = calculate_loss_gcn_gf_double
    else:
        calculate_loss = calculate_loss_gcn_gf_origin

    if continue_learning and not pretrain:
        numpy_log = np.load(model_path.replace('models/', 'board/').replace('.pth', '.npy'), allow_pickle=True)
        log_returns = list(numpy_log[0])
        log_loss = list(numpy_log[1])
        log_eplen = list(numpy_log[2])
        log_epsilon = list(numpy_log[3])
        log_success = list(numpy_log[4])
        log_sdfsuccess = list(numpy_log[5])
        log_out = list(numpy_log[6])
        log_success_block = list(numpy_log[7])
        log_Dloss = list(numpy_log[8])
        log_mean_success_block = [[] for _ in range(env.num_blocks)]
    else:
        log_returns = []
        log_loss = []
        log_eplen = []
        log_epsilon = []
        log_success = []
        log_sdfsuccess = []
        log_out = []
        log_success_block = [[] for _ in range(env.num_blocks)]
        log_Dloss = []
        log_mean_success_block = [[] for _ in range(env.num_blocks)]

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

    #lr_decay = 0.98
    #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)

    if len(log_epsilon) == 0:
        epsilon = 0.5 #1.0
        start_epsilon = 0.5
    else:
        epsilon = log_epsilon[-1]
        start_epsilon = log_epsilon[-1]
    min_epsilon = 0.1
    epsilon_decay = 0.98
    max_success = 0.0
    st = time.time()

    if visualize_q:
        cm = pylab.get_cmap('gist_rainbow')
        fig = plt.figure()
        ax0 = fig.add_subplot(221)
        ax1 = fig.add_subplot(222)
        ax2 = fig.add_subplot(223)
        ax3 = fig.add_subplot(224)
        ax0.set_title('Goal')
        ax1.set_title('Observation')
        ax2.set_title('Goal SDF')
        ax3.set_title('Current SDF')
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax3.set_xticks([])
        ax3.set_yticks([])

        plt.show(block=False)
        fig.canvas.draw()

    count_steps = 0
    for ne in range(total_episodes):
        ep_len = 0
        episode_reward = 0.
        log_minibatchloss = []
        log_minibatchDloss = []

        check_env_ready = False
        while not check_env_ready:
            (state_img, goal_img), info = env.reset()
            sdf_st, sdf_raw, feature_st = sdf_module.get_sdf_features_with_ucn(state_img[0], state_img[1], env.num_blocks, clip=clip_sdf)
            sdf_g, _, feature_g = sdf_module.get_sdf_features_with_ucn(goal_img[0], goal_img[1], env.num_blocks, clip=clip_sdf)
            if round_sdf:
                sdf_g = sdf_module.make_round_sdf(sdf_g)
            check_env_ready = (len(sdf_g)==env.num_blocks) & (len(sdf_st)==env.num_blocks)
            if not check_env_ready:
                continue
            n_detection = len(sdf_st)
            # target: st / source: g
            if oracle_matching:
                sdf_st_align = sdf_module.oracle_align(sdf_st, info['pixel_poses'])
                sdf_raw = sdf_module.oracle_align(sdf_raw, info['pixel_poses'], scale=1)
                sdf_g = sdf_module.oracle_align(sdf_g, info['pixel_goals'])
            else:
                matching = sdf_module.object_matching(feature_st, feature_g)
                sdf_st_align = sdf_module.align_sdf(matching, sdf_st, sdf_g)
                sdf_raw = sdf_module.align_sdf(matching, sdf_raw, np.zeros([env.num_blocks, *sdf_raw.shape[1:]]))

        goal_flag = np.array([False] * max_blocks)

        masks = []
        for s in sdf_raw:
            masks.append(s>0)
        sdf_module.init_tracker(state_img[0], masks)

        if visualize_q:
            if env.env.camera_depth:
                ax0.imshow(goal_img[0])
                ax1.imshow(state_img[0])
            else:
                ax0.imshow(goal_img)
                ax1.imshow(state_img)
            # goal sdfs
            vis_g = norm_npy(sdf_g + 50*(sdf_g>0).astype(float))
            goal_sdfs = np.zeros([sdf_res, sdf_res, 3])
            for _s in range(len(vis_g)):
                goal_sdfs += np.expand_dims(vis_g[_s], 2) * np.array(cm(_s/5)[:3])
            ax2.imshow(norm_npy(goal_sdfs))
            # current sdfs
            vis_c = norm_npy(sdf_st_align + 50*(sdf_st_align>0).astype(float))
            current_sdfs = np.zeros([sdf_res, sdf_res, 3])
            for _s in range(len(vis_c)):
                current_sdfs += np.expand_dims(vis_c[_s], 2) * np.array(cm(_s/5)[:3])
            ax3.imshow(norm_npy(current_sdfs))
            plt.show(block=False)
            fig.canvas.draw()

        for t_step in range(env.max_steps):
            count_steps += 1
            ep_len += 1
            action, pose_action, sdf_mask, q_map = get_action(env, max_blocks, qnet, \
                    state_img[1], sdf_raw, [sdf_st_align, sdf_g], goal_flag, \
                    epsilon=epsilon,  with_q=True, sdf_action=sdf_action, target_res=sdf_res)

            (next_state_img, _), reward, done, info = env.step(pose_action, sdf_mask)
            episode_reward += reward
            sdf_ns, sdf_raw, feature_ns = sdf_module.get_sdf_features(next_state_img[0], next_state_img[1], env.num_blocks, clip=clip_sdf)
            pre_n_detection = n_detection
            n_detection = len(sdf_ns)
            if oracle_matching:
                sdf_ns_align = sdf_module.oracle_align(sdf_ns, info['pixel_poses'])
                sdf_raw = sdf_module.oracle_align(sdf_raw, info['pixel_poses'], scale=1)
            else:
                matching = sdf_module.object_matching(feature_ns, feature_g)
                sdf_ns_align = sdf_module.align_sdf(matching, sdf_ns, sdf_g)
                sdf_raw = sdf_module.align_sdf(matching, sdf_raw, np.zeros([env.num_blocks, *sdf_raw.shape[1:]]))

            # detection failed #
            if n_detection == 0:
                reward = -1.
                done = True

            next_sdf_success = sdf_module.check_sdf_align(sdf_ns_align, sdf_g, env.num_blocks)
            next_goal_flag = pad_flag(next_sdf_success, max_blocks)

            ## check GT poses and SDF centers ##
            if next_sdf_success.all():
                reward += 10
                done = True
                info['sdf_success'] = True
            else:
                info['sdf_success'] = False
            if info['block_success'].all():
                info['success'] = True
            else:
                info['success'] = False

            if visualize_q:
                if env.env.camera_depth:
                    ax1.imshow(next_state_img[0])
                else:
                    ax1.imshow(next_state_img)

                # goal sdfs
                vis_g = norm_npy(sdf_g + 50*(sdf_g>0).astype(float))
                goal_sdfs = np.zeros([sdf_res, sdf_res, 3])
                for _s in range(len(vis_g)):
                    goal_sdfs += np.expand_dims(vis_g[_s], 2) * np.array(cm(_s/5)[:3])
                ax2.imshow(norm_npy(goal_sdfs))
                # current sdfs
                vis_c = norm_npy(sdf_ns_align + 50*(sdf_ns_align>0).astype(float))
                current_sdfs = np.zeros([sdf_res, sdf_res, 3])
                for _s in range(len(vis_c)):
                    current_sdfs += np.expand_dims(vis_c[_s], 2) * np.array(cm(_s/5)[:3])
                ax3.imshow(norm_npy(current_sdfs))
                fig.canvas.draw()

            ## save transition to the replay buffer ##
            if per:
                trajectories = []
                replay_tensors = []

                trajectories.append([sdf_st_align, action, sdf_ns_align, reward, done, sdf_g, sdf_g, goal_flag, next_goal_flag])

                traj_tensor = [
                    torch.FloatTensor(pad_sdf(sdf_st_align, max_blocks, sdf_res)).to(device),
                    torch.FloatTensor(pad_sdf(sdf_ns_align, max_blocks, sdf_res)).to(device),
                    torch.FloatTensor(action).to(device),
                    torch.FloatTensor([reward]).to(device),
                    torch.FloatTensor([1 - done]).to(device),
                    torch.FloatTensor(pad_sdf(sdf_g, max_blocks, sdf_res)).to(device),
                    torch.FloatTensor(pad_sdf(sdf_g, max_blocks, sdf_res)).to(device),
                    torch.LongTensor([len(sdf_st_align)]).to(device),
                    torch.LongTensor([len(sdf_ns_align)]).to(device),
                    torch.FloatTensor(goal_flag).to(device),
                    torch.FloatTensor(next_goal_flag).to(device),
                ]
                replay_tensors.append(traj_tensor)

                ## HER ##
                if her and not done:
                    her_sample = sample_her_transitions(env, info)
                    for sample in her_sample:
                        reward_re, goal_re, done_re, block_success_re = sample
                        if round_sdf:
                            sdf_ns_align_round = sdf_module.make_round_sdf(sdf_ns_align)

                        # check success #
                        if round_sdf:
                            sdf_success_re = sdf_module.check_sdf_align(sdf_st_align, sdf_ns_align_round, env.num_blocks)
                        else:
                            sdf_success_re = sdf_module.check_sdf_align(sdf_st_align, sdf_ns_align, env.num_blocks)
                        goal_flag_re = pad_flag(sdf_success_re, max_blocks)
                        next_goal_flag_re = np.array([1]*env.num_blocks + [0]*(max_blocks-env.num_blocks)).astype(bool)
                        reward_re += 10
                        done_re = True
                        if round_sdf:
                            trajectories.append([sdf_st_align, action, sdf_ns_align, reward_re, done_re, sdf_ns_align_round, sdf_ns_align_round, goal_flag_re, next_goal_flag_re])
                            traj_tensor = [
                                torch.FloatTensor(pad_sdf(sdf_st_align, max_blocks, sdf_res)).to(device),
                                torch.FloatTensor(pad_sdf(sdf_ns_align, max_blocks, sdf_res)).to(device),
                                torch.FloatTensor(action).to(device),
                                torch.FloatTensor([reward_re]).to(device),
                                torch.FloatTensor([1 - done_re]).to(device),
                                torch.FloatTensor(pad_sdf(sdf_ns_align_round, max_blocks, sdf_res)).to(device),
                                torch.FloatTensor(pad_sdf(sdf_ns_align_round, max_blocks, sdf_res)).to(device),
                                torch.LongTensor([len(sdf_st_align)]).to(device),
                                torch.LongTensor([len(sdf_ns_align)]).to(device),
                                torch.FloatTensor(goal_flag_re),
                                torch.FloatTensor(next_goal_flag_re),
                            ]
                        else:
                            trajectories.append([sdf_st_align, action, sdf_ns_align, reward_re, done_re, sdf_ns_align, sdf_ns_align, goal_flag_re, next_goal_flag_re])
                            traj_tensor = [
                                torch.FloatTensor(pad_sdf(sdf_st_align, max_blocks, sdf_res)).to(device),
                                torch.FloatTensor(pad_sdf(sdf_ns_align, max_blocks, sdf_res)).to(device),
                                torch.FloatTensor(action).to(device),
                                torch.FloatTensor([reward_re]).to(device),
                                torch.FloatTensor([1 - done_re]).to(device),
                                torch.FloatTensor(pad_sdf(sdf_ns_align, max_blocks, sdf_res)).to(device),
                                torch.FloatTensor(pad_sdf(sdf_ns_align, max_blocks, sdf_res)).to(device),
                                torch.LongTensor([len(sdf_st_align)]).to(device),
                                torch.LongTensor([len(sdf_ns_align)]).to(device),
                                torch.FloatTensor(goal_flag_re),
                                torch.FloatTensor(next_goal_flag_re),
                            ]
                        replay_tensors.append(traj_tensor)

                minibatch = None
                for data in replay_tensors:
                    minibatch = combine_batch(minibatch, data)
                _, error = calculate_loss(minibatch, qnet, qnet_target)
                error = error.data.detach().cpu().numpy()
                for i, traj in enumerate(trajectories):
                    replay_buffer.add(error[i], *traj)

            else:
                trajectories = []
                trajectories.append([sdf_st_align, action, sdf_ns_align, reward, done, sdf_g, sdf_g, goal_flag, next_goal_flag])

                ## HER ##
                if her and not done:
                    her_sample = sample_her_transitions(env, info)
                    for sample in her_sample:
                        reward_re, goal_re, done_re, block_success_re = sample
                        if round_sdf:
                            sdf_ns_align_round = sdf_module.make_round_sdf(sdf_ns_align)
                        # check success #
                        if round_sdf:
                            sdf_success_re = sdf_module.check_sdf_align(sdf_st_align, sdf_ns_align_round, env.num_blocks)
                        else:
                            sdf_success_re = sdf_module.check_sdf_align(sdf_st_align, sdf_ns_align, env.num_blocks)
                        goal_flag_re = pad_flag(sdf_success_re, max_blocks)
                        next_goal_flag_re = np.array([1]*env.num_blocks + [0]*(max_blocks-env.num_blocks)).astype(bool)
                        reward_re += 10
                        done_re = True
                        if round_sdf:
                            trajectories.append([sdf_st_align, action, sdf_ns_align, reward_re, done_re, sdf_ns_align_round, sdf_ns_align_round, goal_flag_re, next_goal_flag_re])
                        else:
                            trajectories.append([sdf_st_align, action, sdf_ns_align, reward_re, done_re, sdf_ns_align, sdf_ns_align, goal_flag_re, next_goal_flag_re])

                for traj in trajectories:
                    replay_buffer.add(*traj)

            if replay_buffer.size < learn_start:
                if done:
                    break
                else:
                    sdf_st_align = sdf_ns_align
                    goal_flag = next_goal_flag
                    continue
            elif replay_buffer.size == learn_start:
                epsilon = start_epsilon
                count_steps = 0
                break

            ## sample from replay buff & update networks ##
            data = [
                    torch.FloatTensor(pad_sdf(sdf_st_align, max_blocks, sdf_res)).to(device),
                    torch.FloatTensor(pad_sdf(sdf_ns_align, max_blocks, sdf_res)).to(device),
                    torch.FloatTensor(action).to(device),
                    torch.FloatTensor([reward]).to(device),
                    torch.FloatTensor([1 - done]).to(device),
                    torch.FloatTensor(pad_sdf(sdf_g, max_blocks, sdf_res)).to(device),
                    torch.FloatTensor(pad_sdf(sdf_g, max_blocks, sdf_res)).to(device),
                    torch.LongTensor([len(sdf_st_align)]).to(device),
                    torch.LongTensor([len(sdf_ns_align)]).to(device),
                    torch.FloatTensor(goal_flag).to(device),
                    torch.FloatTensor(next_goal_flag).to(device),
                    ]
            if per:
                minibatch, idxs, is_weights = replay_buffer.sample(batch_size-1)
                combined_minibatch = combine_batch(minibatch, data)
                loss, error = calculate_loss(combined_minibatch, qnet, qnet_target)
                errors = error.data.detach().cpu().numpy()[:-1]
                # update priority
                for i in range(batch_size-1):
                    idx = idxs[i]
                    replay_buffer.update(idx, errors[i])
            else:
                minibatch = replay_buffer.sample(batch_size-1)
                combined_minibatch = combine_batch(minibatch, data)
                loss, _ = calculate_loss(combined_minibatch, qnet, qnet_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            log_minibatchloss.append(loss.data.detach().cpu().numpy())

            label = torch.FloatTensor(goal_flag[:env.num_blocks]).to(device)
            s = pad_sdf(sdf_st_align, max_blocks, sdf_res)
            s = torch.FloatTensor(s).to(device)
            g = pad_sdf(sdf_g, max_blocks, sdf_res)
            g = torch.FloatTensor(g).to(device)
            predict = dnet([s, g], env.num_blocks)
            disc_loss = torch.pow(predict - label, 2).sum()
            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()
            log_minibatchDloss.append(disc_loss.data.detach().cpu().numpy())

            #num_collisions += int(info['collision'])
            if done:
                break
            else:
                sdf_st_align = sdf_ns_align
                goal_flag = next_goal_flag

        if replay_buffer.size <= learn_start:
            continue

        log_returns.append(episode_reward)
        log_loss.append(np.mean(log_minibatchloss))
        log_Dloss.append(np.mean(log_minibatchDloss))
        log_eplen.append(ep_len)
        log_epsilon.append(epsilon)
        log_out.append(int(info['out_of_range']))
        log_success.append(int(info['success']))
        log_sdfsuccess.append(int(info['sdf_success']))

        for o in range(env.num_blocks):
            log_success_block[o].append(int(info['block_success'][o]))

        eplog = {
                'reward': episode_reward,
                'loss': np.mean(log_minibatchloss),
                'episode length': ep_len,
                'epsilon': epsilon,
                'out of range': int(info['out_of_range']),
                'success rate': int(info['success']),
                'SDF success rate': int(info['sdf_success']),
                '1block success': np.mean(info['block_success']),
                'D_loss': np.mean(log_minibatchDloss)
                }
        wandb.log(eplog, count_steps)

        if ne % log_freq == 0:
            log_mean_returns = smoothing_log_same(log_returns, log_freq)
            log_mean_loss = smoothing_log_same(log_loss, log_freq)
            log_mean_Dloss = smoothing_log_same(log_Dloss, log_freq)
            log_mean_eplen = smoothing_log_same(log_eplen, log_freq)
            log_mean_out = smoothing_log_same(log_out, log_freq)
            log_mean_success = smoothing_log_same(log_success, log_freq)
            log_mean_sdfsuccess = smoothing_log_same(log_sdfsuccess, log_freq)
            for o in range(env.num_blocks):
                log_mean_success_block[o] = smoothing_log_same(log_success_block[o], log_freq)

            et = time.time()
            now = datetime.datetime.now().strftime("%m/%d %H:%M")
            interval = str(datetime.timedelta(0, int(et-st)))
            st = et
            print(f"{now}({interval}) / ep{ne} ({count_steps} steps)", end=" / ")
            print(f"SDF SR:{log_mean_sdfsuccess[-1]:.2f}", end=" / ")
            print(f"SR:{log_mean_success[-1]:.2f}", end=" / ")
            for o in range(env.num_blocks):
                print("B{0}:{1:.2f}".format(o+1, log_mean_success_block[o][-1]), end=" ")
            print("/ Reward:{0:.2f}".format(log_mean_returns[-1]), end="")
            print(" / Loss:{0:.5f}".format(log_mean_loss[-1]), end="")
            print(" / D Loss:{0:.5f}".format(log_mean_Dloss[-1]), end="")
            print(" / Eplen:{0:.1f}".format(log_mean_eplen[-1]), end="")
            print(" / OOR:{0:.2f}".format(log_mean_out[-1]), end="")

            log_list = [
                    log_returns,  # 0
                    log_loss,  # 1
                    log_eplen,  # 2
                    log_epsilon,  # 3
                    log_success,  # 4
                    log_sdfsuccess,  # 5
                    log_out,  # 6
                    log_success_block, #7
                    log_Dloss,  # 8
                    ]
            numpy_log = np.array(log_list, dtype=object)
            np.save('results/board/%s' %savename, numpy_log)

            if log_mean_success[-1] > max_success:
                max_success = log_mean_success[-1]
                torch.save(qnet.state_dict(), 'results/models/%s.pth' % savename)
                torch.save(dnet.state_dict(), 'results/models/D_%s.pth' % savename)
                print(" <- Highest SR. Saving the model.")
            else:
                print("")

        if ne % update_freq == 0:
            qnet_target.load_state_dict(qnet.state_dict())
            #lr_scheduler.step()
            epsilon = max(epsilon_decay * epsilon, min_epsilon)


    print('Training finished.')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # env config #
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--camera_height", default=480, type=int)
    parser.add_argument("--camera_width", default=480, type=int)
    parser.add_argument("--num_blocks", default=3, type=int)
    parser.add_argument("--max_blocks", default=8, type=int)
    parser.add_argument("--dist", default=0.06, type=float)
    parser.add_argument("--sdf_action", action="store_false")
    parser.add_argument("--real_object", action="store_false")
    parser.add_argument("--testset", action="store_true")
    parser.add_argument("--max_steps", default=100, type=int)
    # sdf #
    parser.add_argument("--convex_hull", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--tracker", default="medianflow", type=str)
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--clip", action="store_true")
    parser.add_argument("--round_sdf", action="store_true")
    parser.add_argument("--reward", default="linear_penalty", type=str)
    # learning params #
    parser.add_argument("--resize", action="store_false") # defalut: True
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--bs", default=16, type=int)
    parser.add_argument("--buff_size", default=1e3, type=float)
    parser.add_argument("--total_episodes", default=1e4, type=float)
    parser.add_argument("--learn_start", default=300, type=float)
    parser.add_argument("--update_freq", default=100, type=int)
    parser.add_argument("--log_freq", default=50, type=int)
    parser.add_argument("--double", action="store_false")
    parser.add_argument("--per", action="store_true")
    parser.add_argument("--her", action="store_false")
    # gcn #
    parser.add_argument("--ver", default=1, type=int)
    parser.add_argument("--normalize", action="store_true")
    # model #
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--continue_learning", action="store_true")
    parser.add_argument("--model_path", default="", type=str)
    # etc #
    parser.add_argument("--show_q", action="store_true")
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--gpu", default=-1, type=int)
    args = parser.parse_args()

    # random seed #
    seed = args.seed
    if seed is not None:
        print("Random seed:", seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # env configuration #
    render = args.render
    num_blocks = args.num_blocks
    max_blocks = args.max_blocks
    sdf_action = args.sdf_action
    real_object = args.real_object
    testset = args.testset
    depth = args.depth
    mov_dist = args.dist
    max_steps = args.max_steps
    camera_height = args.camera_height
    camera_width = args.camera_width
    reward_type = args.reward
    gpu = args.gpu

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if str(gpu) in visible_gpus:
            gpu_idx = visible_gpus.index(str(gpu))
            torch.cuda.set_device(gpu_idx)

    model_path = os.path.join("results/models/DQN_GF_%s.pth"%args.model_path)
    visualize_q = args.show_q

    now = datetime.datetime.now()
    savename = "DQN_GF_%s" % (now.strftime("%m%d_%H%M"))
    if not os.path.exists("results/config/"):
        os.makedirs("results/config/")
    with open("results/config/%s.json" % savename, 'w') as cf:
        json.dump(args.__dict__, cf, indent=2)
    
    convex_hull = args.convex_hull
    oracle_matching = args.oracle
    tracker = args.tracker
    resize = args.resize
    sdf_module = SDFModule(rgb_feature=True, resnet_feature=True, convex_hull=convex_hull, 
            binary_hole=True, using_depth=depth, tracker=tracker, resize=resize)
    if real_object:
        from realobjects_env import UR5Env
    else:
        from ur5_env import UR5Env
    env = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
            control_freq=5, data_format='NHWC', gpu=gpu, camera_depth=True, testset=testset)
    env = objectwise_env(env, num_blocks=num_blocks, mov_dist=mov_dist, max_steps=max_steps, \
            conti=False, detection=True, reward_type=reward_type)
    # learning configuration #
    learning_rate = args.lr
    batch_size = args.bs 
    buff_size = int(args.buff_size)
    total_episodes = int(args.total_episodes)
    learn_start = int(args.learn_start)
    update_freq = args.update_freq
    log_freq = args.log_freq
    double = args.double
    per = args.per
    her = args.her
    ver = args.ver
    graph_normalize = args.normalize
    clip_sdf = args.clip
    round_sdf = args.round_sdf

    pretrain = args.pretrain
    continue_learning = args.continue_learning
    if ver==1:
        # undirected graph
        # [   1      I
        #     I      I  ]
        from models.track_gcn import TrackQNetV1GF as QNet
        n_hidden = 8 #16
    elif ver==2:
        # directed graph
        # [   1      I
        #     0      I  ]
        from models.track_gcn import TrackQNetV2 as QNet
        n_hidden = 8 #16
    elif ver==3:
        # resolution: 480 x 480 
        # directed graph
        # [   1      I
        #     0      I  ]
        from models.track_gcn_v3 import TrackQNetV3 as QNet
        n_hidden = 64

    # wandb model name #
    if real_object:
        log_name = savename + '_real'
    else:
        log_name = savename + '_cube'
    log_name += '_%db' %num_blocks
    log_name += '_v%d' %ver
    wandb.init(project="SDFGCN")
    wandb.run.name = log_name
    wandb.config.update(args)
    wandb.run.save()


    learning(env=env, savename=savename, sdf_module=sdf_module, n_actions=8, n_hidden=n_hidden, \
            learning_rate=learning_rate, batch_size=batch_size, buff_size=buff_size, \
            total_episodes=total_episodes, learn_start=learn_start, update_freq=update_freq, \
            log_freq=log_freq, double=double, her=her, per=per, visualize_q=visualize_q, \
            continue_learning=continue_learning, model_path=model_path, pretrain=pretrain, \
            clip_sdf=clip_sdf, sdf_action=sdf_action, graph_normalize=graph_normalize, \
            max_blocks=max_blocks, oracle_matching=oracle_matching, round_sdf=round_sdf)
