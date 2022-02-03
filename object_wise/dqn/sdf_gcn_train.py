import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../../ur5_mujoco'))
from object_env import *
from training_utils import *

import torch
import torch.nn as nn
import argparse
import json

import copy
import time
import datetime
import random
import pylab

from sdf_module import SDFModule
from replay_buffer import ReplayBuffer, PER
from matplotlib import pyplot as plt

#dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def norm_npy(array):
    positive = array - array.min()
    return positive / positive.max()

def pad_sdf(sdf, nmax):
    h, w = 96, 96
    nsdf = len(sdf)
    padded = np.zeros([nmax, h, w])
    if nsdf > nmax:
        padded[:] = sdf[:nmax]
    elif nsdf > 0:
        padded[:nsdf] = sdf
    return padded

def get_action(env, max_blocks, qnet, sdf_raw, sdfs, epsilon, with_q=False, sdf_action=False):
    if np.random.random() < epsilon:
        #print('Random action')
        obj = np.random.randint(len(sdf_raw))
        theta = np.random.randint(env.num_bins)
        if with_q:
            nsdf = sdfs[0].shape[0]
            s = pad_sdf(sdfs[0], max_blocks)
            s = torch.FloatTensor(s).to(device).unsqueeze(0)
            g = pad_sdf(sdfs[1], max_blocks)
            g = torch.FloatTensor(g).to(device).unsqueeze(0)
            nsdf = torch.LongTensor([nsdf]).to(device)
            q_value = qnet([s, g], nsdf)
            q = q_value[0][:nsdf].detach().cpu().numpy()
    else:
        nsdf = sdfs[0].shape[0]
        s = pad_sdf(sdfs[0], max_blocks)
        s = torch.FloatTensor(s).to(device).unsqueeze(0)
        g = pad_sdf(sdfs[1], max_blocks)
        g = torch.FloatTensor(g).to(device).unsqueeze(0)
        nsdf = torch.LongTensor([nsdf]).to(device)
        q_value = qnet([s, g], nsdf)
        q = q_value[0][:nsdf].detach().cpu().numpy()

        obj = q.max(1).argmax()
        theta = q.max(0).argmax()

    action = [obj, theta]
    sdf_target = sdf_raw[obj]
    px, py = np.where(sdf_target==sdf_target.max())
    px = px[0]
    py = py[0]
    #print(px, py, theta)

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
        return action, [px, py, theta], mask, q
    else:
        return action, [px, py, theta], mask

def learning(env, 
        savename,
        sdf_module,
        n_actions=8,
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
        ):

    qnet = QNet(max_blocks, n_actions, normalize=graph_normalize).to(device)
    if pretrain:
        qnet.load_state_dict(torch.load(model_path))
        print('Loading pre-trained model: {}'.format(model_path))
    elif continue_learning:
        qnet.load_state_dict(torch.load(model_path))
        print('Loading trained model: {}'.format(model_path))
    qnet_target = QNet(max_blocks, n_actions, normalize=graph_normalize).to(device)
    qnet_target.load_state_dict(qnet.state_dict())

    #optimizer = torch.optim.SGD(qnet.parameters(), lr=learning_rate, momentum=0.9, weight_decay=2e-5)
    optimizer = torch.optim.Adam(qnet.parameters(), lr=learning_rate)

    if per:
        replay_buffer = PER([max_blocks, 96, 96], [max_blocks, 96, 96], max_size=int(buff_size))
    else:
        replay_buffer = ReplayBuffer([max_blocks, 96, 96], [max_blocks, 96, 96], max_size=int(buff_size))

    model_parameters = filter(lambda p: p.requires_grad, qnet.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("# of params: %d"%params)

    if double:
        calculate_loss = calculate_loss_gcn_double
    else:
        calculate_loss = calculate_loss_gcn_origin

    if continue_learning and not pretrain:
        numpy_log = np.load(model_path.replace('models/', 'board/').replace('.pth', '.npy'), allow_pickle=True)
        log_returns = list(numpy_log[0])
        log_loss = list(numpy_log[1])
        log_eplen = list(numpy_log[2])
        log_epsilon = list(numpy_log[3])
        log_success = list(numpy_log[4])
        #log_collisions = list(numpy_log[5])
        log_sdf_mismatch = list(numpy_log[5])
        log_out = list(numpy_log[6])
        log_success_block = list(numpy_log[7])
        log_mean_success_block = [[] for _ in range(env.num_blocks)]
    else:
        log_returns = []
        log_loss = []
        log_eplen = []
        log_epsilon = []
        log_success = []
        #log_collisions = []
        log_sdf_mismatch= []
        log_out = []
        log_success_block = [[] for _ in range(env.num_blocks)]
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
    f, axes = plt.subplots(3, 3) # 3,2
    f.set_figheight(12) #9 #15
    f.set_figwidth(20) #12 #10

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
    #axes[2][2].set_title('Num Collisions')  # 9
    axes[2][2].set_title('SDF mismatch')  # 9

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

        check_env_ready = False
        while not check_env_ready:
            (state_img, goal_img) = env.reset()
            sdf_st, sdf_raw, feature_st = sdf_module.get_sdf_features(state_img, clip=clip_sdf)
            sdf_g, _, feature_g = sdf_module.get_sdf_features(goal_img, clip=clip_sdf)
            check_env_ready = (len(sdf_g)==env.num_blocks) & (len(sdf_st)!=0)
            if not check_env_ready:
                continue
            # target: st / source: g
            matching = sdf_module.object_matching(feature_g, feature_st)
            sdf_g_align = sdf_module.align_sdf(matching, sdf_g, sdf_st)

        mismatch = len(sdf_st)!=env.num_blocks
        num_mismatch = int(mismatch) 

        if visualize_q:
            if env.env.camera_depth:
                ax0.imshow(goal_img[0])
                ax1.imshow(state_img[0])
            else:
                ax0.imshow(goal_img)
                ax1.imshow(state_img)
            # goal sdfs
            vis_g = norm_npy(sdf_g_align + 2*(sdf_g_align>0).astype(float))
            goal_sdfs = np.zeros([96, 96, 3])
            for _s in range(len(vis_g)):
                goal_sdfs += np.expand_dims(vis_g[_s], 2) * np.array(cm(_s/5)[:3])
            ax2.imshow(norm_npy(goal_sdfs))
            # current sdfs
            vis_c = norm_npy(sdf_st + 2*(sdf_st>0).astype(float))
            current_sdfs = np.zeros([96, 96, 3])
            for _s in range(len(vis_c)):
                current_sdfs += np.expand_dims(vis_c[_s], 2) * np.array(cm(_s/5)[:3])
            ax3.imshow(norm_npy(current_sdfs))
            plt.show(block=False)
            fig.canvas.draw()

        for t_step in range(env.max_steps):
            count_steps += 1
            ep_len += 1
            action, pixel_action, sdf_mask, q_map = get_action(env, max_blocks, qnet, sdf_raw, \
                    [sdf_st, sdf_g_align], epsilon=epsilon, with_q=True, sdf_action=sdf_action)

            (next_state_img, _), reward, done, info = env.step(pixel_action, sdf_mask)
            episode_reward += reward
            sdf_ns, sdf_raw, feature_ns = sdf_module.get_sdf_features(next_state_img, clip=clip_sdf)
            matching = sdf_module.object_matching(feature_g, feature_ns)
            sdf_ng_align = sdf_module.align_sdf(matching, sdf_g, sdf_ns)

            # detection failed #
            if len(sdf_ns) == 0:
                reward = -1.
                done = True

            if visualize_q:
                if env.env.camera_depth:
                    ax1.imshow(next_state_img[0])
                else:
                    ax1.imshow(next_state_img)

                # goal sdfs
                vis_g = norm_npy(sdf_ng_align + 2*(sdf_ng_align>0).astype(float))
                goal_sdfs = np.zeros([96, 96, 3])
                for _s in range(len(vis_g)):
                    goal_sdfs += np.expand_dims(vis_g[_s], 2) * np.array(cm(_s/5)[:3])
                ax2.imshow(norm_npy(goal_sdfs))
                # current sdfs
                vis_c = norm_npy(sdf_ns + 2*(sdf_ns>0).astype(float))
                current_sdfs = np.zeros([96, 96, 3])
                for _s in range(len(vis_c)):
                    current_sdfs += np.expand_dims(vis_c[_s], 2) * np.array(cm(_s/5)[:3])
                ax3.imshow(norm_npy(current_sdfs))
                fig.canvas.draw()

            ## save transition to the replay buffer ##
            mismatch = len(sdf_ns)!=env.num_blocks
            num_mismatch += int(mismatch) 
            if per:
                trajectories = []
                replay_tensors = []

                trajectories.append([sdf_st, action, sdf_ns, reward, done, sdf_g_align, sdf_ng_align])

                traj_tensor = [
                    torch.FloatTensor(pad_sdf(sdf_st, max_blocks)).to(device),
                    torch.FloatTensor(pad_sdf(sdf_ns, max_blocks)).to(device),
                    torch.FloatTensor(action).to(device),
                    torch.FloatTensor([reward]).to(device),
                    torch.FloatTensor([1 - done]).to(device),
                    torch.FloatTensor(pad_sdf(sdf_g_align, max_blocks)).to(device),
                    torch.FloatTensor(pad_sdf(sdf_ng_align, max_blocks)).to(device),
                    torch.LongTensor([len(sdf_st)]).to(device),
                    torch.LongTensor([len(sdf_ns)]).to(device),
                ]
                replay_tensors.append(traj_tensor)

                ## HER ##
                if her and not done:
                    her_sample = sample_her_transitions(env, info)
                    for sample in her_sample:
                        reward_re, goal_re, done_re, block_success_re = sample

                        matching = sdf_module.object_matching(feature_ns, feature_st)
                        sdf_ns_align = sdf_module.align_sdf(matching, sdf_ns, sdf_st)
                        trajectories.append([sdf_st, action, sdf_ns, reward_re, done_re, sdf_ns_align, sdf_ns])
                        traj_tensor = [
                            torch.FloatTensor(pad_sdf(sdf_st, max_blocks)).to(device),
                            torch.FloatTensor(pad_sdf(sdf_ns, max_blocks)).to(device),
                            torch.FloatTensor(action).to(device),
                            torch.FloatTensor([reward_re]).to(device),
                            torch.FloatTensor([1 - done_re]).to(device),
                            torch.FloatTensor(pad_sdf(sdf_ns_align, max_blocks)).to(device),
                            torch.FloatTensor(pad_sdf(sdf_ns, max_blocks)).to(device),
                            torch.LongTensor([len(sdf_st)]).to(device),
                            torch.LongTensor([len(sdf_ns)]).to(device),
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
                trajectories.append([sdf_st, action, sdf_ns, reward, done, sdf_g_align, sdf_ng_align])

                ## HER ##
                if her and not done:
                    her_sample = sample_her_transitions(env, info)
                    for sample in her_sample:
                        reward_re, goal_re, done_re, block_success_re = sample
                        matching = sdf_module.object_matching(feature_ns, feature_st)
                        sdf_ns_align = sdf_module.align_sdf(matching, sdf_ns, sdf_st)
                        trajectories.append([sdf_st, action, sdf_ns, reward_re, done_re, sdf_ns_align, sdf_ns])

                for traj in trajectories:
                    replay_buffer.add(*traj)

            if replay_buffer.size < learn_start:
                if done:
                    break
                else:
                    sdf_st = sdf_ns
                    feature_st = feature_ns
                    sdf_g_align = sdf_ng_align
                    continue
            elif replay_buffer.size == learn_start:
                epsilon = start_epsilon
                break

            ## sample from replay buff & update networks ##
            data = [
                    torch.FloatTensor(pad_sdf(sdf_st, max_blocks)).to(device),
                    torch.FloatTensor(pad_sdf(sdf_ns, max_blocks)).to(device),
                    torch.FloatTensor(action).to(device),
                    torch.FloatTensor([reward]).to(device),
                    torch.FloatTensor([1 - done]).to(device),
                    torch.FloatTensor(pad_sdf(sdf_g_align, max_blocks)).to(device),
                    torch.FloatTensor(pad_sdf(sdf_ng_align, max_blocks)).to(device),
                    torch.LongTensor([len(sdf_st)]).to(device),
                    torch.LongTensor([len(sdf_ns)]).to(device),
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

            #num_collisions += int(info['collision'])
            if done:
                break
            else:
                sdf_st = sdf_ns
                feature_st = feature_ns
                sdf_g_align = sdf_ng_align

        if replay_buffer.size <= learn_start:
            continue

        log_returns.append(episode_reward)
        log_loss.append(np.mean(log_minibatchloss))
        log_eplen.append(ep_len)
        log_epsilon.append(epsilon)
        log_out.append(int(info['out_of_range']))
        log_success.append(int(info['success']))
        #log_collisions.append(num_collisions)
        log_sdf_mismatch.append(num_mismatch)

        for o in range(env.num_blocks):
            log_success_block[o].append(int(info['block_success'][o]))

        if ne % log_freq == 0:
            log_mean_returns = smoothing_log_same(log_returns, log_freq)
            log_mean_loss = smoothing_log_same(log_loss, log_freq)
            log_mean_eplen = smoothing_log_same(log_eplen, log_freq)
            log_mean_out = smoothing_log_same(log_out, log_freq)
            log_mean_success = smoothing_log_same(log_success, log_freq)
            for o in range(env.num_blocks):
                log_mean_success_block[o] = smoothing_log_same(log_success_block[o], log_freq)
            #log_mean_collisions = smoothing_log_same(log_collisions, log_freq)
            log_mean_sdf_mismatch = smoothing_log_same(log_sdf_mismatch, log_freq)

            et = time.time()
            now = datetime.datetime.now()
            print()
            print("{} - {} seconds".format(now.strftime("%H%M"), et-st))
            print("{}/{} episodes. ({} steps)".format(ne, total_episodes, count_steps))
            print("Success rate: {0:.2f}".format(log_mean_success[-1]))
            for o in range(env.num_blocks):
                print("Block {0}: {1:.2f}".format(o+1, log_mean_success_block[o][-1]))
            print("Mean reward: {0:.2f}".format(log_mean_returns[-1]))
            print("Mean loss: {0:.6f}".format(log_mean_loss[-1]))
            print("Ep length: {}".format(log_mean_eplen[-1]))
            print("Epsilon: {}".format(epsilon))

            axes[1][2].plot(log_loss, color='#ff7f00', linewidth=0.5)  # 3->6
            axes[1][1].plot(log_returns, color='#60c7ff', linewidth=0.5)  # 5
            axes[2][0].plot(log_eplen, color='#83dcb7', linewidth=0.5)  # 7
            #axes[2][2].plot(log_collisions, color='#ff33cc', linewidth=0.5)  # 8->9

            for o in range(3): #env.num_blocks
                axes[0][o].plot(log_mean_success_block[o], color='red')  # 1,2,3

            axes[1][2].plot(log_mean_loss, color='red')  # 3->6
            axes[1][1].plot(log_mean_returns, color='blue')  # 5
            axes[2][0].plot(log_mean_eplen, color='green')  # 7
            axes[1][0].plot(log_mean_success, color='red')  # 4
            axes[2][1].plot(log_mean_out, color='black')  # 6->8
            #axes[2][2].plot(log_mean_collisions, color='#663399')  # 8->9
            axes[2][2].plot(log_mean_sdf_mismatch, color='#663399')  # 8->9

            f.savefig('results/graph/%s.png' % savename)

            log_list = [
                    log_returns,  # 0
                    log_loss,  # 1
                    log_eplen,  # 2
                    log_epsilon,  # 3
                    log_success,  # 4
                    log_sdf_mismatch, #log_collisions,  # 5
                    log_out,  # 6
                    log_success_block, #7
                    ]
            numpy_log = np.array(log_list)
            np.save('results/board/%s' %savename, numpy_log)

            if log_mean_success[-1] > max_success:
                max_success = log_mean_success[-1]
                torch.save(qnet.state_dict(), 'results/models/%s.pth' % savename)
                print("Max performance! saving the model.")

        if ne % update_freq == 0:
            qnet_target.load_state_dict(qnet.state_dict())
            #lr_scheduler.step()
            epsilon = max(epsilon_decay * epsilon, min_epsilon)


    print('Training finished.')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--num_blocks", default=3, type=int)
    parser.add_argument("--max_blocks", default=8, type=int)
    parser.add_argument("--dist", default=0.06, type=float)
    parser.add_argument("--sdf_action", action="store_false") # default: True
    parser.add_argument("--real_object", action="store_true") # default: False
    parser.add_argument("--depth", action="store_false") # default: True
    parser.add_argument("--max_steps", default=100, type=int)
    parser.add_argument("--camera_height", default=480, type=int)
    parser.add_argument("--camera_width", default=480, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--bs", default=6, type=int)
    parser.add_argument("--buff_size", default=1e3, type=float)
    parser.add_argument("--total_episodes", default=1e4, type=float)
    parser.add_argument("--learn_start", default=300, type=float)
    parser.add_argument("--update_freq", default=100, type=int)
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument("--double", action="store_false") # default: True
    parser.add_argument("--per", action="store_true") # default: False
    parser.add_argument("--her", action="store_false") # default: True
    parser.add_argument("--ver", default=3, type=int)
    parser.add_argument("--normalize", action="store_true") # default: False
    parser.add_argument("--clip", action="store_true") # default: False
    parser.add_argument("--reward", default="new", type=str)
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--continue_learning", action="store_true")
    parser.add_argument("--model_path", default="", type=str)
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
    depth = args.depth
    mov_dist = args.dist
    max_steps = args.max_steps
    camera_height = args.camera_height
    camera_width = args.camera_width
    reward_type = args.reward
    gpu = args.gpu

    if gpu!=-1:
        torch.cuda.set_device(gpu)

    model_path = os.path.join("results/models/SDF_%s.pth"%args.model_path)
    visualize_q = args.show_q

    now = datetime.datetime.now()
    savename = "SDF_%s" % (now.strftime("%m%d_%H%M"))
    if not os.path.exists("results/config/"):
        os.makedirs("results/config/")
    with open("results/config/%s.json" % savename, 'w') as cf:
        json.dump(args.__dict__, cf, indent=2)

    sdf_module = SDFModule()
    if real_object:
        from realobjects_env import UR5Env
    else:
        from ur5_env import UR5Env
    env = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
            control_freq=5, data_format='NHWC', gpu=gpu, camera_depth=depth)
    env = objectwise_env(env, num_blocks=num_blocks, mov_dist=mov_dist,max_steps=max_steps,\
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

    pretrain = args.pretrain
    continue_learning = args.continue_learning
    if ver==1:
        from models.sdf_gcn import SDFGCNQNet as QNet
    elif ver==2:
        # ver2: separate edge
        from models.sdf_gcn import SDFGCNQNetV2 as QNet
    elif ver==3:
        # ver3: block flags - 1 for block's sdf, 0 for goal's sdf
        from models.sdf_gcn import SDFGCNQNetV3 as QNet
    elif ver==4:
        # ver4: ch1-sdf, ch2-boundary, ch3-block flags
        from models.sdf_gcn import SDFGCNQNetV4 as QNet

    learning(env=env, savename=savename, sdf_module=sdf_module, n_actions=8, \
            learning_rate=learning_rate, batch_size=batch_size, buff_size=buff_size, \
            total_episodes=total_episodes, learn_start=learn_start, update_freq=update_freq, \
            log_freq=log_freq, double=double, her=her, per=per, visualize_q=visualize_q, \
            continue_learning=continue_learning, model_path=model_path, pretrain=pretrain, \
            clip_sdf=clip_sdf, sdf_action=sdf_action, graph_normalize=graph_normalize, \
            max_blocks=max_blocks)
