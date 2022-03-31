import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../../ur5_mujoco'))
sys.path.append(os.path.join(FILE_PATH, '..'))
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
from replay_buffer import GATReplayBuffer
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

def get_pose_feature(sdfs, scene_flag=0, scale=0.1):
    pose_feature = []
    for i in range(len(sdfs)):
        px, py = np.where(sdfs[i]==sdfs[i].max())
        cx = px[0] * scale
        cy = py[0] * scale
        pose_feature.append([scene_flag, cx, cy])
    pose_feature = np.array(pose_feature)
    return pose_feature

def pad_feature(feature, nmax, fdim):
    nb = len(feature)
    padded = np.zeros([nmax, fdim])
    if nb > nmax:
        padded[:] = feature[:nmax]
    elif nb > 0:
        padded[:nb] = feature
    return padded

def pad_sdf(sdf, nmax):
    h, w = 96, 96
    nsdf = len(sdf)
    padded = np.zeros([nmax, h, w])
    if nsdf > nmax:
        padded[:] = sdf[:nmax]
    elif nsdf > 0:
        padded[:nsdf] = sdf
    return padded

def get_action(env, max_blocks, qnet, sdf_raw, features, sdfs, fdim, epsilon, with_q=False, sdf_action=False):
    if np.random.random() < epsilon:
        #print('Random action')
        obj = np.random.randint(len(sdf_raw))
        theta = np.random.randint(env.num_bins)
        if with_q:
            nb_st = sdfs[0].shape[0]
            nb_g = sdfs[1].shape[0]
            f_st = pad_feature(features[0], max_blocks, fdim)
            f_g = pad_feature(features[1], max_blocks, fdim)
            sdf_st = pad_sdf(sdfs[0], max_blocks)
            sdf_g = pad_sdf(sdfs[1], max_blocks)

            f_st = torch.FloatTensor(f_st).to(device).unsqueeze(0)
            f_g = torch.FloatTensor(f_g).to(device).unsqueeze(0)
            sdf_st = torch.FloatTensor(sdf_st).to(device).unsqueeze(0)
            sdf_g = torch.FloatTensor(sdf_g).to(device).unsqueeze(0)
            nb_st = torch.LongTensor([nb_st]).to(device)
            nb_g = torch.LongTensor([nb_g]).to(device)

            q_value = qnet([f_st, sdf_st], [f_g, sdf_g], nb_st, nb_g)
            q = q_value[0][:nb_st].detach().cpu().numpy()
    else:
        nb_st = sdfs[0].shape[0]
        nb_g = sdfs[1].shape[0]
        f_st = pad_feature(features[0], max_blocks, fdim)
        f_g = pad_feature(features[1], max_blocks, fdim)
        sdf_st = pad_sdf(sdfs[0], max_blocks)
        sdf_g = pad_sdf(sdfs[1], max_blocks)

        f_st = torch.FloatTensor(f_st).to(device).unsqueeze(0)
        f_g = torch.FloatTensor(f_g).to(device).unsqueeze(0)
        sdf_st = torch.FloatTensor(sdf_st).to(device).unsqueeze(0)
        sdf_g = torch.FloatTensor(sdf_g).to(device).unsqueeze(0)
        nb_st = torch.LongTensor([nb_st]).to(device)
        nb_g = torch.LongTensor([nb_g]).to(device)

        q_value = qnet([f_st, sdf_st], [f_g, sdf_g], nb_st, nb_g)
        q = q_value[0][:nb_st].detach().cpu().numpy()

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
        her=True,
        visualize_q=False,
        pretrain=False,
        continue_learning=False,
        model_path='',
        clip_sdf=False,
        sdf_action=False,
        max_blocks=5,
        sdf_penalty=False,
        ):

    print('='*30)
    print('{} learing starts.'.format(savename))
    print('='*30)
    fdim = sdf_module.fdim + 3 # 3-dim additional features: [flag, x, y]
    qnet = QNet(max_blocks, sdim=1, fdim=fdim, n_actions=n_actions).to(device)
    if pretrain:
        qnet.load_state_dict(torch.load(model_path))
        print('Loading pre-trained model: {}'.format(model_path))
    elif continue_learning:
        qnet.load_state_dict(torch.load(model_path))
        print('Loading trained model: {}'.format(model_path))
    qnet_target = QNet(max_blocks, sdim=1, fdim=fdim, n_actions=n_actions).to(device)
    qnet_target.load_state_dict(qnet.state_dict())

    #optimizer = torch.optim.SGD(qnet.parameters(), lr=learning_rate, momentum=0.9, weight_decay=2e-5)
    optimizer = torch.optim.Adam(qnet.parameters(), lr=learning_rate)

    replay_buffer = GATReplayBuffer([max_blocks, fdim], [max_blocks, 96, 96], max_size=int(buff_size))

    model_parameters = filter(lambda p: p.requires_grad, qnet.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("# of params: %d"%params)

    if double:
        calculate_loss = calculate_loss_gat_double
    else:
        calculate_loss = calculate_loss_gat_origin

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
            (state_img, goal_img), info = env.reset()
            sdf_st, sdf_raw, feature_st = sdf_module.get_sdf_features(state_img[0], state_img[1], env.num_blocks, clip=clip_sdf)
            sdf_g, _, feature_g = sdf_module.get_sdf_features(goal_img[0], goal_img[1], env.num_blocks, clip=clip_sdf)
            check_env_ready = (len(sdf_g)==env.num_blocks) & (len(sdf_st)!=0)
            if not check_env_ready:
                continue
            n_detection = len(sdf_st)

            feature_st = [get_pose_feature(sdf_st, 0)] + feature_st
            feature_g = [get_pose_feature(sdf_g, 1)] + feature_g
            feature_st = np.concatenate(feature_st, len(feature_st[0].shape)-1)
            feature_g = np.concatenate(feature_g, len(feature_g[0].shape)-1)

        mismatch = (n_detection!=env.num_blocks)
        num_mismatch = int(mismatch) 

        if visualize_q:
            if env.env.camera_depth:
                ax0.imshow(goal_img[0])
                ax1.imshow(state_img[0])
            else:
                ax0.imshow(goal_img)
                ax1.imshow(state_img)
            # goal sdfs
            vis_g = norm_npy(sdf_g + 50*(sdf_g>0).astype(float))
            goal_sdfs = np.zeros([96, 96, 3])
            for _s in range(len(vis_g)):
                goal_sdfs += np.expand_dims(vis_g[_s], 2) * np.array(cm(_s/5)[:3])
            ax2.imshow(norm_npy(goal_sdfs))
            # current sdfs
            vis_c = norm_npy(sdf_st + 50*(sdf_st>0).astype(float))
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
                    [feature_st, feature_g], [sdf_st, sdf_g], fdim, epsilon, with_q=True, \
                    sdf_action=sdf_action)

            (next_state_img, _), reward, done, info = env.step(pixel_action, sdf_mask)
            episode_reward += reward
            sdf_ns, sdf_raw, feature_ns = sdf_module.get_sdf_features(next_state_img[0], next_state_img[1], env.num_blocks, clip=clip_sdf)
            pre_n_detection = n_detection
            n_detection = len(sdf_ns)

            feature_ns = [get_pose_feature(sdf_ns, 0)] + feature_ns
            feature_ns = np.concatenate(feature_ns, len(feature_ns[0].shape)-1)

            mismatch = (n_detection!=env.num_blocks)
            num_mismatch += int(mismatch) 

            # detection failed #
            if n_detection == 0:
                reward = -1.
                done = True

            # mismatch penalty v2 #
            if sdf_penalty and n_detection<pre_n_detection: #len(sdf_ns) < len(sdf_st):
                reward -= 0.5

            if visualize_q:
                if env.env.camera_depth:
                    ax1.imshow(next_state_img[0])
                else:
                    ax1.imshow(next_state_img)

                # goal sdfs
                vis_g = norm_npy(sdf_g + 50*(sdf_g>0).astype(float))
                goal_sdfs = np.zeros([96, 96, 3])
                for _s in range(len(vis_g)):
                    goal_sdfs += np.expand_dims(vis_g[_s], 2) * np.array(cm(_s/5)[:3])
                ax2.imshow(norm_npy(goal_sdfs))
                # current sdfs
                vis_c = norm_npy(sdf_ns + 50*(sdf_ns>0).astype(float))
                current_sdfs = np.zeros([96, 96, 3])
                for _s in range(len(vis_c)):
                    current_sdfs += np.expand_dims(vis_c[_s], 2) * np.array(cm(_s/5)[:3])
                ax3.imshow(norm_npy(current_sdfs))
                fig.canvas.draw()

            trajectories = []
            trajectories.append([feature_st, feature_ns, feature_g, sdf_st, sdf_ns, sdf_g, action, reward, done])

            ## HER ##
            if her and not done:
                her_sample = sample_her_transitions(env, info)
                for sample in her_sample:
                    reward_re, goal_re, done_re, block_success_re = sample
                    if sdf_penalty and len(sdf_ns) < len(sdf_st):
                        reward_re -= 0.5
                    feature_ns_re = copy.deepcopy(feature_ns)
                    feature_ns_re[:, 0] = 1

                    trajectories.append([feature_st, feature_ns, feature_ns_re, sdf_st, sdf_ns, sdf_ns, action, reward_re, done_re])

            for traj in trajectories:
                replay_buffer.add(*traj)

            if replay_buffer.size < learn_start:
                if done:
                    break
                else:
                    sdf_st = sdf_ns
                    feature_st = feature_ns
                    continue
            elif replay_buffer.size == learn_start:
                epsilon = start_epsilon
                break

            ## sample from replay buff & update networks ##
            data = [
                    torch.FloatTensor(pad_feature(feature_st, max_blocks, fdim)).to(device),
                    torch.FloatTensor(pad_feature(feature_ns, max_blocks, fdim)).to(device),
                    torch.FloatTensor(pad_feature(feature_g, max_blocks, fdim)).to(device),
                    torch.FloatTensor(pad_sdf(sdf_st, max_blocks)).to(device),
                    torch.FloatTensor(pad_sdf(sdf_ns, max_blocks)).to(device),
                    torch.FloatTensor(pad_sdf(sdf_g, max_blocks)).to(device),
                    torch.FloatTensor(action).to(device),
                    torch.FloatTensor([reward]).to(device),
                    torch.FloatTensor([1 - done]).to(device),
                    torch.LongTensor([len(sdf_st)]).to(device),
                    torch.LongTensor([len(sdf_ns)]).to(device),
                    torch.LongTensor([len(sdf_g)]).to(device),
                    ]

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

        eplog = {
                'reward': episode_reward,
                'loss': np.mean(log_minibatchloss),
                'episode length': ep_len,
                'epsilon': epsilon,
                'out of range': int(info['out_of_range']),
                'success rate': int(info['success']),
                'sdf_mismatch': num_mismatch,
                '1block_success': np.mean(info['block_success'])
                }
        wandb.log(eplog)

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
            now = datetime.datetime.now().strftime("%m/%d %H:%M")
            interval = str(datetime.timedelta(0, int(et-st)))
            st = et
            print(f"{now}({interval}) / ep{ne} ({count_steps} steps)", end=" / ")
            print(f"SR:{log_mean_success[-1]:.2f}", end=" / ")
            for o in range(env.num_blocks):
                print("B{0}:{1:.2f}".format(o+1, log_mean_success_block[o][-1]), end=" ")
            print("/ reward:{0:.2f}".format(log_mean_returns[-1]), end="")
            print(" / loss:{0:.5f}".format(log_mean_loss[-1]), end="")
            print(" / Eplen:{0:.1f}".format(log_mean_eplen[-1]), end="")


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
            numpy_log = np.array(log_list, dtype=object)
            np.save('results/board/%s' %savename, numpy_log)

            if log_mean_success[-1] > max_success:
                max_success = log_mean_success[-1]
                torch.save(qnet.state_dict(), 'results/models/%s.pth' % savename)
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
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--num_blocks", default=3, type=int)
    parser.add_argument("--max_blocks", default=8, type=int)
    parser.add_argument("--dist", default=0.06, type=float)
    parser.add_argument("--sdf_action", action="store_false")
    parser.add_argument("--convex_hull", action="store_false")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--real_object", action="store_true")
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--max_steps", default=100, type=int)
    parser.add_argument("--camera_height", default=480, type=int)
    parser.add_argument("--camera_width", default=480, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--bs", default=6, type=int)
    parser.add_argument("--buff_size", default=1e3, type=float)
    parser.add_argument("--total_episodes", default=1e4, type=float)
    parser.add_argument("--learn_start", default=300, type=float)
    parser.add_argument("--update_freq", default=100, type=int)
    parser.add_argument("--log_freq", default=50, type=int)
    parser.add_argument("--double", action="store_false")
    parser.add_argument("--her", action="store_false")
    parser.add_argument("--ver", default=1, type=int)
    parser.add_argument("--clip", action="store_true")
    parser.add_argument("--penalty", action="store_true")
    parser.add_argument("--reward", default="linear", type=str)
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

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if str(gpu) in visible_gpus:
            gpu_idx = visible_gpus.index(str(gpu))
            torch.cuda.set_device(gpu_idx)

    model_path = os.path.join("results/models/GAT_%s.pth"%args.model_path)
    visualize_q = args.show_q

    now = datetime.datetime.now()
    savename = "GAT_%s" % (now.strftime("%m%d_%H%M"))
    if not os.path.exists("results/config/"):
        os.makedirs("results/config/")
    with open("results/config/%s.json" % savename, 'w') as cf:
        json.dump(args.__dict__, cf, indent=2)

    convex_hull = args.convex_hull
    sdf_module = SDFModule(rgb_feature=True, ucn_feature=False, resnet_feature=True, 
            convex_hull=convex_hull, binary_hole=True, using_depth=depth)
    if real_object:
        from realobjects_env import UR5Env
    else:
        from ur5_env import UR5Env
    env = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
            control_freq=5, data_format='NHWC', gpu=gpu, camera_depth=True)
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
    her = args.her
    ver = args.ver
    clip_sdf = args.clip
    sdf_penalty = args.penalty

    pretrain = args.pretrain
    continue_learning = args.continue_learning
    if ver==0:
        from models.sdf_gat import SDFGATQNet as QNet
    if ver==1:
        from models.sdf_gat import SDFGATQNetV1 as QNet

    # wandb model name #
    if real_object:
        log_name = savename + '_real'
    else:
        log_name = savename + '_cube'
    log_name += '_%db' %num_blocks
    log_name += '_v%d' %ver
    wandb.init(project="ur5-pushing")
    wandb.run.name = log_name
    wandb.config.update(args)
    wandb.run.save()


    learning(env=env, savename=savename, sdf_module=sdf_module, n_actions=8, \
            learning_rate=learning_rate, batch_size=batch_size, buff_size=buff_size, \
            total_episodes=total_episodes, learn_start=learn_start, update_freq=update_freq, \
            log_freq=log_freq, double=double, her=her, visualize_q=visualize_q, \
            continue_learning=continue_learning, model_path=model_path, pretrain=pretrain, \
            clip_sdf=clip_sdf, sdf_action=sdf_action, \
            max_blocks=max_blocks, sdf_penalty=sdf_penalty)
