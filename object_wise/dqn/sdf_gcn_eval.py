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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def norm_npy(array):
    positive = array - array.min()
    return positive / positive.max()

def pad_sdf(sdf, nmax):
    h, w = 96, 96
    nsdf = len(sdf)
    padded = np.zeros([nmax, h, w])
    if nsdf > 0:
        padded[:nsdf] = sdf
    return padded

def get_action(env, qnet, sdf_raw, sdfs, epsilon, with_q=False, sdf_action=False):
    if np.random.random() < epsilon:
        #print('Random action')
        obj = np.random.randint(len(sdf_raw))
        theta = np.random.randint(env.num_bins)
        if with_q:
            nsdf = sdfs[0].shape[0]
            s = pad_sdf(sdfs[0], env.num_blocks+2)
            s = torch.FloatTensor(s).to(device).unsqueeze(0)
            g = pad_sdf(sdfs[1], env.num_blocks+2)
            g = torch.FloatTensor(g).to(device).unsqueeze(0)
            nsdf = torch.LongTensor([nsdf]).to(device)
            q_value = qnet([s, g], nsdf)
            q = q_value[0][:nsdf].detach().cpu().numpy()
    else:
        nsdf = sdfs[0].shape[0]
        s = pad_sdf(sdfs[0], env.num_blocks+2)
        s = torch.FloatTensor(s).to(device).unsqueeze(0)
        g = pad_sdf(sdfs[1], env.num_blocks+2)
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

def evaluate(env,
        sdf_module,
        n_actions=8,
        model_path='',
        num_trials=100,
        visualize_q=False,
        clip_sdf=False,
        sdf_action=False,
        graph_normalize=False
        ):
    qnet = QNet(env.num_blocks + 2, n_actions, normalize=graph_normalize).to(device)
    qnet.load_state_dict(torch.load(model_path))
    print('Loading trained model: {}'.format(model_path))

    log_returns = []
    log_eplen = []
    log_out = []
    log_sdf_mismatch = []
    log_success = []
    log_success_block = [[] for i in range(env.num_blocks)]

    if visualize_q:
        plt.rc('axes', labelsize=6)
        plt.rc('font', size=8)

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

        cm = pylab.get_cmap('gist_rainbow')

    epsilon = 0.1
    for ne in range(num_trials):
        ep_len = 0
        episode_reward = 0

        check_env_ready = False
        while not check_env_ready:
            (state_img, goal_img) = env.reset()
            sdf_st, sdf_raw, feature_st = sdf_module.get_sdf_features(state_img, clip=clip_sdf)
            sdf_g, _, feature_g = sdf_module.get_sdf_features(goal_img, clip=clip_sdf)
            matching = sdf_module.object_matching(feature_st, feature_g)
            sdf_st_align = sdf_st[matching]
            sdf_raw = sdf_raw[matching]
            check_env_ready = (len(sdf_g)==env.num_blocks) & (len(sdf_st_align)!=0)

        mismatch = len(sdf_st_align)!=env.num_blocks or len(sdf_g)!=env.num_blocks
        num_mismatch = int(mismatch) 

        if visualize_q:
            ax0.imshow(goal_img)
            ax1.imshow(state_img)
            # goal sdfs
            vis_g = norm_npy(sdf_g + 2*(sdf_g>0).astype(float))
            goal_sdfs = np.zeros([96, 96, 3])
            for _s in range(len(vis_g)):
                goal_sdfs += np.expand_dims(vis_g[_s], 2) * np.array(cm(_s/5)[:3])
            ax2.imshow(norm_npy(goal_sdfs))
            # current sdfs
            vis_c = norm_npy(sdf_st_align + 2*(sdf_st_align>0).astype(float))
            current_sdfs = np.zeros([96, 96, 3])
            for _s in range(len(vis_c)):
                current_sdfs += np.expand_dims(vis_c[_s], 2) * np.array(cm(_s/5)[:3])
            ax3.imshow(norm_npy(current_sdfs))
            fig.canvas.draw()

        for t_step in range(env.max_steps):
            ep_len += 1
            action, pixel_action, sdf_mask, q_map = get_action(env, qnet, sdf_raw, \
                    [sdf_st_align, sdf_g], epsilon=epsilon, with_q=True, sdf_action=sdf_action)

            (next_state_img, _), reward, done, info = env.step(pixel_action, sdf_mask)
            episode_reward += reward
            # print(info['block_success'])

            sdf_ns, sdf_raw, feature_ns = sdf_module.get_sdf_features(next_state_img, clip=clip_sdf)
            matching = sdf_module.object_matching(feature_ns, feature_g)
            sdf_ns_align = sdf_ns[matching]
            sdf_raw = sdf_raw[matching]

            # detection failed #
            if len(sdf_ns_align) == 0:
                reward = -1.
                done = True

            if visualize_q:
                ax1.imshow(next_state_img)

                # goal sdfs
                vis_g = norm_npy(sdf_g + 2*(sdf_g>0).astype(float))
                goal_sdfs = np.zeros([96, 96, 3])
                for _s in range(len(vis_g)):
                    goal_sdfs += np.expand_dims(vis_g[_s], 2) * np.array(cm(_s/5)[:3])
                ax2.imshow(norm_npy(goal_sdfs))
                # current sdfs
                vis_c = norm_npy(sdf_ns_align + 2*(sdf_ns_align>0).astype(float))
                current_sdfs = np.zeros([96, 96, 3])
                for _s in range(len(vis_c)):
                    current_sdfs += np.expand_dims(vis_c[_s], 2) * np.array(cm(_s/5)[:3])
                ax3.imshow(norm_npy(current_sdfs))
                fig.canvas.draw()

            mismatch = len(sdf_ns_align)!=env.num_blocks
            num_mismatch += int(mismatch) 
            if done:
                break
            else:
                sdf_st_align = sdf_ns_align

        log_returns.append(episode_reward)
        log_eplen.append(ep_len)
        log_out.append(int(info['out_of_range']))
        log_success.append(int(np.all(info['block_success'])))
        #log_success.append(int(info['success']))
        for o in range(env.num_blocks):
            log_success_block[o].append(int(info['block_success'][o]))
        log_sdf_mismatch.append(num_mismatch)

        print()
        print("{} episodes.".format(ne))
        print("Ep reward: {}".format(log_returns[-1]))
        print("Ep length: {}".format(log_eplen[-1]))
        print("Success rate: {}% ({}/{})".format(100*np.mean(log_success), np.sum(log_success), len(log_success)))
        for o in range(env.num_blocks):
            print("Block {}: {}% ({}/{})".format(o+1, 100*np.mean(log_success_block[o]), np.sum(log_success_block[o]), len(log_success_block[o])))
        print("Out of range: {}".format(np.mean(log_out)))
        print("Num of mismatches: {}".format(np.mean(log_sdf_mismatch)))

    print()
    print("="*80)
    print("Evaluation Done.")
    print("Mean reward: {0:.2f}".format(np.mean(log_returns)))
    print("Mean episode length: {}".format(np.mean(log_eplen)))
    print("Success rate: {}".format(100*np.mean(log_success)))
    for o in range(env.num_blocks):
        print("Block {}: {}% ({}/{})".format(o+1, 100*np.mean(log_success_block[o]), np.sum(log_success_block[o]), len(log_success_block[o])))
    print("Out of range: {}".format(np.mean(log_out)))
    print("Num of mismatches: {}".format(np.mean(log_sdf_mismatch)))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--num_blocks", default=3, type=int)
    parser.add_argument("--dist", default=0.06, type=float)
    parser.add_argument("--sdf_action", action="store_true") # default: False
    parser.add_argument("--real_object", action="store_true") # default: False
    parser.add_argument("--max_steps", default=100, type=int)
    parser.add_argument("--camera_height", default=480, type=int)
    parser.add_argument("--camera_width", default=480, type=int)
    parser.add_argument("--ver", default=3, type=int)
    parser.add_argument("--normalize", action="store_true") # default: False
    parser.add_argument("--clip", action="store_true") # default: False
    parser.add_argument("--reward", default="new", type=str)
    parser.add_argument("--model_path", default="0105_1223", type=str)
    parser.add_argument("--num_trials", default=100, type=int)
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
    sdf_action = args.sdf_action
    real_object = args.real_object
    mov_dist = args.dist
    max_steps = args.max_steps
    camera_height = args.camera_height
    camera_width = args.camera_width
    reward_type = args.reward
    gpu = args.gpu

    # evaluate configuration
    num_trials = args.num_trials
    model_path = os.path.join("results/models/SDF_%s.pth" % args.model_path)
    visualize_q = args.show_q
    if visualize_q:
        render = True

    sdf_module = SDFModule()
    if real_object:
        from realobjects_env import UR5Env
    else:
        from ur5_env import UR5Env
    env = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
                 control_freq=5, data_format='NHWC', gpu=gpu)
    env = objectwise_env(env, num_blocks=num_blocks, mov_dist=mov_dist, max_steps=max_steps, \
                         conti=False, detection=True, reward_type=reward_type)

    ver = args.ver
    graph_normalize = args.normalize
    clip_sdf = args.clip
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

    evaluate(env=env, sdf_module=sdf_module, n_actions=8, model_path=model_path,\
            num_trials=num_trials, visualize_q=visualize_q, clip_sdf=clip_sdf, \
            sdf_action=sdf_action, graph_normalize=graph_normalize)
