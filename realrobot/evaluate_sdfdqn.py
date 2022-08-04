import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../ur5_mujoco'))
sys.path.append(os.path.join(FILE_PATH, '../object_wise/dqn'))
from realur5_env import *

import torch
import torch.nn as nn
import argparse
import json

import copy
import time
import datetime
import random
import pylab

from real_sdf_module import SDFModule
from matplotlib import pyplot as plt
from PIL import Image

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def norm_npy(array):
    positive = array - array.min()
    return positive / positive.max()

def pad_sdf(sdf, nmax, res=96):
    nsdf = len(sdf)
    padded = np.zeros([nmax, res, res])
    if nsdf > nmax:
        padded[:] = sdf[:nmax]
    elif nsdf > 0:
        padded[:nsdf] = sdf
    return padded

def get_rulebased_action(env, max_blocks, qnet, depth, sdf_raw, sdfs, epsilon, with_q=False, target_res=96):
    if np.random.random() < epsilon:
        #print('Random action')
        obj = np.random.randint(len(sdf_raw))
        theta = np.random.randint(env.num_bins)
    else:
        nsdf = sdfs[0].shape[0]
        s = pad_sdf(sdfs[0], max_blocks, target_res)
        g = pad_sdf(sdfs[1], max_blocks, target_res)
        nonempty = np.where(np.sum(s, (1,2))!=0)[0]

        check_reach = True
        for _ in range(50):
            obj = np.random.choice(nonempty)
            sx, sy = env.get_center_from_sdf(s[obj], depth)
            gx, gy = env.get_center_from_sdf(g[obj], depth)
            check_reach = (np.linalg.norm([sx-gx, sy-gy])<0.05)
            if not check_reach:
                break
        theta = np.arctan2(gy-sy, gx-sx)
        #theta = np.arctan2(gx-sx, gy-sy)
        theta = (np.round(theta / np.pi / 0.25)) % 8

    action = [obj, theta]
    return action, None

def get_action(env, max_blocks, qnet, depth, sdf_raw, sdfs, epsilon, with_q=False, target_res=96):
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
            with torch.no_grad():
                q_value = qnet([s, g], nsdf)
            q = q_value[0][:nsdf].detach().cpu().numpy()
    else:
        nsdf = sdfs[0].shape[0]
        s = pad_sdf(sdfs[0], max_blocks, target_res)
        empty_mask = (np.sum(s, (1,2))==0)[:nsdf]
        s = torch.FloatTensor(s).to(device).unsqueeze(0)
        g = pad_sdf(sdfs[1], max_blocks, target_res)
        g = torch.FloatTensor(g).to(device).unsqueeze(0)
        nsdf = torch.LongTensor([nsdf]).to(device)
        with torch.no_grad():
            q_value = qnet([s, g], nsdf)
        q = q_value[0][:nsdf].detach().cpu().numpy()
        q[empty_mask] = q.min() - 0.1

        obj = q.max(1).argmax()
        theta = q.max(0).argmax()

    action = [obj, theta]
    return action, q


def evaluate(env,
        sdf_module,
        n_actions=8,
        n_hidden=16,
        model_path='',
        num_trials=100,
        visualize_q=False,
        clip_sdf=False,
        graph_normalize=False,
        max_blocks=5,
        round_sdf=False,
        separate=False,
        bias=True,
        adj_ver=1,
        selfloop=False,
        tracker=False, 
        segmentation=False,
        rule_based=False,
        ):
    if rule_based:
        qnet = None
    else:
        qnet = QNet(max_blocks, adj_ver, n_actions, n_hidden=n_hidden, selfloop=selfloop, \
                normalize=graph_normalize, separate=separate, bias=bias).to(device)
        if qnet.resize:
            qnet.ws_mask = np.load('workspace_mask.npy').astype(float)
            #qnet.ws_mask = np.load('real_wsmask.npy').astype(float)
        else:
            qnet.ws_mask = np.load('workspace_mask_480.npy').astype(float)
            #qnet.ws_mask = np.load('real_wsmask_480.npy').astype(float)
        qnet.load_state_dict(torch.load(model_path))
        qnet.eval()
    print('='*30)
    print('Loading trained model: {}'.format(model_path))
    print('='*30)

    if sdf_module.resize:
        sdf_res = 96
    else:
        sdf_res = 480

    log_returns = []
    log_eplen = []
    log_out = []
    log_success = []
    log_distance = []
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
        '''
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax3.set_xticks([])
        ax3.set_yticks([])
        '''
        plt.show(block=False)
        fig.canvas.draw()

        cm = pylab.get_cmap('gist_rainbow')

    epsilon = 0.1
    background_img, _ = env.reset()
    x = input("Set Background? press 'y' if you want.")
    if x=="y" or x=="Y":
        sdf_module.set_background(background_img[1])
        sdf_module.save_background(background_img[1])
    else:
        sdf_module.load_background()

    for ne in range(num_trials):
        ep_len = 0
        episode_reward = 0.

        _ = input("Setup the Goal Scene.")
        check_goal_ready = False
        while not check_goal_ready:
            env.set_goals()
            if segmentation:
                sdf_g_b, _, feature_g = sdf_module.get_seg_features_with_ucn(env.goals[0], env.goals[1], env.num_blocks, clip=clip_sdf)
            else:
                sdf_g_b, _, feature_g = sdf_module.get_sdf_features_with_ucn(env.goals[0], env.goals[1], env.num_blocks, clip=clip_sdf)
            sdf_g = sdf_module.make_round_sdf(sdf_g_b) if round_sdf else sdf_g_b

            if len(sdf_g_b)==0:
                ax0.imshow(env.goals[0])
                fig.canvas.draw()
                x = input('Reset the goals.')
                continue

            if visualize_q:
                ax0.imshow(env.goals[0])
                # visualize goal sdfs
                vis_g = norm_npy(0*sdf_g_b + 2*(sdf_g_b>0).astype(float))
                goal_sdfs = np.zeros([sdf_res, sdf_res, 3])
                for _s in range(len(vis_g)):
                    goal_sdfs += np.expand_dims(vis_g[_s], 2) * np.array(cm(_s/5)[:3])
                ax2.imshow(norm_npy(goal_sdfs))
                fig.canvas.draw()

            x = input('Continue?')
            if x=='r':
                check_goal_ready = False
                print('Reset the goals.')
            else:
                check_goal_ready = True

        _ = input("Setup the Initial Scene.")
        check_env_ready = False
        while not check_env_ready:
            (state_img, goal_img) = env.reset()
            if segmentation:
                sdf_st, sdf_raw, feature_st = sdf_module.get_seg_features_with_ucn(state_img[0], state_img[1], env.num_blocks, clip=clip_sdf)
            else:
                sdf_st, sdf_raw, feature_st = sdf_module.get_sdf_features_with_ucn(state_img[0], state_img[1], env.num_blocks, clip=clip_sdf)

            if visualize_q:
                ax1.imshow(state_img[0])
                # visualize current sdfs
                vis_c = norm_npy(0*sdf_st + 2*(sdf_st>0).astype(float))
                current_sdfs = np.zeros([sdf_res, sdf_res, 3])
                for _s in range(len(vis_c)):
                    current_sdfs += np.expand_dims(vis_c[_s], 2) * np.array(cm(_s/5)[:3])
                ax3.imshow(norm_npy(current_sdfs))
                fig.canvas.draw()

            x = input('Continue?')
            if x=='r':
                check_env_ready = False
                print("Reset the scene.")
            else:
                check_env_ready = True
            #check_env_ready = (len(sdf_g)==env.num_blocks) & (len(sdf_st)==env.num_blocks)

        n_detection = len(sdf_st)
        # target: st / source: g
        matching = sdf_module.object_matching(feature_st, feature_g)
        sdf_st_align = sdf_module.align_sdf(matching, sdf_st, sdf_g)
        sdf_raw = sdf_module.align_sdf(matching, sdf_raw, np.zeros([sdf_g.shape[0], *sdf_raw.shape[1:]]))

        masks = []
        for s in sdf_raw:
            masks.append(s>0)
        if tracker:
            sdf_module.init_tracker(state_img[0], masks)

        if visualize_q:
            ax0.imshow(goal_img[0])
            ax1.imshow(state_img[0])
            # goal sdfs
            vis_g = norm_npy(0*sdf_g_b + 2*(sdf_g_b>0).astype(float))
            goal_sdfs = np.zeros([sdf_res, sdf_res, 3])
            for _s in range(len(vis_g)):
                goal_sdfs += np.expand_dims(vis_g[_s], 2) * np.array(cm(_s/5)[:3])
            ax2.imshow(norm_npy(goal_sdfs))
            # current sdfs
            vis_c = norm_npy(0*sdf_st_align + 2*(sdf_st_align>0).astype(float))
            current_sdfs = np.zeros([sdf_res, sdf_res, 3])
            for _s in range(len(vis_c)):
                current_sdfs += np.expand_dims(vis_c[_s], 2) * np.array(cm(_s/5)[:3])
            ax3.imshow(norm_npy(current_sdfs))
            fig.canvas.draw()

        for t_step in range(env.max_steps):
            print()
            print("{} step.".format(t_step+1))
            ep_len += 1
            if rule_based:
                action, q_map = get_rulebased_action(env, max_blocks, qnet, state_img[1], sdf_raw, \
                        [sdf_st_align, sdf_g], epsilon=epsilon, with_q=True, \
                        target_res=sdf_res)
            else:
                action, q_map = get_action(env, max_blocks, qnet, state_img[1], sdf_raw, \
                        [sdf_st_align, sdf_g], epsilon=epsilon, with_q=True, \
                        target_res=sdf_res)

            print('action:', action)
            (next_state_img, _), reward, done, info = env.step(action, sdf_st_align, sdf_g, state_img[1])

            if tracker:
                if segmentation:
                    sdf_ns, sdf_raw, feature_ns = sdf_module.get_seg_features(next_state_img[0], next_state_img[1], env.num_blocks, clip=clip_sdf)
                else:
                    sdf_ns, sdf_raw, feature_ns = sdf_module.get_sdf_features(next_state_img[0], next_state_img[1], env.num_blocks, clip=clip_sdf)
            else:
                if segmentation:
                    sdf_ns, sdf_raw, feature_ns = sdf_module.get_seg_features_with_ucn(next_state_img[0], next_state_img[1], env.num_blocks, clip=clip_sdf)
                else:
                    sdf_ns, sdf_raw, feature_ns = sdf_module.get_sdf_features_with_ucn(next_state_img[0], next_state_img[1], env.num_blocks, clip=clip_sdf)
            pre_n_detection = n_detection
            n_detection = len(sdf_ns)
            matching = sdf_module.object_matching(feature_ns, feature_g)
            sdf_ns_align = sdf_module.align_sdf(matching, sdf_ns, sdf_g)
            sdf_raw = sdf_module.align_sdf(matching, sdf_raw, np.zeros([sdf_g.shape[0], *sdf_raw.shape[1:]]))

            sdf_dist = sdf_module.get_sdf_center_dist(sdf_ns_align, sdf_g, env.num_blocks)
            print("SDF Dist:", sdf_dist)

            # detection failed #
            if n_detection == 0:
                done = True

            if done:
                break
            else:
                sdf_st_align = sdf_ns_align
                state_img = next_state_img

            if visualize_q:
                ax1.imshow(state_img[0])

                # goal sdfs
                vis_g = norm_npy(0*sdf_g_b + 2*(sdf_g_b>0).astype(float))
                goal_sdfs = np.zeros([sdf_res, sdf_res, 3])
                for _s in range(len(vis_g)):
                    goal_sdfs += np.expand_dims(vis_g[_s], 2) * np.array(cm(_s/5)[:3])
                ax2.imshow(norm_npy(goal_sdfs))
                # current sdfs
                vis_c = norm_npy(0*sdf_st_align + 2*(sdf_st_align>0).astype(float))
                current_sdfs = np.zeros([sdf_res, sdf_res, 3])
                for _s in range(len(vis_c)):
                    current_sdfs += np.expand_dims(vis_c[_s], 2) * np.array(cm(_s/5)[:3])
                ax3.imshow(norm_npy(current_sdfs))
                fig.canvas.draw()

                # save images
                #fnum = len([f for f in os.listdir('test_scenes/sdfs/') if 'o' in f])
                #im = Image.fromarray((next_state_img[0] * 255).astype(np.uint8))
                #im.save('test_scenes/sdfs/o%d.png' %fnum)
                #fnum = len([f for f in os.listdir('test_scenes/sdfs/') if 's' in f])
                #im = Image.fromarray((norm_npy(current_sdfs) * 255).astype(np.uint8))
                #im.save('test_scenes/sdfs/s%d.png' %fnum)


        print("Ep {} done.".format(ne+1))
        print("Ep length: {}".format(ep_len))
        log_eplen.append(ep_len)

    print()
    print("="*80)
    print("Evaluation Done.")
    print("Mean episode length: {}".format(np.mean(log_eplen)))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # env config #
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--num_blocks", default=3, type=int)
    parser.add_argument("--max_blocks", default=8, type=int)
    parser.add_argument("--threshold", default=0.10, type=float)
    parser.add_argument("--max_steps", default=100, type=int)
    # model #
    parser.add_argument("--model", default="nobn", type=str) # rulebased / nsdf / nobn / in
    parser.add_argument("--model_path", default="DQN_0614_1203", type=str)
    # etc #
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
    max_blocks = args.max_blocks
    threshold = args.threshold
    max_steps = args.max_steps
    gpu = args.gpu

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if str(gpu) in visible_gpus:
            gpu_idx = visible_gpus.index(str(gpu))
            torch.cuda.set_device(gpu_idx)

    # evaluate configuration
    num_trials = args.num_trials
    model = args.model
    if model=="rulebased":
        model_path = ""
        adj_ver = 0
        selfloop = False
        graph_normalize = False
        resize = True
        separate = False
        bias = False
        clip_sdf = False
        round_sdf = False
        depth = False
        tracker = False
        convex_hull = False
        segmentation = False
    else:
        model_path = os.path.join("trained_models/%s.pth" % args.model_path)
        config_path = os.path.join("trained_models/%s.json" % args.model_path)

        # model configuration
        with open(config_path, 'r') as cf:
            config = json.load(cf)
        ver = config['ver']
        adj_ver = config['adj_ver']
        selfloop = config['selfloop']
        graph_normalize = config['normalize']
        resize = config['resize']
        separate = config['separate']
        if 'bias' in config:
            bias = config['bias']
        else:
            bias = True
        clip_sdf = config['clip']
        round_sdf = config['round_sdf']
        depth = config['depth']
        tracker = False #config['tracker']
        convex_hull = config['convex_hull']
        if 'segmentation' in config:
            segmentation = config['segmentation']
        else:
            segmentation = False

    visualize_q = args.show_q
    sdf_module = SDFModule(rgb_feature=True, resnet_feature=True, convex_hull=convex_hull, 
            binary_hole=True, using_depth=depth, tracker=None, resize=resize)
    ur5robot = UR5Robot()
    env = RealSDFEnv(ur5robot, sdf_module, num_blocks=num_blocks)

    rule_based = False
    if model=="rulebased":
        rule_based = True
        n_hidden = 0
    elif model=="nsdf":
        if ver==0:
            from models.track_gcn_nsdf import TrackQNetV0 as QNet
            n_hidden = 8 #16
        elif ver==1:
            from models.track_gcn_nsdf import TrackQNetV1 as QNet
            n_hidden = 8
        elif ver==2:
            from models.track_gcn_nsdf import TrackQNetV2 as QNet
            n_hidden = 8
        elif ver==3:
            from models.track_gcn_nsdf import TrackQNetV3 as QNet
            n_hidden = 64
    elif model=="nobn":
        if ver==0:
            from models.track_gcn_nobn import TrackQNetV0 as QNet
            n_hidden = 8 #16
        elif ver==1:
            from models.track_gcn_nobn import TrackQNetV1 as QNet
            n_hidden = 8
        elif ver==2:
            from models.track_gcn_nobn import TrackQNetV2 as QNet
            n_hidden = 8
        elif ver==3:
            from models.track_gcn_nobn import TrackQNetV3 as QNet
            n_hidden = 64
    elif model=="in":
        if ver==0:
            from models.track_gcn_in import TrackQNetV0 as QNet
            n_hidden = 8 #16
        elif ver==1:
            from models.track_gcn_in import TrackQNetV1 as QNet
            n_hidden = 8
        elif ver==2:
            from models.track_gcn_in import TrackQNetV2 as QNet
            n_hidden = 8
        elif ver==3:
            from models.track_gcn_in import TrackQNetV3 as QNet
            n_hidden = 64

    evaluate(env=env, sdf_module=sdf_module, n_actions=8, n_hidden=n_hidden, \
            model_path=model_path, num_trials=num_trials, visualize_q=visualize_q, \
            clip_sdf=clip_sdf, graph_normalize=graph_normalize, \
            max_blocks=max_blocks, round_sdf=round_sdf, \
            separate=separate, bias=bias, adj_ver=adj_ver, selfloop=selfloop, \
            tracker=tracker, segmentation=segmentation, rule_based=rule_based)
