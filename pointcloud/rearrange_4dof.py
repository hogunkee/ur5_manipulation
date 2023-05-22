import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../ur5_mujoco'))
sys.path.append(os.path.join(FILE_PATH, '../object_wise'))
sys.path.append(os.path.join(FILE_PATH, '../backgroundsubtraction'))


from picknplace_env import *
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

from replay_buffer import ReplayBuffer, PER
from matplotlib import pyplot as plt
from PIL import Image

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

sys.path.append('/home/gun/Desktop/contact_graspnet/contact_graspnet')
from visualization_utils import visualize_grasps, show_image

from backgroundsubtraction_module import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def projection(pose_3d, cam_K, cam_mat):
    cam_K = deepcopy(cam_K)
    #cam_K[0, 0] = -cam_K[0, 0]
    x_world = np.ones(4)
    x_world[:3] = pose_3d
    p = cam_K.dot(cam_mat[:3].dot(x_world))
    u = p[0] / p[2]
    v = p[1] / p[2]
    return 480-int(np.round(u)), int(np.round(v))

def norm_npy(array):
    positive = array - array.min()
    return positive / positive.max()

def get_action(env, max_blocks, depth, sdf_raw, sdfs, epsilon, with_q=False, sdf_action=False, target_res=96):

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
        theta = np.arctan2(gx-sx, gy-sy)
        theta = np.round(theta / np.pi / 0.25) % 8

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

    print(action)
    if with_q:
        return action, [cx, cy, theta], mask, None
    else:
        return action, [cx, cy, theta], mask


def evaluate(env,
        n_actions=8,
        num_trials=100,
        max_blocks=5,
        tracker=False, 
        segmentation=False,
        scenario=-1
        ):

    log_returns = []
    log_eplen = []
    log_out = []
    log_success = []
    log_distance = []
    log_success_block = [[] for i in range(env.num_blocks)]

    # Camera Intrinsic #
    fovy = env.env.sim.model.cam_fovy[env.cam_id]
    img_height = env.env.camera_height

    f = 0.5 * img_height / np.tan(fovy * np.pi / 360)
    cam_K = np.array([[f, 0, 240],
                      [0, f, 240],
                      [0, 0, 1]])
    # Background Subtraction #
    res = 96 #480
    backsub = BackgroundSubtraction(res=res)
    backsub.fitting_model_from_data('test_scenes/deg0/goal/')
    backsub.fitting_model_from_data('test_scenes/deg0/state/')

    epsilon = 0.1
    ni = 0
    for ne in range(num_trials):
        ep_len = 0
        episode_reward = 0.

        check_env_ready = False
        while not check_env_ready:
            (state_img, goal_img), info = env.reset(scenario=scenario)
            check_env_ready = True
        ni = 0
        Image.fromarray((state_img[0] * 255).astype(np.uint8)).save('data/state_%d.png' % int(ni))
        Image.fromarray((goal_img[0] * 255).astype(np.uint8)).save('data/goal_%d.png' % int(ni))
        rgb, depth = state_img
        m_s, cm_s ,fm_s = backsub.get_masks(rgb)
        masks = []
        for m in m_s:
            mask = cv2.resize(m, (480, 480), interpolation=cv2.INTER_AREA)
            mask = mask.astype(bool).astype(int)
            masks.append(mask)

        R, t = env.apply_cpd(state_img, goal_img, masks)
        grasps, scores = env.get_grasps(rgb, depth)
        object_grasps = env.extract_grasps(grasps, scores, masks)

        for o in object_grasps:
            if len(object_grasps[o])==0:
                print("No grasp candidates on object '%d'."%o)
                continue
            env.picknplace(object_grasps[o][0], R[o], t[o])
            #env.pick(object_grasps[o][0][0])
            #env.place(object_grasps[o][0][0], R[o], t[o])

        print('stop.')

        # TODO 1. #
        # collect images from cam_theta=0

        # TODO 2. #
        # apply CPD -> R, t
        # transform R, t to robot frame

        rgb_s = state_img[0]
        if use_hsv:
            hsv_s = cv2.cvtColor(cv2.cvtColor(rgb_s, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
        m_s = backsub.get_mask(rgb_s)
        m_s_resized = cv2.resize(m_s, (res, res), interpolation=cv2.INTER_NEAREST)
        rgb_s_resized = cv2.resize(rgb_s, (res, res), interpolation=cv2.INTER_AREA)
        if use_hsv:
            hsv_s_resized = cv2.cvtColor(cv2.cvtColor(rgb_s_resized, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)

        y, x = np.where(m_s_resized)
        points_s = np.concatenate([np.array([x/res-0.5]).T, np.array([y/res-0.5]).T, rgb_s_resized[y, x]/255., hsv_s_resized[y, x]/255.], 1)


        rgb_g = goal_img[0]
        m_g, cm_g ,fm_g = backsub.get_masks(rgb_g)

        num_obj = len(m_g)
        points_g = []
        z_g = []

        rgb_g_resized = cv2.resize(rgb_g, (res, res), interpolation=cv2.INTER_AREA)
        hsv_g_resized = cv2.cvtColor(cv2.cvtColor(rgb_g_resized, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
        for i, m in enumerate(m_g):
            m_resized = cv2.resize(m, (res, res), interpolation=cv2.INTER_NEAREST)
            y, x = np.where(m_resized)

            p = np.concatenate([np.array([x/res-0.5]).T, np.array([y/res-0.5]).T, rgb_g_resized[y, x]/255., hsv_g_resized[y, x]/255.], 1)
            points_g.append(p)

            z = np.zeros([len(p), num_obj])
            z[:, i] = 1.
            z_g.append(z)

        points_g = np.concatenate(points_g, 0)
        z_g = np.concatenate(z_g, 0)

        fig_vis = plt.figure(figsize=(10, 5))
        fig_vis.add_axes([0, 0, 0.5, 1])
        fig_vis.add_axes([0.5, 0, 0.5, 1])
        callback = partial(visualize3, ax1=fig_vis.axes[0], ax2=fig_vis.axes[1])
        remove_colors = False
        if remove_colors:
            points_s[:, 2:] = [0, 0, 0]
            points_g[:, 2:] = [0, 0, 0]

        print("===================== Result ========================")
        reg = RigidPartRegistration(**{'X': points_s, 'Y': points_g, 'Z': z_g, 'RGB': RGB, 'HSV': HSV, 'K': num_obj})
        TY, (R, t) = reg.register(callback)
        #print(TY[0])
        #print(R[0], t[0])
        print("Q value:", reg.q)
        print("=====================================================")

        TY_combined = TY[np.where(z_g)[1], np.where(z_g)[0]]
        fig = plt.figure(figsize=(10, 5))
        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(122)
        ax0.scatter(points_s[:, 0], -points_s[:, 1], c=points_s[:, 2:5], marker=".")
        ax1.scatter(TY_combined[:, 0], -TY_combined[:, 1], c=points_g[:, 2:5], marker=".")

        xmin = min(TY_combined[:, 0].min(), points_s[:, 0].min()) - 0.01
        xmax = max(TY_combined[:, 0].max(), points_s[:, 0].max()) + 0.01
        ymin = -max(TY_combined[:, 1].max(), points_s[:, 1].max()) - 0.01
        ymax = -min(TY_combined[:, 1].min(), points_s[:, 1].min()) + 0.01
        ax0.set_xlim(xmin, xmax)
        ax1.set_xlim(xmin, xmax)
        ax0.set_ylim(ymin, ymax)
        ax1.set_ylim(ymin, ymax)


        # TODO 3. #
        # find grasp candidates for each object
        # by grasp points -> 2D pixel -> compare to masks

        # TODO 4. #
        # implement Place action

        # TODO 5. #
        # metrics: delta_R, delta_t

        for t_step in range(env.max_steps):
            ep_len += 1
            action, pose_action, sdf_mask, q_map = get_action(env, max_blocks, \
                    state_img[1], sdf_raw, [sdf_st_align, sdf_g], epsilon=epsilon,  \
                    with_q=True, sdf_action=sdf_action, target_res=sdf_res)

            (next_state_img, _), reward, done, info = env.step(pose_action, sdf_mask)

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
            sdf_raw = sdf_module.align_sdf(matching, sdf_raw, np.zeros([env.num_blocks, *sdf_raw.shape[1:]]))

            # sdf reward #
            reward += sdf_module.add_sdf_reward(sdf_st_align, sdf_ns_align, sdf_g)
            episode_reward += reward

            # detection failed #
            if n_detection == 0:
                done = True

            if info['block_success'].all():
                info['success'] = True
            else:
                info['success'] = False

            if done:
                break
            else:
                sdf_st_align = sdf_ns_align

        log_returns.append(episode_reward)
        log_eplen.append(ep_len)
        log_out.append(int(info['out_of_range']))
        log_success.append(int(info['success']))
        log_distance.append(info['dist'])
        for o in range(env.num_blocks):
            log_success_block[o].append(int(info['block_success'][o]))# and sdf_success[o]))

        print("EP{}".format(ne+1), end=" / ")
        print("reward:{0:.2f}".format(log_returns[-1]), end=" / ")
        print("eplen:{0:.1f}".format(log_eplen[-1]), end=" / ")
        print("SR:{0:.2f} ({1}/{2})".format(np.mean(log_success),
                np.sum(log_success), len(log_success)), end=" / ")
        for o in range(env.num_blocks):
            print("B{0}:{1:.2f}".format(o+1, np.mean(log_success_block[o])), end=" ")

        #print(" / mean eplen:{0:.1f}".format(np.mean(log_eplen)), end="")
        #print(" / mean error:{0:.1f}".format(np.mean(log_distance)*1e3), end="")
        dist_success = np.array(log_distance)[np.array(log_success)==1] * 1e3 #scale: mm
        print(" / mean error:{0:.1f}".format(np.mean(dist_success)), end="")
        eplen_success = np.array(log_eplen)[np.array(log_success)==1]
        print(" / mean eplen:{0:.1f}".format(np.mean(eplen_success)), end="")
        print(" / oor:{0:.2f}".format(np.mean(log_out)), end="")
        print(" / mean reward:{0:.1f}".format(np.mean(log_returns)))

    print()
    print("="*80)
    print("Evaluation Done.")
    print("Success rate: {}".format(100*np.mean(log_success)))
    #print("Mean episode length: {}".format(np.mean(log_eplen)))
    #print("Mean error: {0:.1f}".format(np.mean(log_distance) * 1e3))
    dist_success = np.array(log_distance)[np.array(log_success)==1] * 1e3 #scale: mm
    print("Error-success: {0:.1f}".format(np.mean(dist_success)))
    eplen_success = np.array(log_eplen)[np.array(log_success)==1]
    print("Mean episode length: {}".format(np.mean(eplen_success)))
    print("Out of range: {}".format(np.mean(log_out)))
    print("Mean reward: {0:.2f}".format(np.mean(log_returns)))
    for o in range(env.num_blocks):
        print("Block {}: {}% ({}/{})".format(o+1, 100*np.mean(log_success_block[o]), np.sum(log_success_block[o]), len(log_success_block[o])))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # env config #
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--num_blocks", default=3, type=int)
    parser.add_argument("--max_blocks", default=8, type=int)
    parser.add_argument("--threshold", default=0.10, type=float)
    parser.add_argument("--real_object", action="store_false")
    parser.add_argument("--dataset", default="test", type=str)
    parser.add_argument("--max_steps", default=100, type=int)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--scenario", default=-1, type=int)
    # etc #
    parser.add_argument("--num_trials", default=100, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--gpu", default=-1, type=int)

    # contact-graspnet #
    parser.add_argument('--ckpt_dir', default='/home/gun/Desktop/contact_graspnet/checkpoints/scene_test_2048_bs3_hor_sigma_001', \
                        help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]')
    parser.add_argument('--K', default=None, help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
    parser.add_argument('--z_range', default=[0.2,1.0], help='Z value threshold to crop the input point cloud')
    parser.add_argument('--local_regions', action='store_true', default=False, help='Crop 3D local regions around given segments.')
    parser.add_argument('--filter_grasps', action='store_true', default=False,  help='Filter grasp contacts according to segmap.')
    parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
    parser.add_argument('--forward_passes', type=int, default=1,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    parser.add_argument('--segmap_id', type=int, default=0,  help='Only return grasps of the given object id')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
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
    real_object = args.real_object
    dataset = args.dataset
    threshold = args.threshold
    max_steps = args.max_steps
    small = args.small
    scenario = args.scenario
    gpu = args.gpu

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if str(gpu) in visible_gpus:
            gpu_idx = visible_gpus.index(str(gpu))
            torch.cuda.set_device(gpu_idx)

    # evaluate configuration
    num_trials = args.num_trials

    # model configuration
    resize = True
    depth = False
    mov_dist = 0.06
    camera_height = 480
    camera_width = 480
    tracker = True
    convex_hull = False
    reward_type = 'linear_penalty'

    if real_object:
        from realobjects_env import UR5Env
    else:
        from ur5_env import UR5Env
    env = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
            control_freq=5, data_format='NHWC', gpu=gpu, camera_depth=True, dataset=dataset,\
            small=small, camera_name='rlview2')
    env = picknplace_env(env, num_blocks=num_blocks, mov_dist=mov_dist, max_steps=max_steps, \
            threshold=threshold, reward_type=reward_type)
    env.load_contactgraspnet(args.ckpt_dir, args.arg_configs)

    evaluate(env=env, n_actions=8, num_trials=num_trials, \
            max_blocks=max_blocks, \
            tracker=tracker, \
            segmentation=False, scenario=scenario)
