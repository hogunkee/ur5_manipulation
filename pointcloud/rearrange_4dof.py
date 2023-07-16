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
        scenario=-1,
        use_hsv=False,
        ):
    log_dist = []
    log_angle = []
    log_dist_place = []
    log_angle_place = []

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
    for ne in range(num_trials):
        ep_len = 0
        episode_reward = 0.

        check_env_ready = False
        while not check_env_ready:
            (state_img, goal_img), info = env.reset(scenario=scenario)
            check_env_ready = True

        ni = len([f for f in os.listdir('data/') if 'state_' in f])
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

        goal_poses = info['goal_poses']
        goal_rotations = info['goal_rotations']
        pre_poses, pre_rotations = env.get_poses()
        flag_feasible_grasp = []
        flag_pick_fail = []
        for o in object_grasps:
            if len(object_grasps[o])==0:
                print("No grasp candidates on object '%d'."%o)
                continue
            placement, failed_to_pick = env.picknplace(object_grasps[o], R[o], t[o])
            if placement is None:
                print('No feasible grasps..')
                flag_feasible_grasp.append(False)
                flag_pick_fail.append(True)
            else:
                flag_feasible_grasp.append(True)
                flag_pick_fail.append(failed_to_pick)
            #env.pick(object_grasps[o][0][0])
            #env.place(object_grasps[o][0][0], R[o], t[o])

        poses, rotations = env.get_poses()
        distances = []
        angles = []
        distances_place = []
        angles_place = []
        for o in range(len(rotations)):
            R1 = goal_rotations[o]
            #R1 = pre_rotations[o]
            R2 = rotations[o]
            dist = np.linalg.norm(goal_poses[o] - poses[o])
            angle = env.get_angle(R1, R2)
            #dist = np.linalg.norm(pre_poses[o] - poses[o])
            #if flag_feasible_grasp[o]:
            distances.append(dist)
            angles.append(angle)
            # if not flag_pick_fail[o]:
            #     distances_place.append(dist)
            #     angles_place.append(angle)
            print(o)
            print(np.linalg.norm(pre_poses[o] - poses[o]))
            print(dist)
            print(angle)
        result_img = env.env.move_to_pos(env.init_pos, grasp=1.0, get_img=True)
        Image.fromarray((result_img[0] * 255).astype(np.uint8)).save('data/result_%d.png' % int(ni))

        log_dist.append(np.mean(distances))
        log_angle.append(np.mean(angles))
        # log_dist_place.append(np.mean(distances_place))
        # log_angle_place.append(np.mean(angles_place))
        print('distance:', np.mean(log_dist))
        print('angle:', np.mean(log_angle))
        # print('distance-place:', np.mean(log_dist_place))
        # print('angle-place:', np.mean(log_angle_place))
        print('-'*80)

    print()
    print('mean distance:', np.mean(log_dist))
    print('mean angle:', np.mean(log_angle))
    # print('mean distance-place:', np.mean(log_dist_place))
    # print('mean angle-place:', np.mean(log_angle_place))
    print('End.')


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

    evaluate(env=env, n_actions=8, num_trials=num_trials, max_blocks=max_blocks, \
            tracker=tracker, segmentation=False, scenario=scenario, use_hsv=False)
