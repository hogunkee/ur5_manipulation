import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../ur5_mujoco'))
sys.path.append(os.path.join(FILE_PATH, '..'))
from object_env import *

import torch
import torch.nn as nn
import argparse
import json

import copy
import time
import datetime
import random
import pylab

from matplotlib import pyplot as plt
from PIL import Image

import warnings
warnings.filterwarnings("ignore", category=UserWarning)



def collect(env):
    if not os.path.exists('test_scenes/deg0/state/'):
        os.makedirs('test_scenes/deg0/state/')
    if not os.path.exists('test_scenes/deg0/goal/'):
        os.makedirs('test_scenes/deg0/goal/')

    for ni in range(num_collect):
        (state_img, goal_img), info = env.reset(scenario=scenario)
        # collecting RGB images #
        print(ni)
        Image.fromarray((state_img[0]*255).astype(np.uint8)).save('test_scenes/deg0/state/%d.png'%int(ni))
        Image.fromarray((goal_img[0]*255).astype(np.uint8)).save('test_scenes/deg0/goal/%d.png'%int(ni))
        np.save('test_scenes/deg0/state/%d.npy'%int(ni), state_img[1])
        np.save('test_scenes/deg0/goal/%d.npy'%int(ni), goal_img[1])


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # env config #
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--num_blocks", default=3, type=int)
    parser.add_argument("--real_object", action="store_false")
    parser.add_argument("--dataset", default="test", type=str)
    parser.add_argument("--small", action="store_true")
    # etc #
    parser.add_argument("--num_collect", default=100, type=int)
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
    real_object = args.real_object
    dataset = args.dataset
    gpu = args.gpu
    small = args.small
    scenario = args.scenario

    # evaluate configuration
    num_collect = args.num_collect

    if real_object:
        from realobjects_env import UR5Env
    else:
        from ur5_env import UR5Env
    env = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
            control_freq=5, data_format='NHWC', gpu=gpu, camera_depth=True, dataset=dataset,\
            small=small, camera_name='rlview2')
    env = objectwise_env(env, num_blocks=num_blocks, mov_dist=0.04, max_steps=100, \
            threshold=0.07, conti=False, detection=True, reward_type='linear_penalty', \
            delta_action=False)

    collect(env=env, num_collect=num_collect)
