import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../ur5_mujoco'))
from realobjects_env import UR5Env
from object_env import *

import argparse
import json
import time
from matplotlib import pyplot as plt

import multiprocessing as mp
from multiprocessing import Process


def collect_npy(process_id, args):
    render = args.render
    small = args.small
    num_blocks = args.num_blocks
    num_scenes = args.num_scenes
    num_sortings = args.num_sortings
    save_freq = args.save_freq
    camera_height = args.camera_height
    camera_width = args.camera_width
    gpu = args.gpu
    align_object = args.align_object

    np.random.seed(process_id)
    if process_id%2 == 0:
        dataset = 'train1'
    else:
        dataset = 'train2'
    env = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
                 control_freq=5, data_format='NHWC', gpu=gpu, camera_depth=True, small=small,
                 camera_name='rlview2', dataset=dataset)
    env = objectwise_env(env, num_blocks=num_blocks)

    buff_rgb = []
    buff_depth = []
    buff_poses = []
    buff_rotations = []
    for n in range(num_scenes):
        #print(n)
        env.env.select_objects(num_blocks, -1)
        images, poses, rotations = [], [], []
        for ns in range(num_sortings):
            if align_object:
                if ns%num_sortings==0:
                    img, p, r = env.init_scene()
                    quats = env.get_quats()
                else:
                    img, p, r = env.init_scene(quats)
            else:
                img, p, r = env.init_scene()
            images.append(img)
            poses.append(p)
            rotations.append(r)
        for i in range(len(images)):
            buff_rgb.append(images[i][0])
            buff_depth.append(images[i][1])
            buff_poses.append(poses[i])
            buff_rotations.append(rotations[i])

        if len(buff_rgb)%save_freq==0:
            num_files = len([f for f in os.listdir(args.data_dir) if 'image_' in f])
            i_filename = os.path.join(args.data_dir, 'image_%d.npy' % num_files)
            d_filename = os.path.join(args.data_dir, 'depth_%d.npy' % num_files)
            p_filename = os.path.join(args.data_dir, 'pose_%d.npy' % num_files)
            r_filename = os.path.join(args.data_dir, 'rotation_%d.npy' % num_files)

            np.save(i_filename, np.array(buff_rgb))
            np.save(d_filename, np.array(buff_depth))
            np.save(p_filename, np.array(buff_poses))
            np.save(r_filename, np.array(buff_rotations))
            print('Saved %s-th batch.'%num_files)

            buff_rgb.clear()
            buff_depth.clear()
            buff_poses.clear()
            buff_rotations.clear()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--num_blocks", default=3, type=int)
    parser.add_argument("--num_scenes", default=20000, type=int)
    parser.add_argument("--num_sortings", default=4, type=int)
    parser.add_argument("--save_freq", default=2048, type=int)
    parser.add_argument("--camera_height", default=96, type=int)
    parser.add_argument("--camera_width", default=96, type=int)
    parser.add_argument("--num_process", default=4, type=int)
    parser.add_argument("--data_dir", default='', type=str)
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--align_object", action="store_true")
    args = parser.parse_args()

    if args.align_object:
        args.data_dir = os.path.join(args.data_dir, '%dblock_align' %args.num_blocks)
    else:
        args.data_dir = os.path.join(args.data_dir, '%dblock' %args.num_blocks)
    if not os.path.isdir(args.data_dir):
        os.makedirs(args.data_dir)

    num_process = args.num_process
    procs = []
    for p in range(num_process):
        proc = Process(target=collect_npy, args=(p, args,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
