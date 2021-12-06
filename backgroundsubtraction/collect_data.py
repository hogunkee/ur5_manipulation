import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../ur5_mujoco'))
from pushpixel_env import *
from backgroundsubtraction_module import BackgroundSubtraction

import argparse
import json
import time
from matplotlib import pyplot as plt
from PIL import Image

import multiprocessing as mp
from multiprocessing import Process


def get_action_near_blocks(env, pad=0.06):
    poses, _ = env.get_poses()
    obj = np.random.randint(len(poses))
    pose = poses[obj]
    x = np.random.uniform(pose[0]-pad, pose[0]+pad)
    y = np.random.uniform(pose[1]-pad, pose[1]+pad)
    py, px = env.pos2pixel(x, y)
    theta = np.arctan2(x-pose[0], y-pose[1]) / np.pi + 1
    ptheta = np.round((theta + np.random.normal(0, 0.5)) * 4) % 8
    # ptheta = np.random.randint(8)
    return [px, py, ptheta]

def collect_images(process_id, args, backsub):
    render = args.render
    num_blocks = args.num_blocks
    mov_dist = 0.08 #args.dist
    max_steps = 30 #args.max_steps
    xml_ver = args.xml
    camera_height = args.camera_height
    camera_width = args.camera_width

    reward_type = 'binary'
    goal_type = 'pixel'

    env = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
                 control_freq=5, data_format='NHWC', camera_depth=True, xml_ver=xml_ver)
    env = pushpixel_env(env, num_blocks=num_blocks, mov_dist=mov_dist, max_steps=max_steps, \
                        task=0, reward_type=reward_type, goal_type=goal_type)

    data_dir = 'data'
    num_scenes = len([scene for scene in os.listdir(data_dir) if 'scene_' in scene])
    while num_scenes < args.num_scenes:
        rgb_list = []
        depth_list = []
        label_list = []
        for view in range(args.num_views_per_scene):
            state = env.reset()
            rgb = (state[0] * 255).astype(np.uint8)
            depth = (state[1] * 1000).astype(np.uint32)
            masks, _, _ = backsub.get_masks(rgb, env.num_blocks)
            ws_mask = backsub.workspace_seg
            label = np.zeros_like(depth).astype(np.uint8)
            label += ws_mask.astype(np.uint8)
            for i, m in enumerate(masks):
                label += (i+2) * m.astype(np.uint8)
            rgb_list.append(rgb)
            depth_list.append(depth)
            label_list.append(label)
        num_scenes = len([scene for scene in os.listdir(data_dir) if 'scene_' in scene])
        scene_dir = os.path.join(data_dir, 'scene_%05d' %num_scenes)
        os.makedirs(scene_dir)
        for view, (rgb, depth, label) in enumerate(zip(rgb_list, depth_list, label_list)):
            Image.fromarray(rgb).save(os.path.join(scene_dir, 'rgb_%05d.jpeg' %view))
            Image.fromarray(depth).save(os.path.join(scene_dir, 'depth_%05d.png' %view))
            Image.fromarray(label).save(os.path.join(scene_dir, 'segmentation_%05d.png' %view))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--num_blocks", default=3, type=int)
    parser.add_argument("--camera_height", default=96, type=int)
    parser.add_argument("--camera_width", default=96, type=int)
    parser.add_argument("--xml", default=0, type=int)
    parser.add_argument("--num_process", default=4, type=int)
    parser.add_argument("--num_views_per_scene", default=20, type=int)
    parser.add_argument("--num_scenes", default=1000, type=int)
    args = parser.parse_args()

    if not os.path.isdir('data'):
        os.makedirs('data')

    num_process = args.num_process
    procs = []
    backsub = BackgroundSubtraction()
    for p in range(num_process):
        proc = Process(target=collect_images, args=(p, args, backsub,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
