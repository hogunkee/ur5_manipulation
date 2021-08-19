import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../ur5_mujoco'))
from pushpixel_env import *

import argparse
import json
import time
from matplotlib import pyplot as plt

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

def collect_npy(process_id, args):
    render = args.render
    num_blocks = args.num_blocks
    mov_dist = args.dist
    max_steps = args.max_steps
    xml_ver = args.xml
    camera_height = args.camera_height
    camera_width = args.camera_width

    reward_type = 'binary'
    goal_type = 'pixel'

    env = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
                 control_freq=5, data_format='NCHW', xml_ver=xml_ver)
    env = pushpixel_env(env, num_blocks=num_blocks, mov_dist=mov_dist, max_steps=max_steps, \
                        task=1, reward_type=reward_type, goal_type=goal_type)

    DATA_DIR = 'data'
    save_freq = args.save_freq
    num_ep = args.num_ep

    frames = []
    actions = []
    for ep in range(num_ep):
        state = env.reset()
        ep_frames = [state[0]]
        ep_actions = []

        stucked = False
        count = 0
        while len(ep_frames) < max_steps:
            action = get_action_near_blocks(env)
            state, reward, done, info = env.step(action)
            if len(ep_frames) < 2 or np.abs(state[0] - ep_frames[-1]).sum() > 10:
                ep_frames.append(state[0])
                ep_actions.append(action)
            count += 1
            if count > 6 * max_steps:
                stucked = True
                break
        if stucked:
            continue

        frames.append(ep_frames)
        actions.append(ep_actions)
        if len(frames)==save_freq:
            num_files = len([f for f in os.listdir(DATA_DIR) if 'state_' in f])
            s_filename = os.path.join(DATA_DIR, 'state_%d.npy' % num_files)
            a_filename = os.path.join(DATA_DIR, 'action_%d.npy' % num_files)
            np.save(s_filename, (np.array(frames)*255).astype(np.uint8))
            np.save(a_filename, np.array(actions))
            frames.clear()
            actions.clear()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--num_blocks", default=3, type=int)
    parser.add_argument("--dist", default=0.08, type=float)
    parser.add_argument("--max_steps", default=50, type=int)
    parser.add_argument("--save_freq", default=100, type=int)
    parser.add_argument("--num_ep", default=5000, type=int)
    parser.add_argument("--camera_height", default=96, type=int)
    parser.add_argument("--camera_width", default=96, type=int)
    parser.add_argument("--xml", default=0, type=int)
    parser.add_argument("--num_process", default=4, type=int)
    args = parser.parse_args()

    if not os.path.isdir('data'):
        os.makedirs('data')

    num_process = args.num_process
    procs = []
    for p in range(num_process):
        proc = Process(target=collect_npy, args=(p, args,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()