import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../../ur5_mujoco'))
from object_env import *

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

    np.random.seed(process_id)
    env = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
                 control_freq=5, data_format='NCHW', xml_ver=xml_ver)
    env = objectwise_env(env, num_blocks=num_blocks, mov_dist=mov_dist, max_steps=max_steps, \
            conti=False, detection=False, reward_type=reward_type)

    save_freq = args.save_freq
    num_ep = args.num_ep

    buff_states = []
    buff_nextstates = []
    buff_actions = []
    buff_frames = []
    buff_nextframes = []
    buff_rewards = []
    buff_dones = []
    log_success = []

    state = env.reset()
    while len(buff_states) < args.num_data:
        action = [np.random.randint(env.num_blocks), np.random.randint(env.num_bins)]
        #action = get_action_near_blocks(env)
        next_state, reward, done, info = env.step(action)
        log_success.append(info['success'])

        buff_states.append(state[0])
        buff_nextstates.append(next_state[0])
        buff_frames.append(state[1])
        buff_nextframes.append(next_state[1])
        buff_actions.append(action)
        buff_rewards.append(reward)
        buff_dones.append(int(done))

        state = next_state
        if done:
            state = env.reset()

        if len(buff_states)%save_freq==0:
            num_files = len([f for f in os.listdir(args.data_dir) if 'reward_' in f])
            s_filename = os.path.join(args.data_dir, 'state_%d.npy' % num_files)
            ns_filename = os.path.join(args.data_dir, 'nextstate_%d.npy' % num_files)
            a_filename = os.path.join(args.data_dir, 'action_%d.npy' % num_files)
            r_filename = os.path.join(args.data_dir, 'reward_%d.npy' % num_files)
            d_filename = os.path.join(args.data_dir, 'done_%d.npy' % num_files)
            f_filename = os.path.join(args.data_dir, 'frame_%d.npy' % num_files)
            nf_filename = os.path.join(args.data_dir, 'nextframe_%d.npy' % num_files)

            np.save(s_filename, np.array(buff_states))
            np.save(ns_filename, np.array(buff_nextstates))
            np.save(f_filename, (np.array(buff_frames)*255).astype(np.uint8))
            np.save(nf_filename, (np.array(buff_nextframes)*255).astype(np.uint8))
            np.save(a_filename, np.array(buff_actions))
            np.save(r_filename, np.array(buff_rewards))
            np.save(d_filename, np.array(buff_dones))
            print('success rate:', np.sum(log_success)/len(log_success))

            buff_states.clear()
            buff_nextstates.clear()
            buff_frames.clear()
            buff_nextframes.clear()
            buff_actions.clear()
            buff_rewards.clear()
            buff_dones.clear()
            log_success.clear()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--num_blocks", default=3, type=int)
    parser.add_argument("--dist", default=0.06, type=float)
    parser.add_argument("--max_steps", default=50, type=int)
    parser.add_argument("--save_freq", default=2048, type=int)
    parser.add_argument("--num_ep", default=5000, type=int)
    parser.add_argument("--camera_height", default=96, type=int)
    parser.add_argument("--camera_width", default=96, type=int)
    parser.add_argument("--xml", default=0, type=int)
    parser.add_argument("--num_process", default=4, type=int)
    parser.add_argument("--num_data", default=50000, type=int)
    parser.add_argument("--data_dir", default='', type=str)
    args = parser.parse_args()

    args.data_dir = 'data/%dblock' %args.num_blocks
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
