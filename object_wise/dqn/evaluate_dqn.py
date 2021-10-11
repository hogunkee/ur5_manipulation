import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../../ur5_mujoco'))
from object_env import *

from utils import *

import torch
import torch.nn as nn
import argparse
import json

import datetime
import random

from replay_buffer import ReplayBuffer, PER
from matplotlib import pyplot as plt

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def get_action(env, qnet, state_goal, epsilon, with_q=False):
    if np.random.random() < epsilon:
        action = [np.random.randint(env.num_blocks), np.random.randint(env.num_bins)]
        if with_q:
            state, goal = state_goal
            state = torch.tensor(state).type(dtype).unsqueeze(0)
            goal = torch.tensor(goal).type(dtype).unsqueeze(0)
            q_value = qnet([state, goal])
            q = q_value[0].detach().cpu().numpy()
    else:
        state, goal = state_goal
        state = torch.tensor(state).type(dtype).unsqueeze(0)
        goal = torch.tensor(goal).type(dtype).unsqueeze(0)
        q_value = qnet([state, goal])
        q = q_value[0].detach().cpu().numpy()

        obj = q.max(1).argmax()
        theta = q.max(0).argmax()
        action = [obj, theta]

    if with_q:
        return action, q
    else:
        return action

def evaluate(env,
        n_actions=8,
        model_path='',
        num_trials=100,
        visualize_q=False,
        ):
    qnet = QNet(n_actions, env.num_blocks).type(dtype)
    qnet.load_state_dict(torch.load(model_path))
    print('Loading trained model: {}'.format(model_path))

    log_returns = []
    log_eplen = []
    log_out = []
    log_success = []
    log_success_block = [[] for i in range(env.num_blocks)]

    epsilon = 0.0
    for ne in range(num_trials):
        ep_len = 0
        episode_reward = 0
        state_goal, _ = env.reset()
        for t_step in range(env.max_steps):
            action, q_map = get_action(env, qnet, state_goal, epsilon=epsilon, with_q=True)

            (next_state_goal, _), reward, done, info = env.step(action)
            episode_reward += reward
            # print(info['block_success'])

            ep_len += 1
            state_goal = next_state_goal
            if done:
                break

        log_returns.append(episode_reward)
        log_eplen.append(ep_len)
        log_out.append(int(info['out_of_range']))
        log_success.append(int(np.all(info['block_success'])))
        #log_success.append(int(info['success']))
        for o in range(env.num_blocks):
            log_success_block[o].append(int(info['block_success'][o]))

        print()
        print("{} episodes.".format(ne))
        print("Ep reward: {}".format(log_returns[-1]))
        print("Ep length: {}".format(log_eplen[-1]))
        print("Success rate: {}% ({}/{})".format(100*np.mean(log_success), np.sum(log_success), len(log_success)))
        for o in range(env.num_blocks):
            print("Block {}: {}% ({}/{})".format(o+1, 100*np.mean(log_success_block[o]), np.sum(log_success_block[o]), len(log_success_block[o])))
        print("Out of range: {}".format(np.mean(log_out)))

    print()
    print("="*80)
    print("Evaluation Done.")
    print("Mean reward: {0:.2f}".format(np.mean(log_returns)))
    print("Mean episode length: {}".format(np.mean(log_eplen)))
    print("Success rate: {}".format(100*np.mean(log_success)))
    for o in range(env.num_blocks):
        print("Block {}: {}% ({}/{})".format(o+1, 100*np.mean(log_success_block[o]), np.sum(log_success_block[o]), len(log_success_block[o])))
    print("Out of range: {}".format(np.mean(log_out)))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--num_blocks", default=3, type=int)
    parser.add_argument("--dist", default=0.06, type=float)
    parser.add_argument("--max_steps", default=100, type=int)
    parser.add_argument("--camera_height", default=96, type=int)
    parser.add_argument("--camera_width", default=96, type=int)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--ver", default=1, type=int)
    parser.add_argument("--reward", default="new", type=str)
    parser.add_argument("--model_path", default="1006_1452", type=str)
    parser.add_argument("--num_trials", default=100, type=int)
    parser.add_argument("--show_q", action="store_true")
    parser.add_argument("--seed", default=None, type=int)
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
    mov_dist = args.dist
    max_steps = args.max_steps
    camera_height = args.camera_height
    camera_width = args.camera_width
    reward_type = args.reward

    # evaluate configuration
    num_trials = args.num_trials
    model_path = os.path.join("results/models/QOBJ_%s.pth" % args.model_path)
    visualize_q = args.show_q
    if visualize_q:
        render = True

    env = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
                 control_freq=5, data_format='NHWC', xml_ver=0)
    env = objectwise_env(env, num_blocks=num_blocks, mov_dist=mov_dist, max_steps=max_steps, \
                         conti=False, detection=False, reward_type=reward_type)

    graph = args.graph
    ver = args.ver
    if graph:
        if ver == 1:
            from models.gcn_dqn import ObjectQNet as QNet
        elif ver == 2:
            from models.gcn_dqn_v2 import ObjectQNet as QNet
    else:
        from models.object_dqn import ObjectQNet as QNet

    evaluate(env=env, n_actions=8, model_path=model_path,\
            num_trials=num_trials, visualize_q=visualize_q)
