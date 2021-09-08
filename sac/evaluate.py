import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../ur5_mujoco'))
from pushpixel_env import *
from continuous_env import *

import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
#from torch.utils.tensorboard import SummaryWriter

crop_min = 19
crop_max = 78

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env', default="relative",
                    help='Env Type: relative | pushpixel (default: relative)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--cuda', action="store_false",
                    help='run on CUDA (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.1, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')

parser.add_argument("--model_name", default="", type=str)
parser.add_argument("--episode", default=100, type=int)
parser.add_argument("--render", action="store_true")
parser.add_argument("--rotation", action="store_true")
parser.add_argument("--num_blocks", default=1, type=int)
parser.add_argument("--dist", default=0.08, type=float)
parser.add_argument("--max_steps", default=30, type=int)
parser.add_argument("--camera_height", default=96, type=int)
parser.add_argument("--camera_width", default=96, type=int)
parser.add_argument("--log_freq", default=100, type=int)
parser.add_argument("--reward", default="new", type=str)
args = parser.parse_args()

render = args.render
rotation = args.rotation
if rotation: task = 3
else: task = 2
num_blocks = args.num_blocks
mov_dist = args.dist
max_steps = int(args.max_steps)
camera_height = args.camera_height
camera_width = args.camera_width
reward_type = args.reward
log_freq = args.log_freq
#her = args.her
now = datetime.datetime.now()
savename = "SAC_%s"%(args.model_name)
episode = args.episode

# Environment
env = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
        control_freq=5, data_format='NCHW', xml_ver=0)
if args.env=='relative':
    env = continuous_env(env, num_blocks=num_blocks, mov_dist=mov_dist, max_steps=max_steps,\
            task=1, reward_type=reward_type)
    action_space = 2
    observation_space = 4 * num_blocks + 2
else:
    env = pushpixel_env(env, num_blocks=num_blocks, mov_dist=mov_dist, max_steps=max_steps,\
            task=task, reward_type=reward_type)
    action_space = 4
    if rotation:
        observation_space = 6 * num_blocks
    else:
        observation_space = 4 * num_blocks

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = SAC(observation_space, action_space, args)
actor_path = "results/models/{}/actor_{}".format(savename, episode)
critic_path = "results/models/{}/critic_{}".format(savename, episode)
agent.load_model(actor_path, critic_path)

log_returns = []
log_eplen = []
log_out = []
log_success = []
log_collisions = []

episode_reward = 0
episode_steps = 0
num_collisions = 0
done = False
state = env.reset()

avg_reward = 0.
episodes = 10
for i_episode in range(episodes):
    state = env.reset()
    episode_reward = 0
    done = False
    while not done:
        action = agent.select_action(state, evaluate=True)
        #action = agent.process_action(action_raw)

        next_state, reward, done, info = env.step(action)

        episode_reward += reward
        num_collisions += int(info['collision'])
        episode_steps += 1

        state = next_state
    log_success.append(int(info['success']))
    log_returns.append(episode_reward)
    log_eplen.append(episode_steps)
    log_out.append(int(info['out_of_range']))
    log_collisions.append(num_collisions)

    print("{} episodes.".format(i_episode))
    print("Success rate: {0:.2f}".format(np.mean(log_success)))
    print("Mean reward: {0:.2f}".format(np.mean(log_returns)))
    print("Ep length: {}".format(np.mean(log_eplen)))
    print("Out of range: {}".format(np.mean(log_out)))
    print("Collisions: {}".format(np.mean(log_collisions)))

#writer.add_scalar('avg_reward/test', avg_reward, i_episode)

print("----------------------------------------")
print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(np.mean(log_returns), 2)))
print("Success rate: {0:.2f}".format(np.mean(log_success)))
print("----------------------------------------")


