import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../ur5_mujoco'))
from pushpixel_env import *

import numpy as np
import torch
from torch.autograd import Variable
import os
import psutil
import gc

import train
import buffer

import argparse
import datetime

from utils import sample_her_transitions, sample_ig_transitions, get_action_near_blocks

crop_min = 19
crop_max = 78

if not os.path.exists("results/graph/"):
    os.makedirs("results/graph/")
if not os.path.exists("results/models/"):
    os.makedirs("results/models/")
if not os.path.exists("results/board/"):
    os.makedirs("results/board/")

parser = argparse.ArgumentParser()
parser.add_argument("--render", action="store_true")
parser.add_argument("--rotation", action="store_true")
parser.add_argument("--num_blocks", default=1, type=int)
parser.add_argument("--dist", default=0.08, type=float)
parser.add_argument("--max_steps", default=20, type=int)
parser.add_argument("--camera_height", default=96, type=int)
parser.add_argument("--camera_width", default=96, type=int)
parser.add_argument("--lr", default=1e-5, type=float)
parser.add_argument("--bs", default=64, type=int)
parser.add_argument("--buff_size", default=1e4, type=int)
parser.add_argument("--max_episodes", default=1e4, type=int)
parser.add_argument("--log_freq", default=100, type=int)
parser.add_argument("--reward", default="new", type=str)
parser.add_argument("--her", action="store_false")
args = parser.parse_args()

# env configuration #
render = args.render
rotation = args.rotation
if rotation: task = 3
else: task = 2
num_blocks = args.num_blocks
mov_dist = args.dist
max_steps = int(args.max_steps)
max_episodes = int(args.max_episodes)
camera_height = args.camera_height
camera_width = args.camera_width
reward_type = args.reward
her = args.her

buff_size = int(args.buff_size)
log_freq = args.log_freq

env = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
        control_freq=5, data_format='NCHW', xml_ver=0)
env = pushpixel_env(env, num_blocks=num_blocks, mov_dist=mov_dist, max_steps=max_steps,task=task,\
                    reward_type = reward_type)

now = datetime.datetime.now()
savename = "DDPG_%s"%(now.strftime("%m%d_%H%M"))

if rotation:
    S_DIM = 6 * num_blocks
else:
    S_DIM = 4 * num_blocks
A_DIM = 4
A_MAX = 1.0

#print ' State Dimensions :- ', S_DIM
#print ' Action Dimensions :- ', A_DIM
#print ' Action Max :- ', A_MAX

ram = buffer.MemoryBuffer(buff_size)
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram, args)

def smoothing_log(log_data):
    return np.convolve(log_data, np.ones(log_freq), 'valid') / log_freq

def clip_action(action):
    pose = (action[:2] + 1.0) * env.env.camera_width / 2
    px, py = np.clip(pose, crop_min, crop_max).astype(int)
    theta = np.arctan2(action[2], action[3])
    theta = int((theta/np.pi)%2 * 4)
    return px, py, theta


#plt.show(block=False)
plt.rc('axes', labelsize=6)
plt.rc('font', size=6)
f, axes = plt.subplots(4, 2)
f.set_figheight(9) #15
f.set_figwidth(12) #10

axes[0][0].set_title('Critic Loss')
axes[1][0].set_title('Actor Loss')
axes[2][0].set_title('Episode Return')
axes[3][0].set_title('Episode Length')
axes[0][1].set_title('Success Rate')
axes[1][1].set_title('Out of Range')
axes[2][1].set_title('Num Collisions')

action_count_map = np.zeros([96, 96, 8])
plt.rc('font', size=6)
fig, axis = plt.subplots(3, 3)
plt.show(block=False)
fig.canvas.draw()
fig.canvas.draw()

max_success = 0.0
log_returns = []
log_critic_loss = []
log_actor_loss = []
log_eplen = []
log_out = []
log_success = []
log_collisions = []
for ne in range(max_episodes):
    episode_reward = 0
    ep_critic_loss = []
    ep_actor_loss = []
    ep_len = 0
    num_collisions = 0

    observation = env.reset()
    for r in range(max_steps):
        state = np.float32(observation)

        action = trainer.get_exploration_action(state)
        action_clipped = clip_action(action)
        new_observation, reward, done, info = env.step(action_clipped)
        episode_reward += reward

        action_count_map[action_clipped[0], action_clipped[1], action_clipped[2]] += 1
        for i in range(8):
            axis[i//3][i%3].imshow(action_count_map[:, :, i])
            axis[i//3][i%3].set_title(action_count_map[:, :, i].sum())
        axis[2][2].imshow(action_count_map.sum(axis=-1))
        fig.canvas.draw()

        new_state = np.float32(new_observation)
        # push this exp in ram
        ram.add(state, action, reward, new_state, int(done))

        if her and not done:
            her_sample = sample_her_transitions(env, info)
            ig_samples = sample_ig_transitions(env, info, num_samples=3)
            samples = her_sample + ig_samples
            for sample in samples:
                reward_re, goal_re, done_re, block_success_re = sample
                state_re = deepcopy(state)
                state_re[2*env.num_blocks:4*env.num_blocks] = goal_re
                ram.add(state_re, action, reward_re, new_state, int(done_re))

        observation = new_observation

        # perform optimization
        loss_c, loss_a = trainer.optimize()
        ep_critic_loss.append(loss_c.data.detach().cpu().numpy())
        ep_actor_loss.append(loss_a.data.detach().cpu().numpy())
        ep_len += 1
        num_collisions += int(info['collision'])
        if done:
            break

    gc.collect()

    log_returns.append(episode_reward)
    log_critic_loss.append(np.mean(ep_critic_loss))
    log_actor_loss.append(np.mean(ep_actor_loss))
    log_eplen.append(ep_len)
    log_out.append(int(info['out_of_range']))
    log_success.append(int(info['success']))
    log_collisions.append(num_collisions)
    if (ne+1)%log_freq==0:
        log_mean_returns = smoothing_log(log_returns)
        log_mean_critic_loss = smoothing_log(log_critic_loss)
        log_mean_actor_loss = smoothing_log(log_actor_loss)
        log_mean_eplen = smoothing_log(log_eplen)
        log_mean_out = smoothing_log(log_out)
        log_mean_success = smoothing_log(log_success)
        log_mean_collisions = smoothing_log(log_collisions)

        print()
        print("{} episodes.".format(ne))
        print("Success rate: {0:.2f}".format(log_mean_success[-1]))
        print("Mean reward: {0:.2f}".format(log_mean_returns[-1]))
        print("Mean critic loss: {0:.6f}".format(log_mean_critic_loss[-1]))
        print("Mean actor loss: {0:.6f}".format(log_mean_actor_loss[-1]))
        print("Ep length: {}".format(log_mean_eplen[-1]))

        axes[0][0].plot(log_critic_loss, color='#ff7f00', linewidth=0.5)
        axes[1][0].plot(log_actor_loss, color='#ff7f00', linewidth=0.5)
        axes[2][0].plot(log_returns, color='#60c7ff', linewidth=0.5)
        axes[3][0].plot(log_eplen, color='#83dcb7', linewidth=0.5)
        axes[2][1].plot(log_collisions, color='#ff33cc', linewidth=0.5)

        axes[0][0].plot(log_mean_critic_loss, color='red')
        axes[1][0].plot(log_mean_actor_loss, color='red')
        axes[2][0].plot(log_mean_returns, color='blue')
        axes[3][0].plot(log_mean_eplen, color='green')
        axes[0][1].plot(log_mean_success, color='red')
        axes[1][1].plot(log_mean_out, color='black')
        axes[2][1].plot(log_mean_collisions, color='#663399')

        f.canvas.draw()
        plt.savefig('results/graph/%s.png'%savename)

        numpy_log = np.array([
            log_returns, #0
            log_critic_loss, #1
            log_actor_loss, #2
            log_eplen, #3
            log_success, #4
            log_collisions, #5
            log_out #6
            ])
        np.save('results/board/%s' %savename, numpy_log)
        
        if not os.path.exists("results/models/"+savename):
            os.makedirs("results/models/"+savename)
        if log_mean_success[-1] > max_success:
            max_success = log_mean_success[-1]
            trainer.save_models(savename, ne)
            print("Max performance! saving the model.")

print('Completed episodes')
