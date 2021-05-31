import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../ur5_mujoco'))
from pushpixel_env import *

import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
#from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory

crop_min = 19
crop_max = 78
if not os.path.exists("results/graph/"):
    os.makedirs("results/graph/")
if not os.path.exists("results/models/"):
    os.makedirs("results/models/")
if not os.path.exists("results/board/"):
    os.makedirs("results/board/")

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=200001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=1000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=10000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')

parser.add_argument("--render", action="store_true")
parser.add_argument("--num_blocks", default=1, type=int)
parser.add_argument("--dist", default=0.08, type=float)
parser.add_argument("--max_steps", default=20, type=int)
parser.add_argument("--camera_height", default=96, type=int)
parser.add_argument("--camera_width", default=96, type=int)
parser.add_argument("--log_freq", default=100, type=int)
parser.add_argument("--reward", default="binary", type=str)
args = parser.parse_args()

render = args.render
task = 2
num_blocks = args.num_blocks
mov_dist = args.dist
max_steps = int(args.max_steps)
camera_height = args.camera_height
camera_width = args.camera_width
reward_type = args.reward
log_freq = args.log_freq
#her = args.her
now = datetime.datetime.now()
savename = "SAC_%s"%(now.strftime("%m%d_%H%M"))

# Environment
env = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
        control_freq=5, data_format='NCHW', xml_ver=0)
env = pushpixel_env(env, num_blocks=num_blocks, mov_dist=mov_dist, max_steps=max_steps,task=task,\
                    reward_type = reward_type)

observation_space = 6 * num_blocks
action_space = 4

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = SAC(observation_space, action_space, args)

def smoothing_log(log_data):
    return np.convolve(log_data, np.ones(log_freq), 'valid') / log_freq

#Tesnorboard
#writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name, args.policy, "autotune" if args.automatic_entropy_tuning else ""))

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

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0
max_success = 0.0

log_returns = []
log_critic_loss = []
log_actor_loss = []
log_eplen = []
log_out = []
log_success = []
log_collisions = []
for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    ep_critic_loss = []
    ep_actor_loss = []
    num_collisions = 0
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            # Sample random action
            action = [np.random.randint(crop_min,crop_max), np.random.randint(crop_min,crop_max), np.random.randint(env.num_bins)]
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                '''
                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                '''
                updates += 1
                ep_critic_loss.append((critic_1_loss + critic_2_loss)/2.)
                ep_actor_loss.append(policy_loss)

        next_state, reward, done, info = env.step(action) # Step
        num_collisions += int(info['collision'])
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == max_steps else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break

    #writer.add_scalar('reward/train', episode_reward, i_episode)
    #print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    log_returns.append(episode_reward)
    log_critic_loss.append(np.mean(ep_critic_loss))
    log_actor_loss.append(np.mean(ep_actor_loss))
    log_eplen.append(episode_steps)
    log_out.append(int(info['out_of_range']))
    log_success.append(int(info['success']))
    log_collisions.append(num_collisions)
    if i_episode%log_freq==0:
        log_mean_returns = smoothing_log(log_returns)
        log_mean_critic_loss = smoothing_log(log_critic_loss)
        log_mean_actor_loss = smoothing_log(log_actor_loss)
        log_mean_eplen = smoothing_log(log_eplen)
        log_mean_out = smoothing_log(log_out)
        log_mean_success = smoothing_log(log_success)
        log_mean_collisions = smoothing_log(log_collisions)

        print()
        print("{} episodes.".format(i_episode))
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
            print("Max performance! saving the model.")

    if False and i_episode % 10 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward


                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes


        #writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")


