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
from copy import deepcopy
from sac import SAC
#from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
from utils import sample_her_transitions, sample_ig_transitions, get_action_near_blocks
import wandb

if not os.path.exists("results/graph/"):
    os.makedirs("results/graph/")
if not os.path.exists("results/models/"):
    os.makedirs("results/models/")
if not os.path.exists("results/board/"):
    os.makedirs("results/board/")

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env', default="relative",
                    help='Env Type: relative | pushpixel (default: relative)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.1, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=1000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1e5, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_false",
                    help='run on CUDA (default: True)')

# env config
parser.add_argument("--render", action="store_true")
parser.add_argument("--num_blocks", default=1, type=int)
parser.add_argument("--dist", default=0.06, type=float)
parser.add_argument("--max_steps", default=100, type=int)
parser.add_argument("--camera_height", default=96, type=int)
parser.add_argument("--camera_width", default=96, type=int)
parser.add_argument("--n1", default=3, type=int)
parser.add_argument("--n2", default=5, type=int)
parser.add_argument("--max_blocks", default=8, type=int)
parser.add_argument("--threshold", default=0.10, type=float)
parser.add_argument("--sdf_action", action="store_false")
parser.add_argument("--real_object", action="store_false")
parser.add_argument("--dataset", default="train", type=str)
parser.add_argument("--max_steps", default=100, type=int)
parser.add_argument("--reward", default="linear_maskpenalty", type=str)
# sdf config
parser.add_argument("--convex_hull", action="store_true")
parser.add_argument("--oracle", action="store_true")
parser.add_argument("--tracker", default="medianflow", type=str)
parser.add_argument("--depth", action="store_true")
parser.add_argument("--clip", action="store_true")
parser.add_argument("--round_sdf", action="store_false")
# learning params #
parser.add_argument("--resize", action="store_false") # defalut: True
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--bs", default=12, type=int)
parser.add_argument("--buff_size", default=1e5, type=float)
parser.add_argument("--total_episodes", default=1e4, type=float)
parser.add_argument("--learn_start", default=300, type=float)
parser.add_argument("--update_freq", default=100, type=int)
parser.add_argument("--log_freq", default=50, type=int)
parser.add_argument("--double", action="store_false")
parser.add_argument("--per", action="store_true")
parser.add_argument("--her", action="store_false")
# gcn #
parser.add_argument("--ver", default=0, type=int)
parser.add_argument("--adj_ver", default=1, type=int)
parser.add_argument("--selfloop", action="store_true")
parser.add_argument("--normalize", action="store_true")
parser.add_argument("--separate", action="store_true")
parser.add_argument("--bias", action="store_false")
# etc #
parser.add_argument("--show_q", action="store_true")
parser.add_argument("--seed", default=None, type=int)
parser.add_argument("--gpu", default=-1, type=int)
parser.add_argument("--wandb_off", action="store_true")
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
n1 = args.n1
n2 = args.n2
max_blocks = args.max_blocks
sdf_action = args.sdf_action
real_object = args.real_object
dataset = args.dataset
depth = args.depth
threshold = args.threshold
mov_dist = args.dist
max_steps = args.max_steps
camera_height = args.camera_height
camera_width = args.camera_width
reward_type = args.reward
gpu = args.gpu

if "CUDA_VISIBLE_DEVICES" in os.environ:
    visible_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if str(gpu) in visible_gpus:
        gpu_idx = visible_gpus.index(str(gpu))
        torch.cuda.set_device(gpu_idx)

model_path = os.path.join("results/models/SAC_%s.pth"%args.model_path)
visualize_q = args.show_q

now = datetime.datetime.now()
savename = "SAC_%s" % (now.strftime("%m%d_%H%M"))
if not os.path.exists("results/config/"):
    os.makedirs("results/config/")
with open("results/config/%s.json" % savename, 'w') as cf:
    json.dump(args.__dict__, cf, indent=2)

convex_hull = args.convex_hull
oracle_matching = args.oracle
tracker = args.tracker
resize = args.resize
sdf_module = SDFModule(rgb_feature=True, resnet_feature=True, convex_hull=convex_hull, 
        binary_hole=True, using_depth=depth, tracker=tracker, resize=resize)

if real_object:
    from realobjects_env import UR5Env
else:
    from ur5_env import UR5Env
if dataset=="train":
    urenv1 = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
            control_freq=5, data_format='NHWC', gpu=gpu, camera_depth=True, dataset="train1")
    env1 = objectwise_env(urenv1, num_blocks=n1, mov_dist=mov_dist, max_steps=max_steps, \
            threshold=threshold, conti=False, detection=True, reward_type=reward_type)
    urenv2 = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
            control_freq=5, data_format='NHWC', gpu=gpu, camera_depth=True, dataset="train2")
    env2 = objectwise_env(urenv2, num_blocks=n1, mov_dist=mov_dist, max_steps=max_steps, \
            threshold=threshold, conti=False, detection=True, reward_type=reward_type)
    env = [env1, env2]
else:
    urenv = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
            control_freq=5, data_format='NHWC', gpu=gpu, camera_depth=True, dataset="test")
    env = [objectwise_env(urenv, num_blocks=n1, mov_dist=mov_dist, max_steps=max_steps, \
            threshold=threshold, conti=False, detection=True, reward_type=reward_type)]

# learning configuration #
learning_rate = args.lr
batch_size = args.bs 
buff_size = int(args.buff_size)
total_episodes = int(args.total_episodes)
learn_start = int(args.learn_start)
update_freq = args.update_freq
log_freq = args.log_freq
double = args.double
per = args.per
her = args.her
ver = args.ver
adj_ver = args.adj_ver
selfloop = args.selfloop
graph_normalize = args.normalize
separate = args.separate
bias = args.bias
clip_sdf = args.clip
round_sdf = args.round_sdf

pretrain = args.pretrain
continue_learning = args.continue_learning

# wandb model name #
wandb_off = args.wandb_off
if not wandb_off:
    log_name = savename
    if n1==n2:
        log_name += '_%db' %n1
    else:
        log_name += '_%d-%db' %(n1, n2)
    log_name += '_v%d' %ver
    log_name += 'a%d' %adj_ver
    wandb.init(project="TrackGCN")
    wandb.run.name = log_name
    wandb.config.update(args)
    wandb.run.save()

# Agent
agent = SAC(max_blocks, args)

def smoothing_log(log_data):
    return np.convolve(log_data, np.ones(log_freq), 'valid') / log_freq

# Memory
# TODO
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
    state_goal, _ = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            # Sample random action
            sidx, action = agent.random_action(sdfs)

        else:
            sidx, action = agent.select_action(sdfs, evaluate=False)  # Sample action from policy

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

        (next_state_goal, _), reward, done, info = env.step(action) # Step
        num_collisions += int(info['collision'])
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == max_steps else float(not done)

        memory.push(state_goal[0], action, reward, next_state_goal[0], mask, state_goal[1]) # Append transition to memory

        if her and not done:
            her_sample = sample_her_transitions(env, info)
            ig_samples = sample_ig_transitions(env, info, num_samples=3)
            samples = her_sample + ig_samples
            for sample in samples:
                reward_re, goal_re, done_re, block_success_re = sample
                state_re = deepcopy(state)
                state_re[-2*env.num_blocks:] = goal_re
                # Append HER transition to memory
                memory.push(state_re, action, reward_re, next_state, mask) 

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
            actor_path = "results/models/{}/actor_{}".format(savename, i_episode)
            critic_path = "results/models/{}/critic_{}".format(savename, i_episode)
            agent.save_model(actor_path, critic_path)

