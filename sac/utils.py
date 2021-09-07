import math
import torch
import numpy as np
from copy import deepcopy

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def get_action_near_blocks(env, pad=0.06):
    poses, _ = env.get_poses()
    obj = np.random.randint(len(poses))
    pose = poses[obj]
    x = np.random.uniform(pose[0]-pad, pose[0]+pad)
    y = np.random.uniform(pose[1]-pad, pose[1]+pad)
    py, px = env.pos2pixel(x, y)
    a0 = 2 * px / env.env.camera_width - 1
    a1 = 2 * py / env.env.camera_width - 1
    a2 = 50 * (- x + pose[0]) + np.random.normal(0, 0.1)
    a3 = 50 * (- y + pose[1]) + np.random.normal(0, 0.1)
    action = np.array([a0, a1, a2, a3])
    return action

def sample_her_transitions(env, info):
    _info = deepcopy(info)
    move_threshold = 0.005
    range_x = env.block_range_x
    range_y = env.block_range_y

    pre_poses = info['pre_poses']
    poses = info['poses']
    pos_diff = np.linalg.norm(poses - pre_poses, axis=1)
    if np.linalg.norm(poses - pre_poses) < move_threshold:
        return []

    for i in range(env.num_blocks):
        if pos_diff[i] < move_threshold:
            continue
        ## 1. archived goal ##
        archived_goal = poses[i]

        ## clipping goal pose ##
        x, y = archived_goal
        _info['goals'][i] = np.array([x, y])

    ## recompute reward  ##
    reward_recompute, done_recompute, block_success_recompute = env.get_reward(_info)
    if _info['out_of_range']:
        if env.seperate:
            reward_recompute = [-1.] * env.num_blocks
        else:
            reward_recompute = -1.

    return [[reward_recompute, _info['goals'].flatten(), done_recompute, block_success_recompute]]

def sample_ig_transitions(env, info, num_samples=1):
    move_threshold = 0.005
    range_x = env.block_range_x
    range_y = env.block_range_y
    n_blocks = env.num_blocks

    pre_poses = info['pre_poses']
    poses = info['poses']
    pos_diff = np.linalg.norm(poses - pre_poses, axis=1)
    if np.linalg.norm(poses - pre_poses) < move_threshold:
        return []

    transitions = []
    for s in range(num_samples):
        _info = deepcopy(info)
        for i in range(n_blocks):
            if pos_diff[i] < move_threshold:
                continue
            ## 1. archived goal ##
            gx = np.random.uniform(*range_x)
            gy = np.random.uniform(*range_y)
            archived_goal = np.array([gx, gy])
            _info['goals'][i] = archived_goal

        ## recompute reward  ##
        reward_recompute, done_recompute, block_success_recompute = env.get_reward(_info)
        if _info['out_of_range']:
            if env.seperate:
                reward_recompute = [-1.] * env.num_blocks
            else:
                reward_recompute = -1.
        transitions.append([reward_recompute, _info['goals'].flatten(), done_recompute, block_success_recompute])

    return transitions
