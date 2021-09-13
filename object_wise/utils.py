import cv2
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn


dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
criterion = nn.SmoothL1Loss(reduction=None).type(dtype)

def smoothing_log(log_data, log_freq):
    return np.convolve(log_data, np.ones(log_freq), 'valid') / log_freq

def smoothing_log_same(log_data, log_freq):
    return np.concatenate([np.array([np.nan] * (log_freq-1)), np.convolve(log_data, np.ones(log_freq), 'valid') / log_freq])

def combine_batch(minibatch, data):
    combined = []
    if minibatch is None:
        for i in range(len(data)):
            combined.append(data[i].unsqueeze(0))
    else:
        for i in range(len(minibatch)):
            combined.append(torch.cat([minibatch[i], data[i].unsqueeze(0)]))
    return combined

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

    return [[reward_recompute, _info['goals'], done_recompute, block_success_recompute]]

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
        transitions.append([reward_recompute, _info['goals'], done_recompute, block_success_recompute])

    return transitions



## FCDQN Loss ##
def calculate_loss_origin(minibatch, Q, Q_target, gamma=0.5):
    state = minibatch[0]
    next_state = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal = minibatch[5]
    batch_size = state.size()[0]

    state_goal = [state, goal]
    next_state_goal = [next_state, goal]

    next_q = Q_target(next_state_goal)
    next_q_max = next_q.max(1)[0].max(1)[0]
    y_target = rewards + gamma * not_done * next_q_max

    q_values = Q(state_goal)
    pred = q_values[torch.arange(batch_size), actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error

def calculate_loss_double(minibatch, Q, Q_target, gamma=0.5):
    state = minibatch[0]
    next_state = minibatch[1]
    rewards = minibatch[3]
    actions = minibatch[2].type(torch.long)
    not_done = minibatch[4]
    goal = minibatch[5]
    batch_size = state.size()[0]

    state_goal = [state, goal]
    next_state_goal = [next_state, goal]

    def get_a_prime():
        next_q = Q(next_state_goal)
        obj = next_q.max(2)[0].max(1)[1]
        theta = next_q.max(1)[0].max(1)[1]
        return obj, theta

    a_prime = get_a_prime()

    next_q_target = Q_target(next_state_goal)
    q_target_s_a_prime = next_q_target[torch.arange(batch_size), a_prime[0], a_prime[1]].unsqueeze(1)
    y_target = rewards + gamma * not_done * q_target_s_a_prime

    q_values = Q(state_goal)
    pred = q_values[torch.arange(batch_size), actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error


