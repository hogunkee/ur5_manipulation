import cv2
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn


dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
criterion = nn.SmoothL1Loss(reduction=None).type(dtype)

def smoothing_log(log_data, log_freq):
    return np.convolve(log_data, np.ones(log_freq), 'valid') / log_freq

def combine_batch(minibatch, data):
    combined = []
    for i in range(len(minibatch)):
        combined.append(torch.cat([minibatch[i], data[i].unsqueeze(0)]))
    return combined

def sample_her_transitions(env, info, next_state, targets=[0,1,2]):
    _info = deepcopy(info)
    move_threshold = 0.005
    range_x = env.block_range_x
    range_y = env.block_range_y

    pre_poses = info['pre_poses']
    poses = info['poses']
    pos_diff = np.linalg.norm(poses - pre_poses, axis=1)
    if np.linalg.norm(poses - pre_poses) < move_threshold:
        return []

    if env.goal_type=='circle':
        goal_image = deepcopy(env.background_img)
        for i in range(env.num_blocks):
            if i not in targets:
                continue
            if pos_diff[i] < move_threshold:
                continue
            ## 1. archived goal ##
            archived_goal = poses[i]

            ## clipping goal pose ##
            x, y = archived_goal
            x = np.max((x, range_x[0]))
            x = np.min((x, range_x[1]))
            y = np.max((y, range_y[0]))
            y = np.min((y, range_y[1]))
            archived_goal = np.array([x, y])
            _info['goals'][i] = archived_goal
        _info['goal_flags'] = np.linalg.norm(_info['goals'] - _info['poses'], axis=1) < env.threshold
        ## generate goal image ##
        for i in range(env.num_blocks):
            if env.hide_goal and _info['goal_flags'][i]:
                continue
            cv2.circle(goal_image, env.pos2pixel(*_info['goals'][i]), 1, env.colors[i], -1)
        goal_image = np.transpose(goal_image, [2, 0, 1])

    elif env.goal_type=='pixel':
        goal_image = deepcopy(env.background_img)
        for i in range(env.num_blocks):
            if i not in targets:
                continue
            if pos_diff[i] < move_threshold:
                continue
            ## 1. archived goal ##
            archived_goal = poses[i]
            ## clipping goal pose ##
            x, y = archived_goal
            x = np.max((x, range_x[0]))
            x = np.min((x, range_x[1]))
            y = np.max((y, range_y[0]))
            y = np.min((y, range_y[1]))
            archived_goal = np.array([x, y])
            _info['goals'][i] = archived_goal
        _info['goal_flags'] = np.linalg.norm(_info['goals'] - _info['poses'], axis=1) < env.threshold
        ## generate goal image ##
        goal_ims = []
        for i in range(env.num_blocks):
            zero_array = np.zeros([env.env.camera_height, env.env.camera_width])
            if not (env.hide_goal and _info['goal_flags'][i]):
                cv2.circle(zero_array, env.pos2pixel(*_info['goals'][i]), 1, 1, -1)
            goal_ims.append(zero_array)
        goal_image = np.concatenate(goal_ims)
        goal_image = goal_image.reshape([env.num_blocks, env.env.camera_height, env.env.camera_width])

    elif env.goal_type=='block':
        for i in range(env.num_blocks):
            if i not in targets:
                continue
            if pos_diff[i] < move_threshold:
                continue
            x, y = poses[i]
            _info['goals'][i] = np.array([x, y])
        goal_image = deepcopy(next_state[0])

    ## recompute reward  ##
    reward_recompute, done_recompute, block_success_recompute = env.get_reward(_info)
    if _info['out_of_range']:
        if env.seperate:
            reward_recompute = [-1.] * env.num_blocks
        else:
            reward_recompute = -1.

    return [[reward_recompute, goal_image, done_recompute, block_success_recompute]]

def sample_ig_transitions(env, info, next_state, num_samples=1, targets=[0,1,2]):
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
        if env.goal_type=='circle':
            goal_image = deepcopy(env.background_img)
            for i in range(n_blocks):
                if i not in targets:
                    continue
                if pos_diff[i] < move_threshold:
                    continue
                ## 1. archived goal ##
                gx = np.random.uniform(*range_x)
                gy = np.random.uniform(*range_y)
                archived_goal = np.array([gx, gy])

                _info['goals'][i] = archived_goal
            _info['goal_flags'] = np.linalg.norm(_info['goals'] - _info['poses'], axis=1) < env.threshold
            ## generate goal image ##
            for i in range(n_blocks):
                if env.hide_goal and _info['goal_flags'][i]:
                    continue
                cv2.circle(goal_image, env.pos2pixel(*_info['goals'][i]), 1, env.colors[i], -1)
            goal_image = np.transpose(goal_image, [2, 0, 1])

        elif env.goal_type=='pixel':
            for i in range(n_blocks):
                if i not in targets:
                    continue
                if pos_diff[i] < move_threshold:
                    continue
                ## 1. archived goal ##
                gx = np.random.uniform(*range_x)
                gy = np.random.uniform(*range_y)
                archived_goal = np.array([gx, gy])

                _info['goals'][i] = archived_goal
            _info['goal_flags'] = np.linalg.norm(_info['goals'] - _info['poses'], axis=1) < env.threshold
            ## generate goal image ##
            goal_ims = []
            for i in range(n_blocks):
                zero_array = np.zeros([env.env.camera_height, env.env.camera_width])
                if not (env.hide_goal and _info['goal_flags'][i]):
                    cv2.circle(zero_array, env.pos2pixel(*_info['goals'][i]), 1, 1, -1)
                goal_ims.append(zero_array)
            goal_image = np.concatenate(goal_ims)
            goal_image = goal_image.reshape([n_blocks, env.env.camera_height, env.env.camera_width])

        elif env.goal_type=='block':
            pass

        ## recompute reward  ##
        reward_recompute, done_recompute, block_success_recompute = env.get_reward(_info)
        if _info['out_of_range']:
            if env.seperate:
                reward_recompute = [-1.] * env.num_blocks
            else:
                reward_recompute = -1.
        transitions.append([reward_recompute, goal_image, done_recompute, block_success_recompute])

    return transitions


## FCDQN Loss ##
def calculate_loss_fcdqn(minibatch, FCQ, FCQ_target, gamma=0.5):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state = torch.cat((state_im, goal_im), 1)
    next_state = torch.cat((next_state_im, goal_im), 1)

    next_q = FCQ_target(next_state)
    next_q_max = next_q.max(1)[0].max(1)[0].max(1)[0]
    #next_q_max = next_q[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]].max(1, True)[0]
    y_target = rewards + gamma * not_done * next_q_max

    q_values = FCQ(state)
    pred = q_values[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error

def calculate_loss_double_fcdqn(minibatch, FCQ, FCQ_target, gamma=0.5):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state = torch.cat((state_im, goal_im), 1)
    next_state = torch.cat((next_state_im, goal_im), 1)

    def get_a_prime_pixel():
        next_q = FCQ(next_state)
        next_q_chosen = next_q[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]]
        _, a_prime = next_q_chosen.max(1, True)
        return a_prime

    def get_a_prime():
        next_q = FCQ(next_state)
        aidx_x = next_q.max(1)[0].max(2)[0].max(1)[1]
        aidx_y = next_q.max(1)[0].max(1)[0].max(1)[1]
        aidx_th = next_q.max(2)[0].max(2)[0].max(1)[1]
        return aidx_th, aidx_x, aidx_y

    a_prime = get_a_prime()

    next_q_target = FCQ_target(next_state)
    q_target_s_a_prime = next_q_target[torch.arange(batch_size), a_prime[0], a_prime[1], a_prime[2]].unsqueeze(1)
    #next_q_target_chosen = next_q_target[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]]
    #q_target_s_a_prime = next_q_target_chosen.gather(1, a_prime)
    y_target = rewards + gamma * not_done * q_target_s_a_prime

    q_values = FCQ(state)
    pred = q_values[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error


## Seperate FCDQN Loss ##
def calculate_loss_seperate(minibatch, FCQ, FCQ_target, n_blocks, gamma=0.5):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state = torch.cat((state_im, goal_im), 1)
    next_state = torch.cat((next_state_im, goal_im), 1)

    next_q = FCQ_target(next_state)
    q_values = FCQ(state)

    loss = []
    error = []
    for o in range(n_blocks):
        next_q_max = next_q.max(1)[0].max(1)[0].max(1)[0]
        #next_q_max = next_q[torch.arange(batch_size), o, :, actions[:, 0], actions[:, 1]].max(1, True)[0]
        y_target = rewards[:, o].unsqueeze(1) + gamma * not_done * next_q_max

        pred = q_values[torch.arange(batch_size), o, actions[:, 2], actions[:, 0], actions[:, 1]]
        pred = pred.view(-1, 1)

        loss.append(criterion(y_target, pred))
        error.append(torch.abs(pred - y_target))

    loss = torch.sum(torch.stack(loss))
    error = torch.sum(torch.stack(error), dim=0)
    return loss, error

def calculate_loss_double_seperate(minibatch, FCQ, FCQ_target, n_blocks, gamma=0.5):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state = torch.cat((state_im, goal_im), 1)
    next_state = torch.cat((next_state_im, goal_im), 1)

    next_q_target = FCQ_target(next_state, True)
    q_values = FCQ(state, True)
    next_q = FCQ(next_state, True)

    def get_a_prime_pixel(obj):
        next_q_chosen = next_q[torch.arange(batch_size), obj, :, actions[:, 0], actions[:, 1]]
        _, a_prime = next_q_chosen.max(1, True)
        return a_prime

    def get_a_prime(obj):
        next_q_obj = next_q[:, obj]
        aidx_x = next_q_obj.max(1)[0].max(2)[0].max(1)[1]
        aidx_y = next_q_obj.max(1)[0].max(1)[0].max(1)[1]
        aidx_th = next_q_obj.max(2)[0].max(2)[0].max(1)[1]
        return aidx_th, aidx_x, aidx_y

    loss = []
    error = []
    for o in range(n_blocks):
        a_prime = get_a_prime(o)
        q_target_s_a_prime = next_q_target[torch.arange(batch_size), o, a_prime[0], a_prime[1], a_prime[2]].unsqueeze(1)
        #next_q_target_chosen = next_q_target[torch.arange(batch_size), o, :, actions[:, 0], actions[:, 1]]
        #q_target_s_a_prime = next_q_target_chosen.gather(1, a_prime)
        y_target = rewards[:, o].unsqueeze(1) + gamma * not_done * q_target_s_a_prime

        pred = q_values[torch.arange(batch_size), o, actions[:, 2], actions[:, 0], actions[:, 1]]
        pred = pred.view(-1, 1)

        loss.append(criterion(y_target, pred))
        error.append(torch.abs(pred - y_target))

    loss = torch.sum(torch.stack(loss))
    error = torch.sum(torch.stack(error), dim=0)
    return loss, error

## Constrained Seperate FCDQN ##
def calculate_loss_constrained(minibatch, FCQ, FCQ_target, n_blocks, gamma=0.5):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state = torch.cat((state_im, goal_im), 1)
    next_state = torch.cat((next_state_im, goal_im), 1)

    next_q = FCQ_target(next_state, True)
    q_values = FCQ(state, True)

    loss = []
    error = []
    for o in range(n_blocks):
        next_q_max = next_q.max(1)[0].max(1)[0].max(1)[0]
        #next_q_max = next_q[torch.arange(batch_size), 0, o, :, actions[:, 0], actions[:, 1]].max(1, True)[0]
        y_target = rewards[:, o].unsqueeze(1) + gamma * not_done * next_q_max

        pred = q_values[torch.arange(batch_size), 0, o, actions[:, 2], actions[:, 0], actions[:, 1]]
        pred = pred.view(-1, 1)

        loss.append(criterion(y_target, pred))
        error.append(torch.abs(pred - y_target))

    loss = torch.sum(torch.stack(loss))
    error = torch.sum(torch.stack(error), dim=0).view(-1)
    return loss, error

def calculate_loss_double_constrained(minibatch, FCQ, FCQ_target, n_blocks, gamma=0.5):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state = torch.cat((state_im, goal_im), 1)
    next_state = torch.cat((next_state_im, goal_im), 1)

    next_q_target = FCQ_target(next_state, True)
    q_values = FCQ(state, True)
    next_q = FCQ(next_state, True)

    def get_a_prime(obj):
        next_q_chosen = next_q[torch.arange(batch_size), 0, obj, :, actions[:, 0], actions[:, 1]]
        _, a_prime = next_q_chosen.max(1, True)
        return a_prime

    loss = []
    error = []
    for o in range(n_blocks):
        a_prime = get_a_prime(o)
        next_q_target_chosen = next_q_target[torch.arange(batch_size), 0, o, :, actions[:, 0], actions[:, 1]]
        q_target_s_a_prime = next_q_target_chosen.gather(1, a_prime)
        y_target = rewards[:, o].unsqueeze(1) + gamma * not_done * q_target_s_a_prime

        pred = q_values[torch.arange(batch_size), 0, o, actions[:, 2], actions[:, 0], actions[:, 1]]
        pred = pred.view(-1, 1)

        loss.append(criterion(y_target, pred))
        error.append(torch.abs(pred - y_target))

    loss = torch.sum(torch.stack(loss))
    error = torch.sum(torch.stack(error), dim=0).view(-1)
    return loss, error

def calculate_loss_next_v(minibatch, FCQ, FCQ_target, n_blocks):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    # rewards = minibatch[3]
    # not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state = torch.cat((state_im, goal_im), 1)
    next_state = torch.cat((next_state_im, goal_im), 1)

    next_q = FCQ_target(next_state, True) # bs x 2 x nb x 8 x h x w
    q_values = FCQ(state, True)

    loss = []
    error = []
    for o in range(n_blocks):
        y_target = next_q[torch.arange(batch_size), 0, o].mean([1,2,3])
        pred = q_values[torch.arange(batch_size), 1, o, actions[:, 2], actions[:, 0], actions[:, 1]]

        loss.append(criterion(y_target, pred))
        error.append(torch.abs(pred - y_target))

    loss = torch.sum(torch.stack(loss))
    error = torch.sum(torch.stack(error), dim=0)
    return loss, error

def calculate_loss_next_q(minibatch, FCQ, FCQ_target, n_blocks):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    # rewards = minibatch[3]
    # not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state = torch.cat((state_im, goal_im), 1)
    next_state = torch.cat((next_state_im, goal_im), 1)

    next_q = FCQ_target(next_state, True) # bs x 2 x nb x 8 x h x w
    q_values = FCQ(state, True)

    loss = []
    error = []
    for o in range(n_blocks):
        y_target = next_q[torch.arange(batch_size), 0, o].max(1)[0].max(1)[0].max(1)[0]
        pred = q_values[torch.arange(batch_size), 1, o, actions[:, 2], actions[:, 0], actions[:, 1]]

        loss.append(criterion(y_target, pred))
        error.append(torch.abs(pred - y_target))

    loss = torch.sum(torch.stack(loss))
    error = torch.sum(torch.stack(error), dim=0)
    return loss, error


## Cascade FCDQN v1 (step-by-step) ##
def calculate_loss_cascade_v1(minibatch, FCQ, FCQ_target, gamma=0.5):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state = torch.cat((state_im, goal_im), 1)
    next_state = torch.cat((next_state_im, goal_im), 1)

    next_q = FCQ_target(next_state, True)
    next_q_max = next_q.max(1)[0].max(1)[0].max(1)[0]
    #next_q_max = next_q[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]].max(1, True)[0]
    y_target = rewards[:,0].unsqueeze(1) + gamma * not_done * next_q_max

    q_values = FCQ(state, True)
    pred = q_values[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error

def calculate_loss_double_cascade_v1(minibatch, FCQ, FCQ_target, gamma=0.5):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state = torch.cat((state_im, goal_im), 1)
    next_state = torch.cat((next_state_im, goal_im), 1)

    next_q = FCQ(next_state, True)
    next_q_chosen = next_q[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]]
    _, a_prime = next_q_chosen.max(1, True)

    next_q_target = FCQ_target(next_state, True)
    next_q_target_chosen = next_q_target[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]]
    q_target_s_a_prime = next_q_target_chosen.gather(1, a_prime)
    y_target = rewards[:,0].unsqueeze(1) + gamma * not_done * q_target_s_a_prime

    q_values = FCQ(state, True)
    pred = q_values[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error

def calculate_cascade_loss_cascade_v1(minibatch, FCQ, CQN, CQN_target, gamma=0.5, add=False):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state = torch.cat((state_im, goal_im), 1)
    q1_map = FCQ(state, True)

    state_withq = torch.cat((state_im, goal_im, q1_map), 1)
    next_state_withq = torch.cat((next_state_im, goal_im, q1_map), 1)
    next_q2 = CQN_target(next_state_withq, True)
    q2_values = CQN(state_withq, True)

    if add:
        rewards = rewards[:,0] + rewards[:,1]
    else:
        rewards = rewards[:,1]

    next_q2_max = next_q2.max(1)[0].max(1)[0].max(1)[0]
    #next_q2_max = next_q2[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]].max(1, True)[0]
    y_target = rewards.unsqueeze(1) + gamma * not_done * next_q2_max

    pred = q2_values[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error

def calculate_cascade_loss_double_cascade_v1(minibatch, FCQ, CQN, CQN_target, gamma=0.5, add=False):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state = torch.cat((state_im, goal_im), 1)
    q1_map = FCQ(state, True)
    next_state = torch.cat((next_state_im, goal_im), 1)
    next_q1_map = FCQ(next_state, True)

    state_withq = torch.cat((state_im, goal_im, q1_map), 1)
    next_state_withq = torch.cat((next_state_im, goal_im, q1_map), 1)
    next_q2_target = CQN_target(next_state_withq, True)
    q2_values = CQN(state_withq, True)

    def get_a_prime():
        next_q2 = CQN(next_state_withq, True)
        next_q2_chosen = next_q2[torch.arange(batch_size), :, actions[:,0], actions[:,1]]
        if add:
            next_q1_chosen = next_q1_map[torch.arange(batch_size), :, actions[:,0], actions[:,1]]
            next_q_chosen = next_q1_chosen + next_q2_chosen
            _, a_prime = next_q_chosen.max(1, True)
        else:
            _, a_prime = next_q2_chosen.max(1, True)
        return a_prime

    loss = []
    error = []
    if add:
        rewards = rewards[:, 0] + rewards[:, 1]
    else:
        rewards = rewards[:, 1]

    a_prime = get_a_prime()
    next_q2_target_chosen = next_q2_target[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]]
    q2_target_s_a_prime = next_q2_target_chosen.gather(1, a_prime)
    y_target = rewards.unsqueeze(1) + gamma * not_done * q2_target_s_a_prime

    pred = q2_values[torch.arange(batch_size), actions[:,2], actions[:,0], actions[:,1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error


## Cascade FCDQN v2 (end-to-end) ##
def calculate_loss_cascade_v2(minibatch, FCQ, FCQ_target, n_blocks, gamma=0.5):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state = torch.cat((state_im, goal_im), 1)
    next_state = torch.cat((next_state_im, goal_im), 1)

    next_q = FCQ_target(next_state)
    q_values = FCQ(state)

    loss = []
    error = []
    for o in range(n_blocks):
        next_q_max = next_q.max(1)[0].max(1)[0].max(1)[0]
        #next_q_max = next_q[torch.arange(batch_size), o, :, actions[:, 0], actions[:, 1]].max(1, True)[0]
        y_target = rewards[:, o].unsqueeze(1) + gamma * not_done * next_q_max

        pred = q_values[torch.arange(batch_size), o, actions[:, 2], actions[:, 0], actions[:, 1]]
        pred = pred.view(-1, 1)

        loss.append(criterion(y_target, pred))
        error.append(torch.abs(pred - y_target))

    loss = torch.sum(torch.stack(loss))
    error = torch.sum(torch.stack(error), dim=0)
    return loss, error

def calculate_loss_double_cascade_v2(minibatch, FCQ, FCQ_target, n_blocks, gamma=0.5):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state = torch.cat((state_im, goal_im), 1)
    next_state = torch.cat((next_state_im, goal_im), 1)

    next_q_target = FCQ_target(next_state)
    q_values = FCQ(state)
    next_q = FCQ(next_state)

    def get_a_prime(obj):
        next_q_obj = next_q[:, obj]
        aidx_x = next_q_obj.max(1)[0].max(2)[0].max(1)[1]
        aidx_y = next_q_obj.max(1)[0].max(1)[0].max(1)[1]
        aidx_th = next_q_obj.max(2)[0].max(2)[0].max(1)[1]
        return aidx_th, aidx_x, aidx_y

    loss = []
    error = []
    for o in range(n_blocks):
        a_prime = get_a_prime(o)
        q_target_s_a_prime = next_q_target[torch.arange(batch_size), o, a_prime[0], a_prime[1], a_prime[2]].unsqueeze(1)
        y_target = rewards[:, o].unsqueeze(1) + gamma * not_done * q_target_s_a_prime

        pred = q_values[torch.arange(batch_size), o, actions[:, 2], actions[:, 0], actions[:, 1]]
        pred = pred.view(-1, 1)

        loss.append(criterion(y_target, pred))
        error.append(torch.abs(pred - y_target))

    loss = torch.sum(torch.stack(loss))
    error = torch.sum(torch.stack(error), dim=0)
    return loss, error


## Cascade FCDQN v3 (pre-trained) ##
def normalize_q(q_map):
    q_mean = q_map.mean([-3,-2,-1], True)
    return (q_map - q_mean) / q_mean


def calculate_cascade_loss_cascade_v3(minibatch, FCQ, CQN, CQN_target, goal_type, gamma=0.5, output='', normalize=False):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    if goal_type=='pixel':
        state_goal = torch.cat((state_im, goal_im[:, 0:1]), 1)
    else:
        state_goal = torch.cat((state_im, goal_im), 1)
    q1_values = FCQ(state_goal, True)
    if normalize:
        q1_values = normalize_q(q1_values)
    state_goal_q = torch.cat((state_im, goal_im, q1_values), 1)
    q2_values = CQN(state_goal_q, True)

    if goal_type=='pixel':
        next_state_goal = torch.cat((next_state_im, goal_im[:, 0:1]), 1)
    else:
        next_state_goal = torch.cat((next_state_im, goal_im), 1)
    next_q1_values = FCQ(next_state_goal, True)
    if normalize:
        next_q1_values = normalize_q(next_q1_values)
    next_state_goal_q = torch.cat((next_state_im, goal_im, next_q1_values), 1)
    next_q2_targets = CQN_target(next_state_goal_q, True)

    next_q2_max = next_q2_targets.max(1)[0].max(1)[0].max(1)[0]
    next_qsum_max = (next_q1_values + next_q2_targets).max(1)[0].max(1)[0].max(1)[0]
    #next_q2_max = next_q2_targets[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]].max(1, True)[0]
    #next_qsum_max = (next_q1_values + next_q2_targets)[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]].max(1, True)[0]
    if output=='':
        rewards = rewards[:,1]
        y_target = rewards.unsqueeze(1) + gamma * not_done * next_q2_max
        pred = q2_values[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]
    elif output=='addR':
        rewards = rewards[:,0] + rewards[:,1]
        y_target = rewards.unsqueeze(1) + gamma * not_done * next_q2_max
        pred = q2_values[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]
    elif output=='addQ':
        rewards = rewards[:,0] + rewards[:,1]
        y_target = rewards.unsqueeze(1) + gamma * not_done * next_qsum_max
        pred = (q1_values+q2_values)[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error

def calculate_cascade_loss_double_cascade_v3(minibatch, FCQ, CQN, CQN_target, goal_type, gamma=0.5, output='', normalize=False):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    if goal_type=='pixel':
        state_goal = torch.cat((state_im, goal_im[:, 0:1]), 1)
    else:
        state_goal = torch.cat((state_im, goal_im), 1)
    q1_values = FCQ(state_goal, True)
    if normalize:
        q1_values = normalize_q(q1_values)
    state_goal_q = torch.cat((state_im, goal_im, q1_values), 1)
    q2_values = CQN(state_goal_q, True)

    if goal_type=='pixel':
        next_state_goal = torch.cat((next_state_im, goal_im[:, 0:1]), 1)
    else:
        next_state_goal = torch.cat((next_state_im, goal_im), 1)
    next_q1_values = FCQ(next_state_goal, True)
    if normalize:
        next_q1_values = normalize_q(next_q1_values)
    next_state_goal_q = torch.cat((next_state_im, goal_im, next_q1_values), 1)
    next_q2_targets = CQN_target(next_state_goal_q, True)

    def get_a_prime():
        next_q2 = CQN(next_state_goal_q, True)
        if output == '' or output == 'addR':
            next_q = next_q2
        elif output == 'addQ':
            next_q = next_q1_values + next_q2
        aidx_x = next_q.max(1)[0].max(2)[0].max(1)[1]
        aidx_y = next_q.max(1)[0].max(1)[0].max(1)[1]
        aidx_th = next_q.max(2)[0].max(2)[0].max(1)[1]
        return aidx_th, aidx_x, aidx_y

    if output=='':
        rewards = rewards[:, 1]
        a_prime = get_a_prime()
        q2_target_s_a_prime = next_q2_targets[torch.arange(batch_size), a_prime[0], a_prime[1], a_prime[2]].unsqueeze(1)
        # next_q2_target_chosen = next_q2_targets[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]]
        # q2_target_s_a_prime = next_q2_target_chosen.gather(1, a_prime)
        y_target = rewards.unsqueeze(1) + gamma * not_done * q2_target_s_a_prime
        pred = q2_values[torch.arange(batch_size), actions[:,2], actions[:,0], actions[:,1]]

    elif output=='addR':
        rewards = rewards[:, 0] + rewards[:, 1]
        a_prime = get_a_prime()
        q2_target_s_a_prime = next_q2_targets[torch.arange(batch_size), a_prime[0], a_prime[1], a_prime[2]].unsqueeze(1)
        # next_q2_target_chosen = next_q2_targets[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]]
        # q2_target_s_a_prime = next_q2_target_chosen.gather(1, a_prime)
        y_target = rewards.unsqueeze(1) + gamma * not_done * q2_target_s_a_prime
        pred = q2_values[torch.arange(batch_size), actions[:,2], actions[:,0], actions[:,1]]
        
    elif output=='addQ':
        rewards = rewards[:, 0] + rewards[:, 1]
        a_prime = get_a_prime()
        next_qsum = next_q1_values + next_q2_targets
        qsum_target_s_a_prime = next_qsum[torch.arange(batch_size), a_prime[0], a_prime[1], a_prime[2]].unsqueeze(1)
        # next_qsum_target_chosen = next_qsum[torch.arange(batch_size), :, actions[:, 0], actions[:, 1]]
        # qsum_target_s_a_prime = next_qsum_target_chosen.gather(1, a_prime)
        y_target = rewards.unsqueeze(1) + gamma * not_done * qsum_target_s_a_prime
        pred = (q1_values+q2_values)[torch.arange(batch_size), actions[:,2], actions[:,0], actions[:,1]]

    pred = pred.view(-1, 1)
    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error


## Seperate Cascade FCDQN ##
def calculate_cascade_loss_sppcqn(minibatch, FCQ, CQN, CQN_target, goal_type, gamma=0.5):
    n_blocks = 2
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    if goal_type=='pixel':
        state_goal = torch.cat((state_im, goal_im[:, 0:1]), 1)
    else:
        state_goal = torch.cat((state_im, goal_im), 1)
    q1_values = FCQ(state_goal, True)
    state_goal_q = torch.cat((state_im, goal_im, q1_values), 1)
    q2_values = CQN(state_goal_q, True)

    if goal_type=='pixel':
        next_state_goal = torch.cat((next_state_im, goal_im[:, 0:1]), 1)
    else:
        next_state_goal = torch.cat((next_state_im, goal_im), 1)
    next_q1_values = FCQ(next_state_goal, True)
    next_state_goal_q = torch.cat((next_state_im, goal_im, next_q1_values), 1)
    next_q2_targets = CQN_target(next_state_goal_q, True)

    loss = []
    error = []
    for o in range(n_blocks):
        next_q2_max = next_q2_targets.max(1)[0].max(1)[0].max(1)[0]
        y_target = rewards[:, o].unsqueeze(1) + gamma * not_done * next_q2_max

        pred = q2_values[torch.arange(batch_size), o, actions[:, 2], actions[:, 0], actions[:, 1]]
        pred = pred.view(-1, 1)

        loss.append(criterion(y_target, pred))
        error.append(torch.abs(pred - y_target))

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error

def calculate_cascade_loss_double_sppcqn(minibatch, FCQ, CQN, CQN_target, goal_type, gamma=0.5):
    n_blocks = 2
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    if goal_type=='pixel':
        state_goal = torch.cat((state_im, goal_im[:, 0:1]), 1)
    else:
        state_goal = torch.cat((state_im, goal_im), 1)
    q1_values = FCQ(state_goal, True)
    state_goal_q = torch.cat((state_im, goal_im, q1_values), 1)
    q2_values = CQN(state_goal_q, True)

    if goal_type=='pixel':
        next_state_goal = torch.cat((next_state_im, goal_im[:, 0:1]), 1)
    else:
        next_state_goal = torch.cat((next_state_im, goal_im), 1)
    next_q1_values = FCQ(next_state_goal, True)
    next_state_goal_q = torch.cat((next_state_im, goal_im, next_q1_values), 1)
    next_q2_targets = CQN_target(next_state_goal_q, True)
    next_q2 = CQN(next_state_goal_q, True)

    def get_a_prime(obj):
        next_q2_obj = next_q2[:, obj]
        aidx_x = next_q2_obj.max(1)[0].max(2)[0].max(1)[1]
        aidx_y = next_q2_obj.max(1)[0].max(1)[0].max(1)[1]
        aidx_th = next_q2_obj.max(2)[0].max(2)[0].max(1)[1]
        return aidx_th, aidx_x, aidx_y

    loss = []
    error = []
    for o in range(n_blocks):
        a_prime = get_a_prime(o)
        q2_target_s_a_prime = next_q2_targets[torch.arange(batch_size), o, a_prime[0], a_prime[1], a_prime[2]].unsqueeze(1)
        y_target = rewards[:, o].unsqueeze(1) + gamma * not_done * q2_target_s_a_prime

        pred = q2_values[torch.arange(batch_size), o, actions[:, 2], actions[:, 0], actions[:, 1]]
        pred = pred.view(-1, 1)

        loss.append(criterion(y_target, pred))
        error.append(torch.abs(pred - y_target))

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error


## Cascade FCDQN 3blocks (pre-trained) ##
def calculate_cascade_loss_cascade_3blocks(minibatch, FCQ, FCQ2, FCQ3, FCQ3_target, gamma=0.5, output=''):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state_goal = torch.cat((state_im, goal_im), 1)
    q1_values = FCQ(state_goal, True)

    state_goal_q1 = torch.cat((state_im, goal_im, q1_values), 1)
    q2_values = FCQ2(state_goal_q1, True)

    state_goal_q2 = torch.cat((state_im, goal_im, q2_values), 1)
    q3_values = FCQ3(state_goal_q2, True)

    next_state_goal = torch.cat((next_state_im, goal_im), 1)
    next_q1_values = FCQ(next_state_goal, True)

    next_state_goal_q1 = torch.cat((next_state_im, goal_im, next_q1_values), 1)
    next_q2_values = FCQ2(next_state_goal_q1, True)

    next_state_goal_q2 = torch.cat((next_state_im, goal_im, next_q2_values), 1)
    next_q3_targets = FCQ3_target(next_state_goal_q2, True)

    next_q3_max = next_q3_targets.max(1)[0].max(1)[0].max(1)[0]
    next_qsum_max = (next_q1_values + next_q2_values + next_q3_targets).max(1)[0].max(1)[0].max(1)[0]

    if output == '':
        rewards = rewards[:, 2]
        y_target = rewards.unsqueeze(1) + gamma * not_done * next_q3_max
        pred = q3_values[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]
    elif output == 'addR':
        rewards = rewards[:, 0] + rewards[:, 1] + rewards[:, 2]
        y_target = rewards.unsqueeze(1) + gamma * not_done * next_q3_max
        pred = q3_values[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]
    elif output == 'addQ':
        rewards = rewards[:, 0] + rewards[:, 1] + rewards[:, 2]
        y_target = rewards.unsqueeze(1) + gamma * not_done * next_qsum_max
        pred = (q1_values + q2_values + q3_values)[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error


def calculate_cascade_loss_double_cascade_3blocks(minibatch, FCQ, FCQ2, FCQ3, FCQ3_target, gamma=0.5, output=''):
    state_im = minibatch[0]
    next_state_im = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal_im = minibatch[5]
    batch_size = state_im.size()[0]

    state_goal = torch.cat((state_im, goal_im), 1)
    q1_values = FCQ(state_goal, True)

    state_goal_q1 = torch.cat((state_im, goal_im, q1_values), 1)
    q2_values = FCQ2(state_goal_q1, True)

    state_goal_q2 = torch.cat((state_im, goal_im, q2_values), 1)
    q3_values = FCQ3(state_goal_q2, True)

    next_state_goal = torch.cat((next_state_im, goal_im), 1)
    next_q1_values = FCQ(next_state_goal, True)

    next_state_goal_q1 = torch.cat((next_state_im, goal_im, next_q1_values), 1)
    next_q2_values = FCQ2(next_state_goal_q1, True)

    next_state_goal_q2 = torch.cat((next_state_im, goal_im, next_q2_values), 1)
    next_q3_targets = FCQ3_target(next_state_goal_q2, True)

    def get_a_prime():
        next_q3 = FCQ3(next_state_goal_q2, True)
        if output == '' or output == 'addR':
            next_q = next_q3
        elif output == 'addQ':
            next_q = next_q1_values + next_q2_values + next_q3
        aidx_x = next_q.max(1)[0].max(2)[0].max(1)[1]
        aidx_y = next_q.max(1)[0].max(1)[0].max(1)[1]
        aidx_th = next_q.max(2)[0].max(2)[0].max(1)[1]
        return aidx_th, aidx_x, aidx_y

    if output == '':
        rewards = rewards[:, 2]
        a_prime = get_a_prime()
        q3_target_s_a_prime = next_q3_targets[torch.arange(batch_size), a_prime[0], a_prime[1], a_prime[2]].unsqueeze(1)

        y_target = rewards.unsqueeze(1) + gamma * not_done * q3_target_s_a_prime
        pred = q3_values[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]

    elif output == 'addR':
        rewards = rewards[:, 0] + rewards[:, 1] + rewards[:, 2]
        a_prime = get_a_prime()
        q3_target_s_a_prime = next_q3_targets[torch.arange(batch_size), a_prime[0], a_prime[1], a_prime[2]].unsqueeze(1)

        y_target = rewards.unsqueeze(1) + gamma * not_done * q3_target_s_a_prime
        pred = q3_values[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]

    elif output == 'addQ':
        rewards = rewards[:, 0] + rewards[:, 1] + rewards[:, 2]
        a_prime = get_a_prime()
        next_qsum = next_q1_values + next_q2_values + next_q3_targets
        qsum_target_s_a_prime = next_qsum[torch.arange(batch_size), a_prime[0], a_prime[1], a_prime[2]].unsqueeze(1)

        y_target = rewards.unsqueeze(1) + gamma * not_done * qsum_target_s_a_prime
        pred = (q1_values + q2_values + q3_values)[torch.arange(batch_size), actions[:, 2], actions[:, 0], actions[:, 1]]

    pred = pred.view(-1, 1)
    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error
