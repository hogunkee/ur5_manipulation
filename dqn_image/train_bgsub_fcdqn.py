import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../ur5_mujoco'))
from mrcnn_env import *

from backgroundsubtraction_module import BackgroundSubtraction
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

crop_min = 9 #19 #11 #13
crop_max = 88 #78 #54 #52


def get_state_goal(env, segmodule, state, target_color=None):
    if env.goal_type=='pixel':
        image = state[0] * 255
        goal_image = state[1]

        masks, colors, fm = segmodule.get_masks(image, env.num_blocks)
        if len(masks)<=2:
            print('no masks!!')
            masks, colors, fm = segmodule.get_masks(image, env.num_blocks, sub=True)
            if len(masks) == 0:
                return None, None
        if target_color is not None:
            t_obj = np.argmin(np.linalg.norm(colors - target_color, axis=1))
        else:
            t_obj = np.random.randint(len(masks))
        target_seg = masks[t_obj]
        obstacle_seg = np.any([masks[o] for o in range(len(masks)) if o != t_obj], 0)
        workspace_seg = segmodule.workspace_seg
        target_color = colors[t_obj]

        env.set_target_with_color(target_color)
        target_idx = env.seg_target

        state = np.concatenate([target_seg, obstacle_seg, workspace_seg]).reshape(-1, 96, 96)
        goal = goal_image[target_idx: target_idx+1]
    return [state, goal], target_color

def get_action(env, fc_qnet, state, epsilon, pre_action=None, with_q=False):
    if np.random.random() < epsilon:
        action = [np.random.randint(crop_min,crop_max), np.random.randint(crop_min,crop_max), np.random.randint(env.num_bins)]
        # action = [np.random.randint(env.env.camera_height), np.random.randint(env.env.camera_width), np.random.randint(env.num_bins)]
        if with_q:
            state_im = torch.tensor([state[0]]).type(dtype)
            goal_im = torch.tensor([state[1]]).type(dtype)
            state_goal = torch.cat((state_im, goal_im), 1)
            q_value = fc_qnet(state_goal)
            q_raw = q_value[0].detach().cpu().numpy()
            q = np.zeros_like(q_raw)
            q[:, crop_min:crop_max, crop_min:crop_max] = q_raw[:, crop_min:crop_max, crop_min:crop_max]
    else:
        state_im = torch.tensor([state[0]]).type(dtype)
        goal_im = torch.tensor([state[1]]).type(dtype)
        state_goal = torch.cat((state_im, goal_im), 1)
        q_value = fc_qnet(state_goal)
        q_raw = q_value[0].detach().cpu().numpy()
        q = np.zeros_like(q_raw)
        q[:, crop_min:crop_max, crop_min:crop_max] = q_raw[:, crop_min:crop_max, crop_min:crop_max]
        # avoid redundant motion #
        if pre_action is not None:
            q[pre_action[2], pre_action[0], pre_action[1]] = q.min()
        # image coordinate #
        aidx_x = q.max(0).max(1).argmax()
        aidx_y = q.max(0).max(0).argmax()
        aidx_th = q.argmax(0)[aidx_x, aidx_y]
        action = [aidx_x, aidx_y, aidx_th]

    if with_q:
        return action, q
    else:
        return action

def learning(env, 
        seg,
        savename,
        n_actions=8,
        in_channel=6,
        learning_rate=1e-4, 
        batch_size=64, 
        buff_size=1e4, 
        total_steps=1e6,
        learn_start=1e4,
        update_freq=100,
        log_freq=1e3,
        double=True,
        per=True,
        her=True,
        visualize_q=False,
        pre_train=False,
        continue_learning=False,
        model_path=''
        ):

    FCQ = FC_QNet(n_actions, in_channel).type(dtype)
    if pre_train:
        FCQ.load_state_dict(torch.load(model_path))
        print('Loading pre-trained model: {}'.format(model_path))
    elif continue_learning:
        FCQ.load_state_dict(torch.load(model_path))
        print('Loading trained model: {}'.format(model_path))
    FCQ_target = FC_QNet(n_actions, in_channel).type(dtype)
    FCQ_target.load_state_dict(FCQ.state_dict())

    # criterion = nn.SmoothL1Loss(reduction=None).type(dtype)
    # criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(FCQ.parameters(), lr=learning_rate, momentum=0.9, weight_decay=2e-5)
    # optimizer = torch.optim.Adam(FCQ.parameters(), lr=learning_rate)

    goal_ch = 1
    if per:
        replay_buffer = PER([3, env.env.camera_height, env.env.camera_width], \
                    [goal_ch, env.env.camera_height, env.env.camera_width], 1, \
                    save_goal=True, save_gripper=False, max_size=int(buff_size))
    else:
        replay_buffer = ReplayBuffer([3, env.env.camera_height, env.env.camera_width], \
                    [goal_ch, env.env.camera_height, env.env.camera_width], 1, \
                    save_goal=True, save_gripper=False, max_size=int(buff_size))

    model_parameters = filter(lambda p: p.requires_grad, FCQ.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("# of params: %d"%params)


    if double:
        calculate_loss = calculate_loss_double_fcdqn
    else:
        calculate_loss = calculate_loss_fcdqn

    if continue_learning and not pre_train:
        numpy_log = np.load(model_path.replace('models/', 'board/').replace('.pth', '.npy'), allow_pickle=True)
        log_returns = list(numpy_log[0])
        log_loss = list(numpy_log[1])
        log_eplen = list(numpy_log[2])
        log_epsilon = list(numpy_log[3])
        log_success = list(numpy_log[4])
        log_collisions = list(numpy_log[5])
        log_out = list(numpy_log[6])
        log_success_block = list(numpy_log[7])
        log_target = list(numpy_log[8])
    else:
        log_returns = []
        log_loss = []
        log_eplen = []
        log_epsilon = []
        log_success = []
        log_collisions = []
        log_out = []
        log_success_block = [[], [], []]
        log_mean_success_block = [[], [], []]
        log_target = []
    log_minibatchloss = []

    if not os.path.exists("results/graph/"):
        os.makedirs("results/graph/")
    if not os.path.exists("results/models/"):
        os.makedirs("results/models/")
    if not os.path.exists("results/board/"):
        os.makedirs("results/board/")

    #plt.ion()
    plt.show(block=False)
    plt.rc('axes', labelsize=6)
    plt.rc('font', size=6)
    f, axes = plt.subplots(3, 3) # 3,2
    f.set_figheight(12) #9 #15
    f.set_figwidth(20) #12 #10

    axes[0][0].set_title('Block 1 success')  # 1
    axes[0][0].set_ylim([0, 1])
    axes[0][1].set_title('Block 2 success')  # 2
    axes[0][1].set_ylim([0, 1])
    axes[0][2].set_title('Block 3 success')  # 3
    axes[0][2].set_ylim([0, 1])
    axes[1][0].set_title('Success Rate')  # 4
    axes[1][0].set_ylim([0, 1])
    axes[1][1].set_title('Episode Return')  # 5
    axes[1][2].set_title('Loss')  # 6
    axes[2][0].set_title('Episode Length')  # 7
    axes[2][1].set_title('Out of Range')  # 8
    axes[2][1].set_ylim([0, 1])
    axes[2][2].set_title('Num Collisions')  # 9

    lr_decay = 0.98
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)

    if len(log_epsilon) == 0:
        epsilon = 0.5 #1.0
        start_epsilon = 0.5
    else:
        epsilon = log_epsilon[-1]
        start_epsilon = log_epsilon[-1]
    min_epsilon = 0.1
    epsilon_decay = 0.98
    episode_reward = 0.0
    max_success = 0.0
    ep_len = 0
    ne = 0
    t_step = 0
    num_collisions = 0

    state_raw = env.reset()
    state, target_color = get_state_goal(env, seg, state_raw, None)
    pre_action = None

    if visualize_q:
        fig = plt.figure()
        ax0 = fig.add_subplot(231)
        ax1 = fig.add_subplot(232)
        ax2 = fig.add_subplot(233)
        ax3 = fig.add_subplot(234)
        ax4 = fig.add_subplot(235)
        ax5 = fig.add_subplot(236)
        ax0.set_title('Goal')
        ax1.set_title('Observation')
        ax2.set_title('Q-map')
        ax3.set_title('Target')
        ax4.set_title('Obstacles')
        ax5.set_title('Background')
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax5.set_xticks([])
        ax5.set_yticks([])

        s0 = deepcopy(state[0]).transpose([1, 2, 0])
        s1 = deepcopy(state[1]).reshape(96, 96)
        ax0.imshow(s1)
        s0[s0[:, :, 0].astype(bool)] = [1, 0, 0]
        s0[s0[:, :, 1].astype(bool)] = [0, 1, 0]
        ax1.imshow(s0)
        ax2.imshow(np.zeros_like(s0))
        ax3.imshow(s0[:, :, 0])
        ax4.imshow(s0[:, :, 1])
        ax5.imshow(s0[:, :, 2])
        plt.show(block=False)
        fig.canvas.draw()
        fig.canvas.draw()

    while t_step < total_steps:
        action, q_map = get_action(env, FCQ, state, epsilon=epsilon, pre_action=pre_action, with_q=True)
        if visualize_q:
            s0 = deepcopy(state[0]).transpose([1, 2, 0])
            s1 = deepcopy(state[1]).reshape(96, 96)
            ax0.imshow(s1)
            ax3.imshow(s0[:, :, 0])
            ax4.imshow(s0[:, :, 1])
            ax5.imshow(s0[:, :, 2])

            s0[action[0], action[1]] = [1, 0, 0]
            s0[s0[:, :, 0].astype(bool)] = [1, 0, 0]
            s0[s0[:, :, 1].astype(bool)] = [0, 1, 0]
            ax1.imshow(s0)
            q_map = q_map.transpose([1,2,0]).max(2)
            ax2.imshow(q_map/q_map.max())
            #print('min_q:', q_map.min(), '/ max_q:', q_map.max())
            fig.canvas.draw()

        next_state_raw, reward, done, info = env.step(action)
        next_state, target_color = get_state_goal(env, seg, next_state_raw, target_color)
        episode_reward += reward

        ## save transition to the replay buffer ##
        if per:
            trajectories = []
            trajectories.append([[state[0], 0.0], action, [next_state[0], 0.0], reward, done, state[1]])

            replay_tensors = []
            traj_tensor = [
                torch.FloatTensor(state[0]).type(dtype),
                torch.FloatTensor(next_state[0]).type(dtype),
                torch.FloatTensor(action).type(dtype),
                torch.FloatTensor([reward]).type(dtype),
                torch.FloatTensor([1 - done]).type(dtype),
                torch.FloatTensor(state[1]).type(dtype),
            ]
            replay_tensors.append(traj_tensor)

            ## HER ##
            if her and not done:
                her_sample = sample_her_transitions(env, info, next_state)
                ig_samples = sample_ig_transitions(env, info, next_state, num_samples=3)
                samples = her_sample + ig_samples
                for sample in samples:
                    reward_re, goal_image, done_re, block_success_re = sample
                    if env.goal_type=='pixel':
                        state_re = [state[0], goal_image[env.seg_target: env.seg_target+1]]

                    traj_tensor = [
                        torch.FloatTensor(state_re[0]).type(dtype),
                        torch.FloatTensor(next_state[0]).type(dtype),
                        torch.FloatTensor(action).type(dtype),
                        torch.FloatTensor([reward_re]).type(dtype),
                        torch.FloatTensor([1 - done_re]).type(dtype),
                        torch.FloatTensor(state_re[1]).type(dtype),
                    ]
                    replay_tensors.append(traj_tensor)
                    trajectories.append([[state[0], 0.0], action, [next_state[0], 0.0], reward_re, done_re, state_re[1]])

            minibatch = None
            for data in replay_tensors:
                minibatch = combine_batch(minibatch, data)
            _, error = calculate_loss(minibatch, FCQ, FCQ_target)
            error = error.data.detach().cpu().numpy()
            for i, traj in enumerate(trajectories):
                replay_buffer.add(error[i], *traj)

        else:
            trajectories = []
            trajectories.append([[state[0], 0.0], action, [next_state[0], 0.0], reward, done, state[1]])

            ## HER ##
            if her and not done:
                her_sample = sample_her_transitions(env, info, next_state)
                ig_samples = sample_ig_transitions(env, info, next_state, num_samples=3)
                samples = her_sample + ig_samples
                for sample in samples:
                    reward_re, goal_image, done_re, block_success_re = sample
                    if env.goal_type=='pixel':
                        state_re = [state[0], goal_image[env.seg_target: env.seg_target+1]]
                    trajectories.append([[state[0], 0.0], action, [next_state[0], 0.0], reward_re, done_re, state_re[1]])

            for traj in trajectories:
                replay_buffer.add(*traj)

        if t_step < learn_start:
            if done:
                state_raw = env.reset()
                state, target_color = get_state_goal(env, seg, state_raw, None)
                pre_action = None
                episode_reward = 0.
            else:
                state = next_state
                pre_action = action
            learn_start -= 1
            if learn_start==0:
                epsilon = start_epsilon
            continue

        ## sample from replay buff & update networks ##
        data = [
                torch.FloatTensor(state[0]).type(dtype),
                torch.FloatTensor(next_state[0]).type(dtype),
                torch.FloatTensor(action).type(dtype),
                torch.FloatTensor([reward]).type(dtype),
                torch.FloatTensor([1 - done]).type(dtype),
                torch.FloatTensor(state[1]).type(dtype),
                ]
        if per:
            minibatch, idxs, is_weights = replay_buffer.sample(batch_size-1)
            combined_minibatch = combine_batch(minibatch, data)
            loss, error = calculate_loss(combined_minibatch, FCQ, FCQ_target)
            errors = error.data.detach().cpu().numpy()[:-1]
            # update priority
            for i in range(batch_size-1):
                idx = idxs[i]
                replay_buffer.update(idx, errors[i])
        else:
            minibatch = replay_buffer.sample(batch_size-1)
            combined_minibatch = combine_batch(minibatch, data)
            loss, _ = calculate_loss(combined_minibatch, FCQ, FCQ_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        log_minibatchloss.append(loss.data.detach().cpu().numpy())

        state = next_state
        pre_action = action
        ep_len += 1
        t_step += 1
        num_collisions += int(info['collision'])

        if done:
            ne += 1
            log_returns.append(episode_reward)
            log_loss.append(np.mean(log_minibatchloss))
            log_eplen.append(ep_len)
            log_epsilon.append(epsilon)
            log_out.append(int(info['out_of_range']))
            log_success.append(int(info['success']))
            log_collisions.append(num_collisions)

            log_target.append(env.seg_target)
            recent_target = np.array(log_target[-log_freq:])
            for o in range(3):
                log_success_block[o].append(int(info['block_success'][o]))
                recent_block_success = np.array(log_success_block[o])[-log_freq:][recent_target==o]
                log_mean_success_block[o].append(np.mean(recent_block_success))

            if ne % log_freq == 0:
                log_mean_returns = smoothing_log_same(log_returns, log_freq)
                log_mean_loss = smoothing_log_same(log_loss, log_freq)
                log_mean_eplen = smoothing_log_same(log_eplen, log_freq)
                log_mean_out = smoothing_log_same(log_out, log_freq)
                log_mean_success = smoothing_log_same(log_success, log_freq)
                log_mean_collisions = smoothing_log_same(log_collisions, log_freq)

                print()
                print("{} episodes. ({}/{} steps)".format(ne, t_step, total_steps))
                print("Success rate: {0:.2f}".format(log_mean_success[-1]))
                for o in range(3):
                    print("Block {0}: {1:.2f}".format(o+1, log_mean_success_block[o][-1]))
                print("Mean reward: {0:.2f}".format(log_mean_returns[-1]))
                print("Mean loss: {0:.6f}".format(log_mean_loss[-1]))
                print("Ep length: {}".format(log_mean_eplen[-1]))
                print("Epsilon: {}".format(epsilon))

                axes[1][2].plot(log_loss, color='#ff7f00', linewidth=0.5)  # 3->6
                axes[1][1].plot(log_returns, color='#60c7ff', linewidth=0.5)  # 5
                axes[2][0].plot(log_eplen, color='#83dcb7', linewidth=0.5)  # 7
                axes[2][2].plot(log_collisions, color='#ff33cc', linewidth=0.5)  # 8->9

                for o in range(3):
                    axes[0][o].plot(log_mean_success_block[o], color='red')  # 1,2,3

                axes[1][2].plot(log_mean_loss, color='red')  # 3->6
                axes[1][1].plot(log_mean_returns, color='blue')  # 5
                axes[2][0].plot(log_mean_eplen, color='green')  # 7
                axes[1][0].plot(log_mean_success, color='red')  # 4
                axes[2][1].plot(log_mean_out, color='black')  # 6->8
                axes[2][2].plot(log_mean_collisions, color='#663399')  # 8->9

                f.savefig('results/graph/%s.png' % savename)

                log_list = [
                        log_returns,  # 0
                        log_loss,  # 1
                        log_eplen,  # 2
                        log_epsilon,  # 3
                        log_success,  # 4
                        log_collisions,  # 5
                        log_out,  # 6
                        log_success_block, #7
                        log_target #8
                        ]
                numpy_log = np.array(log_list)
                np.save('results/board/%s' %savename, numpy_log)

                if log_mean_success[-1] > max_success:
                    max_success = log_mean_success[-1]
                    torch.save(FCQ.state_dict(), 'results/models/%s.pth' % savename)
                    print("Max performance! saving the model.")

            state_raw = env.reset()
            state, target_color = get_state_goal(env, seg, state_raw, None)

            episode_reward = 0.
            log_minibatchloss = []
            pre_action = None
            ep_len = 0
            num_collisions = 0

            if ne % update_freq == 0:
                FCQ_target.load_state_dict(FCQ.state_dict())
                lr_scheduler.step()
                epsilon = max(epsilon_decay * epsilon, min_epsilon)


    print('Training finished.')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--task", default=1, type=int)
    parser.add_argument("--num_blocks", default=3, type=int)
    parser.add_argument("--dist", default=0.08, type=float)
    parser.add_argument("--max_steps", default=30, type=int)
    parser.add_argument("--camera_height", default=96, type=int)
    parser.add_argument("--camera_width", default=96, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--bs", default=6, type=int)
    parser.add_argument("--buff_size", default=1e3, type=float)
    parser.add_argument("--total_steps", default=2e5, type=float)
    parser.add_argument("--learn_start", default=1e3, type=float)
    parser.add_argument("--update_freq", default=100, type=int)
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument("--double", action="store_false") # default: True
    parser.add_argument("--per", action="store_false") # default: True
    parser.add_argument("--her", action="store_false") # default: True
    parser.add_argument("--goal", default="pixel", type=str)
    parser.add_argument("--reward", default="new", type=str)
    parser.add_argument("--fcn_ver", default=1, type=int)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--resnet", action="store_false") # default: True
    parser.add_argument("--small", action="store_true") # default: False
    parser.add_argument("--pre_train", action="store_true")
    parser.add_argument("--continue_learning", action="store_true")
    parser.add_argument("--model_path", default="", type=str)
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
    task = 1 #args.task
    num_blocks = args.num_blocks
    mov_dist = args.dist
    max_steps = args.max_steps
    camera_height = args.camera_height
    camera_width = args.camera_width
    goal_type = args.goal
    reward_type = args.reward

    model_path = os.path.join("results/models/BGSB_%s.pth"%args.model_path)
    visualize_q = args.show_q
    if visualize_q:
        render = True

    now = datetime.datetime.now()
    savename = "BGSB_%s" % (now.strftime("%m%d_%H%M"))
    if not os.path.exists("results/config/"):
        os.makedirs("results/config/")
    with open("results/config/%s.json" % savename, 'w') as cf:
        json.dump(args.__dict__, cf, indent=2)

    backsub = BackgroundSubtraction()
    env = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
            control_freq=5, data_format='NHWC', xml_ver=0)
    env = mrcnn_env(env, num_blocks=num_blocks, mov_dist=mov_dist,max_steps=max_steps,\
            goal_type=goal_type, reward_type=reward_type)

    # learning configuration #
    learning_rate = args.lr
    batch_size = args.bs 
    buff_size = int(args.buff_size)
    total_steps = int(args.total_steps)
    learn_start = int(args.learn_start)
    update_freq = args.update_freq
    log_freq = args.log_freq
    double = args.double
    per = args.per
    her = args.her

    fcn_ver = args.fcn_ver
    half = args.half
    resnet = args.resnet
    small = args.small
    pre_train = args.pre_train
    continue_learning = args.continue_learning
    if resnet:
        if small:
            from models.fcn_resnet import FCQ_ResNet_Small as FC_QNet
        else:
            from models.fcn_resnet import FCQ_ResNet as FC_QNet
    elif fcn_ver==1:
        if half:
            from models.fcn import FC_QNet_half as FC_QNet
        else:
            from models.fcn import FC_QNet
    elif fcn_ver==2:
        from models.fcn_upsample import FC_QNet
    elif fcn_ver==3:
        from models.fcn_v3 import FC_QNet
    else:
        exit()

    in_channel = 4
    learning(env=env, seg=backsub, savename=savename, n_actions=8, in_channel=in_channel, \
            learning_rate=learning_rate, batch_size=batch_size, buff_size=buff_size, \
            total_steps=total_steps, learn_start=learn_start, update_freq=update_freq, \
            log_freq=log_freq, double=double, her=her, per=per, visualize_q=visualize_q, \
            continue_learning=continue_learning, model_path=model_path, pre_train=pre_train)
