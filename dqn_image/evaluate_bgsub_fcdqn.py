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

from replay_buffer import ReplayBuffer, PER
from matplotlib import pyplot as plt

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

crop_min = 9 #19 #11 #13
crop_max = 88 #78 #54 #52


def get_centers(masks):
    centers = []
    for mask in masks:
        x, y = np.nonzero(mask)
        points = np.array(list(zip(x, y)))
        M = cv2.moments(points)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centers.append([cX, cY])
    return np.array(centers)

def color_matching(colors_to, colors_from):
    nc = len(colors_to)
    assert colors_to.shape == colors_from
    tiled = np.tile(colors_to, nc).reshape(nc, nc, 3)
    matching = np.argmin(np.linalg.norm(tiled - colors_from, axis=-1), axis=0)
    return matching

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
            t_obj= np.argmin(np.linalg.norm(colors - target_color, axis=1))
        else:
            t_obj = np.random.randint(len(masks))
        target_color = colors[t_obj]
        target_seg = masks[t_obj]
        obstacle_seg = np.any([masks[o] for o in range(len(masks)) if o != t_obj], 0)
        workspace_seg = segmodule.workspace_seg
        # workspace_seg = segmodule.get_workspace_seg(image)

        env.set_target_with_color(target_color)
        target_idx = env.seg_target

        try:
            state = np.concatenate([target_seg, obstacle_seg, workspace_seg]).reshape(-1, 96, 96)
        except:
            print(len(masks))
            print(target_seg.shape)
            print(obstacle_seg.shape)
            print(workspace_seg.shape)
        goal = goal_image[target_idx: target_idx+1]

    elif env.goal_type=='circle':
        image = state[0] * 255
        goal_image = state[1] * 255
        masks, colors, _ = segmodule.get_masks(image, env.num_blocks)
        gmasks, gcolors, _ = segmodule.get_masks(goal_image, env.num_blocks)

        # current - goal object matching #
        match = color_matching(colors, gcolors)
        gmasks = gmasks[match]
        goal_centers = get_centers(gmasks)
        current_centers = get_centers(masks)

        # find target object #
        center_dist = np.linalg.norm(goal_centers - current_centers, axis=1)
        if target_color is not None:
            t_obj= np.argmin(np.linalg.norm(colors - target_color, axis=1))
            # seems to have reached #
            if center_dist[t_obj] > 4:
                t_obj = np.random.randint(len(masks))
        else:
            t_obj = np.random.randint(len(masks))
        target_color = colors[t_obj]
        target_seg = masks[t_obj]
        obstacle_seg = np.any([masks[o] for o in range(len(masks)) if o != t_obj], 0)
        workspace_seg = segmodule.workspace_seg
        try:
            state = np.concatenate([target_seg, obstacle_seg, workspace_seg]).reshape(-1, 96, 96)
        except:
            print(len(masks))
            print(target_seg.shape)
            print(obstacle_seg.shape)
            print(workspace_seg.shape)

        # make a pixel-goal image #
        cX, cY = get_mask_center(gmasks[t_obj])
        pixel_goal = np.zeros([env.camera_height, env.camera_width])
        cv2.circle(pixel_goal, (cY, cX), 1, 1, -1)
        goal = np.array([pixel_goal])

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

def evaluate(env, 
        seg,
        n_actions=8,
        in_channel=6,
        model_path='',
        num_trials=10,
        visualize_q=False,
        ):
    FCQ = FC_QNet(n_actions, in_channel).type(dtype)
    print('Loading trained model: {}'.format(model_path))
    FCQ.load_state_dict(torch.load(model_path))

    ne = 0
    ep_len = 0
    episode_reward = 0
    log_returns = []
    log_eplen = []
    log_out = []
    log_success = []
    log_success_block = [[], [], []]

    plt.rc('axes', labelsize=6)
    plt.rc('font', size=6)
    if visualize_q:
        plt.show()
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

        '''
        s0 = deepcopy(state[0]).transpose([1,2,0])
        s1 = deepcopy(state[1]).reshape(96, 96)
        ax0.imshow(s1)
        ax1.imshow(s0)
        ax2.imshow(np.zeros_like(s0))
        ax3.imshow(s0[:, :, 0])
        ax4.imshow(s0[:, :, 1])
        ax5.imshow(s0[:, :, 2])
        '''
        plt.show(block=False)
        fig.canvas.draw()
        fig.canvas.draw()

    # imc = 0
    # rgbs, depths = [], []
    for ne in range(num_trials):
        env.set_target(-1)
        state = env.reset()
        pre_action = None
        target_color = None

        ep_len = 0
        episode_reward = 0
        for t_step in range(env.max_steps):
            state, target_color = get_state_goal(env, seg, state, target_color)
            action, q_map = get_action(env, FCQ, state, epsilon=0.0, pre_action=pre_action, with_q=True)
            if visualize_q:
                s0 = deepcopy(state[0]).transpose([1, 2, 0])
                s1 = deepcopy(state[1]).reshape(96, 96)
                ax0.imshow(s1)
                ax3.imshow(s0[:, :, 0])
                ax4.imshow(s0[:, :, 1])
                ax5.imshow(s0[:, :, 2])

                s0[action[0], action[1]] = [1, 0, 0]
                ax1.imshow(s0)
                q_map = q_map.transpose([1, 2, 0]).max(2)
                ax2.imshow(q_map/q_map.max())
                fig.canvas.draw()

            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            # rgbs.append(info['rgb'])
            # depths.append(info['depth'])
            # np.save(f'scenes/rgb_{imc}', info['rgb'])
            # np.save(f'scenes/depth_{imc}', info['depth'])
            # rgb = (info['rgb'] * 255).astype(np.uint8)
            # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            # depth = (info['depth'] - info['depth'].min()) * 10000
            # cv2.imwrite(f'scenes/rgb_{imc}.png', rgb)
            # cv2.imwrite(f'scenes/depth_{imc}.png', depth)
            # imc += 1

            ep_len += 1
            state = next_state
            pre_action = action
            if done:
                break

        log_returns.append(episode_reward)
        log_eplen.append(ep_len)
        log_out.append(int(info['out_of_range']))
        log_success.append(int(np.all(info['block_success'])))
        #log_success.append(int(info['success']))
        for o in range(3):
            log_success_block[o].append(int(info['block_success'][o]))

        print()
        print("{} episodes.".format(ne))
        print("Ep reward: {}".format(log_returns[-1]))
        print("Ep length: {}".format(log_eplen[-1]))
        print("Success rate: {}% ({}/{})".format(100*np.mean(log_success), np.sum(log_success), len(log_success)))
        for o in range(3):
            print("Block {}: {}% ({}/{})".format(o+1, 100*np.mean(log_success_block[o]), np.sum(log_success_block[o]), len(log_success_block[o])))
        print("Out of range: {}".format(np.mean(log_out)))

    # np.save(f'scenes/rgb', np.array(rgbs))
    # np.save(f'scenes/depth', np.array(depths))
    print()
    print("="*80)
    print("Evaluation Done.")
    print("Mean reward: {0:.2f}".format(np.mean(log_returns)))
    print("Mean episode length: {}".format(np.mean(log_eplen)))
    print("Success rate: {}".format(100*np.mean(log_success)))
    for o in range(3):
        print("Block {}: {}% ({}/{})".format(o+1, 100*np.mean(log_success_block[o]), np.sum(log_success_block[o]), len(log_success_block[o])))
    print("Out of range: {}".format(np.mean(log_out)))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--task", default=1, type=int)
    parser.add_argument("--num_blocks", default=3, type=int)
    parser.add_argument("--dist", default=0.08, type=float)
    parser.add_argument("--max_steps", default=50, type=int)
    parser.add_argument("--camera_height", default=96, type=int)
    parser.add_argument("--camera_width", default=96, type=int)
    parser.add_argument("--goal", default="circle", type=str)
    parser.add_argument("--reward", default="new", type=str)
    parser.add_argument("--fcn_ver", default=1, type=int)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--resnet", action="store_false") # default: True
    parser.add_argument("--model_path", default="0731_1254", type=str)
    parser.add_argument("--num_trials", default=100, type=int)
    parser.add_argument("--show_q", action="store_true")
    args = parser.parse_args()

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

    # evaluate configuration
    model_path = os.path.join("results/models/BGSB_%s.pth"%args.model_path)
    num_trials = args.num_trials
    visualize_q = args.show_q
    if visualize_q:
        render = True

    backsub = BackgroundSubtraction()
    env = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
            control_freq=5, data_format='NHWC', xml_ver=0)
    env = mrcnn_env(env, num_blocks=num_blocks, mov_dist=mov_dist,max_steps=max_steps,\
            goal_type=goal_type, reward_type=reward_type)

    fcn_ver = args.fcn_ver
    half = args.half
    resnet = args.resnet
    pre_train = args.pre_train
    continue_learning = args.continue_learning
    if resnet:
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
    evaluate(env=env, seg=backsub, n_actions=8, in_channel=in_channel, model_path=model_path,\
            num_trials=num_trials, visualize_q=visualize_q)
