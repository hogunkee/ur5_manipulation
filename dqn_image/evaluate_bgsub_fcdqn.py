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
        cX = int(x.mean())
        cY = int(y.mean())
        # points = np.array(list(zip(x, y)))
        # M = cv2.moments(points)
        # print(M)
        # cX = int(M["m10"] / M["m00"])
        # cY = int(M["m01"] / M["m00"])
        centers.append([cX, cY])
    return np.array(centers)

def color_matching(colors_to, colors_from):
    nc = len(colors_to)
    assert colors_to.shape == colors_from.shape
    tiled = np.tile(colors_from, nc).reshape(nc, nc, 3)
    matching = np.argmin(np.linalg.norm(tiled - colors_to, axis=-1), axis=0)
    # print(colors_to)
    # print(colors_from)
    # print(matching)
    return matching

def get_state_goal(env, segmodule, state, target_color=None):
    flag_success = False
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

    elif env.goal_type=='block':
        image = state[0] * 255
        goal_image = state[1] * 255
        masks, colors, _ = segmodule.get_masks(image, env.num_blocks)
        gmasks, gcolors, _ = segmodule.get_masks(goal_image, env.num_blocks)

        # current - goal object matching #
        match = color_matching(colors, gcolors)
        gmasks = gmasks[match]
        current_centers = get_centers(masks)
        goal_centers = get_centers(gmasks)

        # find target object #
        center_dist = np.linalg.norm(goal_centers - current_centers, axis=1)
        if target_color is not None:
            t_obj = np.argmin(np.linalg.norm(colors - target_color, axis=1))
            # seems to have reached #
            if center_dist[t_obj] < 6:
                # print("changing targets")
                t_obj = np.random.randint(len(masks))
        else:
            t_obj = np.random.randint(len(masks))
        target_color = colors[t_obj]
        target_seg = masks[t_obj]
        obstacle_seg = np.any([masks[o] for o in range(len(masks)) if o != t_obj], 0)
        workspace_seg = segmodule.workspace_seg
        state = np.concatenate([target_seg, obstacle_seg, workspace_seg]).reshape(-1, 96, 96)

        # make a pixel-goal image #
        cX, cY = goal_centers[t_obj]
        pixel_goal = np.zeros([env.env.camera_height, env.env.camera_width])
        cv2.circle(pixel_goal, (cY, cX), 1, 1, -1)
        goal = np.array([pixel_goal])
    return [state, goal], target_color, [image, goal_image, goal_centers]

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

    log_returns = []
    log_eplen = []
    log_out = []
    log_success = []
    log_success_block = [[] for i in range(env.num_blocks)]

    plt.rc('axes', labelsize=6)
    plt.rc('font', size=6)
    if visualize_q:
        plt.show()
        fig, ax = plt.subplots(2, 4)
        ax[0][0].set_title('Observation')
        ax[0][1].set_title('Goal')
        ax[0][2].set_title('State-Goal')
        ax[0][3].set_title('Q-map')
        ax[1][0].set_title('Goal centers')
        ax[1][1].set_title('Target')
        ax[1][2].set_title('Obstacles')
        ax[1][3].set_title('Background')
        for i in range(2):
            for j in range(4):
                ax[i][j].set_xticks([])
                ax[i][j].set_yticks([])

        fig.set_figheight(4)
        fig.set_figwidth(8)
        plt.show(block=False)
        fig.canvas.draw()
        fig.canvas.draw()

    for ne in range(num_trials):
        env.set_target(-1)
        state = env.reset()
        pre_action = None
        target_color = None

        ep_len = 0
        episode_reward = 0
        for t_step in range(env.max_steps):
            state, target_color, [im, gim, gcenters] = get_state_goal(env, seg, state, target_color)
            action, q_map = get_action(env, FCQ, state, epsilon=0.0, pre_action=pre_action, with_q=True)
            if visualize_q:
                # ax[0][4].imshow(np.array(masks).transpose([1,2,0]))
                # ax[1][4].imshow(np.array(gmasks).transpose([1, 2, 0]))
                ax[0][0].imshow(im/255)
                ax[0][1].imshow(gim/255)

                im_gpixel = np.zeros([env.env.camera_height, env.env.camera_width])
                for center in gcenters:
                    cx, cy = center
                    cv2.circle(im_gpixel, (cy, cx), 1, 1, -1)
                ax[1][0].imshow(im_gpixel)

                im_obs = deepcopy(state[0]).transpose([1, 2, 0])
                ax[1][3].imshow(im_obs[:, :, 2])
                im_obs[im_obs[:, :, 0].astype(bool)] = [1, 0, 0]
                im_obs[im_obs[:, :, 1].astype(bool)] = [0, 1, 0]
                ax[1][1].imshow(im_obs[:, :, 0])
                ax[1][2].imshow(im_obs[:, :, 1])

                im_goal = deepcopy(state[1]).reshape(96, 96)
                im_obs[:, :, 0] += im_goal
                im_obs[im_obs[:, :, 0].astype(bool)] = [1, 0, 0]
                im_obs[action[0], action[1]] = [1, 1, 1]
                ax[0][2].imshow(im_obs)
                q_map = q_map.transpose([1, 2, 0]).max(2)
                ax[0][3].imshow(q_map/q_map.max())
                fig.canvas.draw()

            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            # print(info['block_success'])

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

    # np.save(f'scenes/rgb', np.array(rgbs))
    # np.save(f'scenes/depth', np.array(depths))
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
    parser.add_argument("--task", default=1, type=int)
    parser.add_argument("--xml", default=0, type=int)
    parser.add_argument("--color", action="store_true")
    parser.add_argument("--num_blocks", default=3, type=int)
    parser.add_argument("--dist", default=0.08, type=float)
    parser.add_argument("--max_steps", default=50, type=int)
    parser.add_argument("--camera_height", default=96, type=int)
    parser.add_argument("--camera_width", default=96, type=int)
    parser.add_argument("--goal", default="block", type=str)
    parser.add_argument("--reward", default="new", type=str)
    parser.add_argument("--fcn_ver", default=1, type=int)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--small", action="store_true") # default: False
    parser.add_argument("--resnet", action="store_false") # default: True
    parser.add_argument("--model_path", default="0803_1746", type=str)
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
    task = 1 #args.task
    xml_ver = args.xml
    color = args.color
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
            control_freq=5, data_format='NHWC', xml_ver=xml_ver, color=color)
    env = mrcnn_env(env, num_blocks=num_blocks, mov_dist=mov_dist,max_steps=max_steps,\
            goal_type=goal_type, reward_type=reward_type)

    fcn_ver = args.fcn_ver
    half = args.half
    resnet = args.resnet
    small = args.small
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
    evaluate(env=env, seg=backsub, n_actions=8, in_channel=in_channel, model_path=model_path,\
            num_trials=num_trials, visualize_q=visualize_q)
