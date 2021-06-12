import numpy as np
import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../ur5_mujoco'))
from pushpixel_env import *
from PIL import Image

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

def generate_goal_image(goal_poses, n_blocks):
    range_x = env.block_range_x
    range_y = env.block_range_y

    if goal_type=='circle':
        goal_image = deepcopy(env.background_img)
        for i in range(n_blocks):
            cv2.circle(goal_image, env.pos2pixel(*goal_poses[i]), 1, env.colors[i], -1)
        goal_image = np.transpose(goal_image, [2, 0, 1])

    elif goal_type=='pixel':
        goal_ims = []
        for i in range(n_blocks):
            zero_array = np.zeros([env.env.camera_height, env.env.camera_width])
            cv2.circle(zero_array, env.pos2pixel(*goal_poses[i]), 1, 1, -1)
            goal_ims.append(zero_array)
        goal_image = np.concatenate(goal_ims)
        goal_image = goal_image.reshape([n_blocks, env.env.camera_height, env.env.camera_width])

    elif goal_type=='block':
        pass

    return goal_image

def get_action(env, fc_qnet, state, epsilon, pre_action=None, with_q=False, sample='sum'):
    if np.random.random() < epsilon:
        action = [np.random.randint(crop_min,crop_max), np.random.randint(crop_min,crop_max), np.random.randint(env.num_bins)]
        # action = [np.random.randint(env.env.camera_height), np.random.randint(env.env.camera_width), np.random.randint(env.num_bins)]
        if with_q:
            state_im = torch.tensor([state[0]]).type(dtype)
            goal_im = torch.tensor([state[1]]).type(dtype)
            state_goal = torch.cat((state_im, goal_im), 1)
            q_value = fc_qnet(state_goal, True)
            q_raw = q_value[0].detach().cpu().numpy()
            q = np.zeros_like(q_raw)
            q[:, crop_min:crop_max, crop_min:crop_max] = q_raw[:, crop_min:crop_max, crop_min:crop_max]
    else:
        state_im = torch.tensor([state[0]]).type(dtype)
        goal_im = torch.tensor([state[1]]).type(dtype)
        state_goal = torch.cat((state_im, goal_im), 1)
        q_value = fc_qnet(state_goal, True)
        q_raw = q_value[0].detach().cpu().numpy() # q_raw: nb x 8 x 96 x 96

        q = np.zeros_like(q_raw[0])
        # summation of Q-values
        if sample=='sum':
            for o in range(env.num_blocks):
                q[:, crop_min:crop_max, crop_min:crop_max] += q_raw[o, :, crop_min:crop_max, crop_min:crop_max]
        # sampling with object-wise q_max
        elif sample=='choice':
            prob = []
            for o in range(env.num_blocks):
                prob.append(np.max([q_raw[o].max(), 0.1]))
            prob /= np.sum(prob)
            selected_obj = np.random.choice(env.num_blocks, 1, p=prob)[0]
            q[:, crop_min:crop_max, crop_min:crop_max] += q_raw[selected_obj, :, crop_min:crop_max, crop_min:crop_max]
        elif sample=='max':
            q[:, crop_min:crop_max, crop_min:crop_max] += q_raw.max(0)[:, crop_min:crop_max, crop_min:crop_max]

        # avoid redundant motion #
        if pre_action is not None:
            q[pre_action[2], pre_action[0], pre_action[1]] = q.min()
        # image coordinate #
        aidx_x = q.max(0).max(1).argmax()
        aidx_y = q.max(0).max(0).argmax()
        aidx_th = q.argmax(0)[aidx_x, aidx_y]
        action = [aidx_x, aidx_y, aidx_th]

    if with_q:
        return action, q, q_raw
    else:
        return action


def evaluate(env, n_blocks=3, in_channel=6, model_path='', visualize_q=False, sampling='choice'):
    FCQ = FC_QNet(8, in_channel, n_blocks).type(dtype)
    print('Loading trained model: {}'.format(model_path))
    FCQ.load_state_dict(torch.load(model_path))

    state = env.reset()
    pre_action = None
    if visualize_q:
        plt.show()
        fig = plt.figure()
        if sampling == 'sum':
            ax0 = fig.add_subplot(131)
            ax1 = fig.add_subplot(132)
            ax2 = fig.add_subplot(133)
        else:
            ax0 = fig.add_subplot(231)
            ax1 = fig.add_subplot(232)
            ax2 = fig.add_subplot(233)
            ax3 = fig.add_subplot(234)
            ax4 = fig.add_subplot(235)
            ax5 = fig.add_subplot(236)

        s0 = deepcopy(state[0]).transpose([1,2,0])
        if env.goal_type == 'pixel':
            s1 = np.zeros([env.env.camera_height, env.env.camera_width, 3])
            s1[:, :, :n_blocks] = state[1].transpose([1, 2, 0])
        else:
            s1 = deepcopy(state[1]).transpose([1, 2, 0])
        ax0.imshow(s1)
        ax1.imshow(s0)
        ax2.imshow(np.zeros_like(s0))
        ax3.imshow(np.zeros_like(s0))
        ax4.imshow(np.zeros_like(s0))
        ax5.imshow(np.zeros_like(s0))
        plt.show(block=False)
        fig.canvas.draw()
        fig.canvas.draw()

    poses = [
        [[0.1, 0.1], [0, 0]],
        [[0.1, 0.1001], [-0.1, 0.100001]],
        [[0.05, 0.1001], [-0.05, 0.100001]],
        [[0.0, 0.09], [0.0, 0.12]],
        ]
    goal_poses = [
        [[-0.1, -0.08], [0.17, 0.17]],
        [[-0.14, 0.11], [0.13, 0.095]],
        [[-0.1, 0.08], [0.09, 0.095]],
        [[0.0, 0.18], [0.05, 0.025]],
        ]
    if False:
        for i in range(len(poses)):
            pose = poses[i]
            goal_pose = goal_poses[i]
            for obj_idx in range(num_blocks):
                tx, ty = pose[obj_idx]
                tz = 0.9
                env.env.sim.data.qpos[7*obj_idx+12: 7*obj_idx+15] = [tx, ty, tz]
                x, y, z, w = euler2quat([0, 0, np.random.uniform(2*np.pi)])
                env.env.sim.data.qpos[7*obj_idx+15: 7*obj_idx+19] = [w, x, y, z]
            env.env.sim.step()
            im_state = env.env.move_to_pos(env.init_pos, grasp=1.0)
            s_im = Image.fromarray((255 * im_state.transpose([1, 2, 0])).astype(np.uint8))
            s_im.save('test_scenes/state_%d.png'%i)

            goal_state = generate_goal_image(goal_poses[i], n_blocks)
            g_im = Image.fromarray((255 * goal_state.transpose([1, 2, 0])).astype(np.uint8))
            g_im.save('test_scenes/goal_%d.png' % i)

    for i in range(len(poses)):
        s_im = imageio.imread('test_scenes/state_%d.png' %i).transpose([2, 0, 1]) / 255.
        g_im = imageio.imread('test_scenes/goal_%d.png' %i).transpose([2, 0, 1]) / 255.
        state = [s_im, g_im]
        action, q_map, q_raw = get_action(env, FCQ, state, epsilon=0.0, pre_action=pre_action, with_q=True, sample=sampling)
        if visualize_q:
            s0 = deepcopy(state[0]).transpose([1, 2, 0])
            if env.goal_type == 'pixel':
                s1 = np.zeros([env.env.camera_height, env.env.camera_width, 3])
                s1[:, :, :n_blocks] = state[1].transpose([1, 2, 0])
            else:
                s1 = deepcopy(state[1]).transpose([1, 2, 0])

            ax0.imshow(s1)
            s0[action[0], action[1]] = [1, 0, 0]
            # q_map = q_map[0]
            q_map = q_map.transpose([1,2,0]).max(2)
            # print(q_map.max())
            # print(q_map.min())
            # q_map[action[0], action[1]] = 1.5
            ax1.imshow(s0)
            ax2.imshow(q_map, vmax=1.8, vmin=-0.2)
            if sampling != 'sum':
                q0 = q_raw[0].transpose([1,2,0]).max(2)
                q1 = q_raw[1].transpose([1, 2, 0]).max(2)
                ax3.imshow(q0, vmax=1.8, vmin=-0.2)
                ax4.imshow(q1, vmax=1.8, vmin=-0.2)
                if num_blocks==3:
                    q2 = q_raw[2].transpose([1, 2, 0]).max(2)
                    ax5.imshow(q2)
            plt.savefig('test_scenes/plot_%d.png'%i)
            fig.canvas.draw()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--num_blocks", default=1, type=int)
    parser.add_argument("--dist", default=0.08, type=float)
    parser.add_argument("--max_steps", default=30, type=int)
    parser.add_argument("--camera_height", default=96, type=int)
    parser.add_argument("--camera_width", default=96, type=int)
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--per", action="store_true")
    parser.add_argument("--reward", default="binary", type=str)
    parser.add_argument("--goal", default="circle", type=str)
    parser.add_argument("--fcn_ver", default=1, type=int)
    parser.add_argument("--sampling", default="choice", type=str)
    parser.add_argument("--half", action="store_true")
    ## Evaluate ##
    parser.add_argument("--model_path", default="SP_####_####.pth", type=str)
    parser.add_argument("--show_q", action="store_true")
    args = parser.parse_args()

    # env configuration #
    render = args.render
    num_blocks = args.num_blocks
    mov_dist = args.dist
    max_steps = args.max_steps
    camera_height = args.camera_height
    camera_width = args.camera_width
    reward_type = args.reward
    goal_type = args.goal

    # nn structure
    half = args.half
    if half:
        from models.seperate_fcn import FC_QNet_half as FC_QNet
    else:
        from models.seperate_fcn import FC_QNet

    # evaluate configuration #
    model_path = os.path.join("results/models/", args.model_path)
    visualize_q = args.show_q

    env = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
            control_freq=5, data_format='NCHW', xml_ver=0)
    env = pushpixel_env(env, num_blocks=num_blocks, mov_dist=mov_dist, max_steps=max_steps, \
            task=1, reward_type=reward_type, goal_type=goal_type, seperate=True)

    # learning configuration #
    double = args.double
    per = args.per
    fcn_ver = args.fcn_ver
    sampling = args.sampling  # 'sum' / 'choice' / 'max'

    if goal_type=="pixel":
        in_channel = 3 + num_blocks
    else:
        in_channel = 6
            
    evaluate(env=env, n_blocks=num_blocks, in_channel=in_channel, model_path=model_path, \
            visualize_q=visualize_q, sampling=sampling)
