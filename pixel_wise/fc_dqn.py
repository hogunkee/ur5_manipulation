import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../ur5_mujoco'))
from pushpixel_env import *
from utils import *

import torch
import torch.nn as nn
import argparse
import json

import datetime
import skfmm

from replay_buffer import ReplayBuffer, PER
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import wandb

crop_min = 9 #19 #11 #13
crop_max = 88 #78 #54 #52


def get_sdf(mask):
    return skfmm.distance(mask.astype(int) - 0.5, dx=1)

def get_action(env, fc_qnet, state, epsilon, pre_action=None, with_q=False):
    if np.random.random() < epsilon:
        action = [np.random.randint(crop_min,crop_max), np.random.randint(crop_min,crop_max), np.random.randint(env.num_bins)]
        if with_q:
            state_im = torch.tensor([state[0]]).cuda()
            goal_im = torch.tensor([state[1]]).cuda()
            state_goal = torch.cat((state_im, goal_im), 1)
            q_value = fc_qnet(state_goal)
            q_raw = q_value[0].detach().cpu().numpy()
            q = np.zeros_like(q_raw)
            q[:, crop_min:crop_max, crop_min:crop_max] = q_raw[:, crop_min:crop_max, crop_min:crop_max]
    else:
        state_im = torch.tensor([state[0]]).cuda()
        goal_im = torch.tensor([state[1]]).cuda()
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


def evaluate(env, n_actions=8, in_channel=6, model_path='', num_trials=10, visualize_q=False):
    FCQ = FC_QNet(n_actions, in_channel).cuda()
    print('Loading trained model: {}'.format(model_path))
    FCQ.load_state_dict(torch.load(model_path))

    ne = 0
    ep_len = 0
    episode_reward = 0
    log_returns = []
    log_eplen = []
    log_success = []
    log_success_b1 = []
    if env.num_blocks>1: log_success_b2 = []
    if env.num_blocks>2: log_success_b3 = []

    env.set_targets(list(range(env.num_blocks)))
    state = env.reset()
    pre_action = None
    if visualize_q:
        plt.rc('axes', labelsize=8)
        plt.rc('font', size=8)
        plt.show()
        fig = plt.figure()
        fig.set_figheight(3)
        fig.set_figwidth(7)

        ax0 = fig.add_subplot(131)
        ax1 = fig.add_subplot(132)
        ax2 = fig.add_subplot(133)
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax0.set_title('Goal')
        ax1.set_title('State')
        ax2.set_title('Q-value')

        fig2, ax = plt.subplots(2, 4)
        fig2.set_figwidth(10)
        for i in range(2):
            for j in range(4):
                ax[i][j].set_xticks([])
                ax[i][j].set_yticks([])
                ax[i][j].set_title("%d\xb0" %((4*i+j)*45))

        s0 = deepcopy(state[0]).transpose([1,2,0])
        if env.goal_type == 'pixel':
            s1 = np.zeros([env.env.camera_height, env.env.camera_width, 3])
            s1[:, :, :env.num_blocks] = state[1].transpose([1, 2, 0])
        else:
            s1 = deepcopy(state[1]).transpose([1, 2, 0])
        im0 = ax0.imshow(s1)
        im = ax1.imshow(s0)
        im2 = ax2.imshow(np.zeros_like(s0))
        plt.show(block=False)
        fig.canvas.draw()
        fig.canvas.draw()

    while ne < num_trials:
        action, q_map = get_action(env, FCQ, state, epsilon=0.0, pre_action=pre_action, with_q=True)
        if visualize_q:
            s0 = deepcopy(state[0]).transpose([1, 2, 0])
            s1 = deepcopy(state[1]).transpose([1, 2, 0])
            im0 = ax0.imshow(s1)
            s0[action[0], action[1]] = [1, 0, 0]
            # q_map = q_map[0]
            for i in range(2):
                for j in range(4):
                    ax[i][j].imshow(q_map[4*i+j], vmin=-0.2, vmax=1.5)
            q_map = q_map.transpose([1,2,0]).max(2)
            # q_map[action[0], action[1]] = 1.5
            ax1.imshow(s0)
            ax2.imshow(q_map, vmin=-0.2, vmax=1.5)

            fig.canvas.draw()
            fig2.canvas.draw()

        next_state, reward, done, info = env.step(action)
        episode_reward += reward

        ep_len += 1
        state = next_state
        pre_action = action

        if done:
            ne += 1
            log_returns.append(episode_reward)
            log_eplen.append(ep_len)
            log_success.append(int(info['success']))
            log_success_b1.append(int(info['block_success'][0]))
            if env.num_blocks>1: log_success_b2.append(int(info['block_success'][1]))
            if env.num_blocks>2: log_success_b3.append(int(info['block_success'][2]))

            print()
            print("{} episodes.".format(ne))
            print("Ep reward: {}".format(log_returns[-1]))
            print("Ep length: {}".format(log_eplen[-1]))
            print("Success rate: {}% ({}/{})".format(100*np.mean(log_success), np.sum(log_success), len(log_success)))
            print("Block 1: {}% ({}/{})".format(100*np.mean(log_success_b1), np.sum(log_success_b1), len(log_success_b1)))
            if env.num_blocks>1: print("Block 2: {}% ({}/{})".format(100*np.mean(log_success_b2), np.sum(log_success_b2), len(log_success_b2)))
            if env.num_blocks>2: print("Block 3: {}% ({}/{})".format(100*np.mean(log_success_b3), np.sum(log_success_b3), len(log_success_b3)))

            state = env.reset()
            pre_action = None
            ep_len = 0
            episode_reward = 0
    print()
    print("="*80)
    print("Evaluation Done.")
    # print("Rewards: {}".format(log_returns))
    print("Mean reward: {0:.2f}".format(np.mean(log_returns)))
    print("Mean episode length: {}".format(np.mean(log_eplen)))
    print("Success rate: {}".format(100*np.mean(log_success)))
    print("Block 1: {}".format(100*np.mean(log_success_b1)))
    if env.num_blocks>1: print("Block 2: {}".format(100*np.mean(log_success_b2)))
    if env.num_blocks>2: print("Block 3: {}".format(100*np.mean(log_success_b3)))


def learning(env, 
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
        goal_type='circle',
        continue_learning=False,
        model_path='',
        wandb_off=False,
        ):

    FCQ = FCQNet(n_actions, in_channel).cuda()
    if continue_learning:
        FCQ.load_state_dict(torch.load(model_path))
    FCQ_target = FCQNet(n_actions, in_channel).cuda()
    FCQ_target.load_state_dict(FCQ.state_dict())

    # criterion = nn.SmoothL1Loss(reduction=None).cuda()
    # criterion = nn.MSELoss(reduction='mean')
    #optimizer = torch.optim.SGD(FCQ.parameters(), lr=learning_rate, momentum=0.9, weight_decay=2e-5)
    optimizer = torch.optim.Adam(FCQ.parameters(), lr=learning_rate)

    if per:
        if goal_type=='pixel':
            goal_ch = env.num_blocks
        else:
            goal_ch = 3
        replay_buffer = PER([3, env.env.camera_height, env.env.camera_width], \
                    [goal_ch, env.env.camera_height, env.env.camera_width], 1, \
                    save_goal=True, save_gripper=False, max_size=int(buff_size))
    else:
        replay_buffer = ReplayBuffer([2, env.env.camera_height, env.env.camera_width], \
                [2, env.env.camera_height, env.env.camera_width], dim_action=3, \
                max_size=int(buff_size))

    model_parameters = filter(lambda p: p.requires_grad, FCQ.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("# of params: %d"%params)


    if double:
        calculate_loss = calculate_loss_double_fcdqn
    else:
    if continue_learning:
        numpy_log = np.load(model_path.replace('models/', 'board/').replace('.pth', '.npy'))
        log_returns = numpy_log[0].tolist()
        log_loss = numpy_log[1].tolist()
        log_eplen = numpy_log[2].tolist()
        log_epsilon = numpy_log[3].tolist()
        log_success = numpy_log[4].tolist()
        log_collisions = numpy_log[5].tolist()
        log_out = numpy_log[6].tolist()
        log_success_b1 = numpy_log[7].tolist()
        if env.num_blocks>1: log_success_b2 = numpy_log[8].tolist()
        if env.num_blocks>2: log_success_b3 = numpy_log[9].tolist()
    else:
        log_returns = []
        log_loss = []
        log_eplen = []
        log_epsilon = []
        log_out = []
        log_success = []
        log_success_b1 = []
        if env.num_blocks>1: log_success_b2 = []
        if env.num_blocks>2: log_success_b3 = []
        log_collisions = []
    log_minibatchloss = []

    if not os.path.exists("results/graph/"):
        os.makedirs("results/graph/")
    if not os.path.exists("results/models/"):
        os.makedirs("results/models/")
    if not os.path.exists("results/board/"):
        os.makedirs("results/board/")

    plt.show(block=False)
    plt.rc('axes', labelsize=6)
    plt.rc('font', size=6)
    f, axes = plt.subplots(3, 3) # 3,2
    f.set_figheight(12) #9 #15
    f.set_figwidth(20) #12 #10

    # axes[0][0].set_title('Loss')
    # axes[1][0].set_title('Episode Return')
    # axes[2][0].set_title('Episode Length')
    # axes[0][1].set_title('Success Rate')
    # axes[1][1].set_title('Out of Range')
    # axes[2][1].set_title('Num Collisions')

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

    env.set_targets(list(range(env.num_blocks)))
    state = env.reset()
    pre_action = None

    if visualize_q:
        fig = plt.figure()
        ax0 = fig.add_subplot(131)
        ax1 = fig.add_subplot(132)
        ax2 = fig.add_subplot(133)

        s0 = deepcopy(state[0]).transpose([1,2,0])
        s1 = deepcopy(state[1]).transpose([1, 2, 0])
        im0 = ax0.imshow(s1)
        im = ax1.imshow(s0)
        im2 = ax2.imshow(np.zeros_like(s0))
        plt.show(block=False)
        fig.canvas.draw()
        fig.canvas.draw()

    while t_step < total_steps:
        action, q_map = get_action(env, FCQ, state, epsilon=epsilon, pre_action=pre_action, with_q=True)
        if visualize_q:
            s0 = deepcopy(state[0]).transpose([1, 2, 0])
            s1 = deepcopy(state[1]).transpose([1, 2, 0])
            im0 = ax0.imshow(s1)
            s0[action[0], action[1]] = [1, 0, 0]
            # q_map = q_map[0]
            q_map = q_map.transpose([1,2,0]).max(2)
            im = ax1.imshow(s0)
            im2 = ax2.imshow(q_map/q_map.max())
            print('min_q:', q_map.min(), '/ max_q:', q_map.max())
            fig.canvas.draw()

        next_state, reward, done, info = env.step(action)
        episode_reward += reward

        ## save transition to the replay buffer ##
        if per:
            state_im = torch.tensor([state[0]]).cuda()
            goal_im = torch.tensor([state[1]]).cuda()
            next_state_im = torch.tensor([next_state[0]]).cuda()
            action_tensor = torch.tensor([action]).cuda()

            batch = [state_im, next_state_im, action_tensor, reward, 1-int(done), goal_im]
            _, error = calculate_loss(batch, FCQ, FCQ_target)
            error = error.data.detach().cpu().numpy()
            replay_buffer.add(error, [state[0], 0.0], action, [next_state[0], 0.0], reward, done, state[1])

        else:
            replay_buffer.add([state[0], 0.0], action, [next_state[0], 0.0], reward, done, state[1])
        ## HER ##
        if her and not done:
            her_sample = sample_her_transitions(env, info, next_state)
            ig_samples = sample_ig_transitions(env, info, next_state, num_samples=3)
            samples = her_sample + ig_samples
            for sample in samples:
                reward_re, goal_image, done_re, block_success_re = sample
                if per:
                    goal_im_re = torch.tensor([goal_image]).cuda() # replaced goal
                    batch = [state_im, next_state_im, action_tensor, reward_re, 1-int(done_re), goal_im_re]
                    _, error = calculate_loss(batch, FCQ, FCQ_target)
                    error = error.data.detach().cpu().numpy()
                    replay_buffer.add(error, [state[0], 0.0], action, [next_state[0], 0.0], reward_re, done_re, goal_image)
                else:
                    replay_buffer.add([state[0], 0.0], action, [next_state[0], 0.0], reward_re, done_re, goal_image)

        if t_step < learn_start:
            if done:
                state = env.reset()
                pre_action = None
                episode_reward = 0
            else:
                state = next_state
                pre_action = action
            learn_start -= 1
            if learn_start==0:
                epsilon = start_epsilon
            continue

        ## sample from replay buff & update networks ##
        data = [
                torch.FloatTensor(state[0]).cuda(),
                torch.FloatTensor(next_state[0]).cuda(),
                torch.FloatTensor(action).cuda(),
                torch.FloatTensor([reward]).cuda(),
                torch.FloatTensor([1 - done]).cuda(),
                torch.FloatTensor(state[1]).cuda()
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
            log_success_b1.append(int(info['block_success'][0]))
            if env.num_blocks>1: log_success_b2.append(int(info['block_success'][1]))
            if env.num_blocks>2: log_success_b3.append(int(info['block_success'][2]))
            log_collisions.append(num_collisions)

            if ne % log_freq == 0:
                log_mean_returns = smoothing_log(log_returns, log_freq)
                log_mean_loss = smoothing_log(log_loss, log_freq)
                log_mean_eplen = smoothing_log(log_eplen, log_freq)
                log_mean_out = smoothing_log(log_out, log_freq)
                log_mean_success = smoothing_log(log_success, log_freq)
                log_mean_success_b1 = smoothing_log(log_success_b1, log_freq)
                if env.num_blocks>1: log_mean_success_b2 = smoothing_log(log_success_b2, log_freq)
                if env.num_blocks>2: log_mean_success_b3 = smoothing_log(log_success_b3, log_freq)
                log_mean_collisions = smoothing_log(log_collisions, log_freq)

                print()
                print("{} episodes. ({}/{} steps)".format(ne, t_step, total_steps))
                print("Success rate: {0:.2f}".format(log_mean_success[-1]))
                print("Block 1: {0:.2f}".format(log_mean_success_b1[-1]))
                if env.num_blocks > 1: print("Block 2: {0:.2f}".format(log_mean_success_b2[-1]))
                if env.num_blocks > 2: print("Block 3: {0:.2f}".format(log_mean_success_b3[-1]))
                print("Mean reward: {0:.2f}".format(log_mean_returns[-1]))
                print("Mean loss: {0:.6f}".format(log_mean_loss[-1]))
                # print("Ep reward: {}".format(log_returns[-1]))
                print("Ep length: {}".format(log_mean_eplen[-1]))
                print("Epsilon: {}".format(epsilon))

                # axes[0][0].plot(log_loss, color='#ff7f00', linewidth=0.5)
                # axes[1][0].plot(log_returns, color='#60c7ff', linewidth=0.5)
                # axes[2][0].plot(log_eplen, color='#83dcb7', linewidth=0.5)
                # axes[2][1].plot(log_collisions, color='#ff33cc', linewidth=0.5)
                #
                # axes[0][0].plot(log_mean_loss, color='red')
                # axes[1][0].plot(log_mean_returns, color='blue')
                # axes[2][0].plot(log_mean_eplen, color='green')
                # axes[0][1].plot(log_mean_success, color='red')
                # axes[1][1].plot(log_mean_out, color='black')
                # axes[2][1].plot(log_mean_collisions, color='#663399')

                axes[1][2].plot(log_loss, color='#ff7f00', linewidth=0.5)  # 3->6
                axes[1][1].plot(log_returns, color='#60c7ff', linewidth=0.5)  # 5
                axes[2][0].plot(log_eplen, color='#83dcb7', linewidth=0.5)  # 7
                axes[2][2].plot(log_collisions, color='#ff33cc', linewidth=0.5)  # 8->9

                axes[0][0].plot(log_mean_success_b1, color='red')  # 1
                if env.num_blocks>1: axes[0][1].plot(log_mean_success_b2, color='red')  # 2
                if env.num_blocks>2: axes[0][2].plot(log_mean_success_b3, color='red')  # 3

                axes[1][2].plot(log_mean_loss, color='red')  # 3->6
                axes[1][1].plot(log_mean_returns, color='blue')  # 5
                axes[2][0].plot(log_mean_eplen, color='green')  # 7
                axes[1][0].plot(log_mean_success, color='red')  # 4
                axes[2][1].plot(log_mean_out, color='black')  # 6->8
                axes[2][2].plot(log_mean_collisions, color='#663399')  # 8->9

                #f.canvas.draw()
                # plt.pause(0.001)
                plt.savefig('results/graph/%s.png' % savename)
                # plt.close()

                log_list = [
                        log_returns,  # 0
                        log_loss,  # 1
                        log_eplen,  # 2
                        log_epsilon,  # 3
                        log_success,  # 4
                        log_collisions,  # 5
                        log_out,  # 6
                        log_success_b1 #7
                        ]
                if env.num_blocks>1: log_list.append(log_success_b2) #8
                if env.num_blocks>2: log_list.append(log_success_b3) #9
                numpy_log = np.array(log_list)
                np.save('results/board/%s' %savename, numpy_log)

                if log_mean_success[-1] > max_success:
                    max_success = log_mean_success[-1]
                    torch.save(FCQ.state_dict(), 'results/models/%s.pth' % savename)
                    print("Max performance! saving the model.")

            episode_reward = 0.
            log_minibatchloss = []
            state = env.reset()
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
    ## env ##
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--num_blocks", default=1, type=int)
    parser.add_argument("--dist", default=0.08, type=float)
    parser.add_argument("--max_steps", default=30, type=int)
    parser.add_argument("--camera_height", default=96, type=int)
    parser.add_argument("--camera_width", default=96, type=int)
    ## learning ##
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--bs", default=6, type=int)
    parser.add_argument("--buff_size", default=1e3, type=float)
    parser.add_argument("--total_steps", default=2e5, type=float)
    parser.add_argument("--learn_start", default=1e3, type=float)
    parser.add_argument("--update_freq", default=100, type=int)
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--per", action="store_true")
    parser.add_argument("--her", action="store_true")
    parser.add_argument("--reward", default="binary", type=str)
    parser.add_argument("--goal", default="circle", type=str)
    parser.add_argument("--hide_goal", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--continue_learning", action="store_true")
    ## Evaluate ##
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--model_path", default="FCDQN_reach_0412_1714.pth", type=str)
    parser.add_argument("--num_trials", default=50, type=int)
    # etc #
    parser.add_argument("--show_q", action="store_true")
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--wandb_off", action="store_true")
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
    hide_goal = args.hide_goal

    # evaluate configuration #
    evaluation = args.evaluate
    model_path = os.path.join("results/models/FCDQN_%s.pth"%args.model_path)
    num_trials = args.num_trials
    visualize_q = args.show_q

    gpu = args.gpu
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if str(gpu) in visible_gpus:
            gpu_idx = visible_gpus.index(str(gpu))
            torch.cuda.set_device(gpu_idx)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    now = datetime.datetime.now()
    savename = "FCDQN_%s" % (now.strftime("%m%d_%H%M"))
    if not evaluation:
        if not os.path.exists("results/config/"):
            os.makedirs("results/config/")
        with open("results/config/%s.json" % savename, 'w') as cf:
            json.dump(args.__dict__, cf, indent=2)

    # wandb log #
    log_name = savename
    if n1==n2:
        log_name += '_%db' %n1
    else:
        log_name += '_%d-%db' %(n1, n2)
    wandb_off = args.wandb_off
    if not wandb_off:
        wandb.init(project="SDF Matching")
        wandb.run.name = log_name
        wandb.config.update(args)
        wandb.run.save()

    env = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
            control_freq=5, data_format='NCHW', xml_ver=0)
    env = pushpixel_env(env, num_blocks=num_blocks, mov_dist=mov_dist, max_steps=max_steps, \
            task=1, reward_type=reward_type, goal_type=goal_type, hide_goal=hide_goal)

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
    continue_learning = args.continue_learning
    if half:
        from models.fcn_resnet import FCQResNetSmall as FCQNet
    else:
        from models.fcn_resnet import FCQResNet as FCQNet

    in_channel = 6
            
    if evaluation:
        evaluate(env=env, n_actions=8, in_channel=in_channel, model_path=model_path, \
                num_trials=num_trials, visualize_q=visualize_q)
    else:
        learning(env=env, savename=savename, n_actions=8, in_channel=in_channel, \
                learning_rate=learning_rate, batch_size=batch_size, buff_size=buff_size, \
                total_steps=total_steps, learn_start=learn_start, update_freq=update_freq, \
                log_freq=log_freq, double=double, her=her, per=per, visualize_q=visualize_q, \
                goal_type=goal_type, continue_learning=continue_learning, \
                model_path=model_path, wandb_off=wandb_off)
