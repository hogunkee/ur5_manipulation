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

from replay_buffer import ReplayBuffer, PER
from matplotlib import pyplot as plt

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

crop_min = 9 #19 #11 #13
crop_max = 88 #78 #54 #52


def get_action(env, fc_qnet, cqn, state, epsilon, pre_action=None, with_q=False, cascade=True, sample='sum', output='addR'):
    if np.random.random() < epsilon:
        action = [np.random.randint(crop_min,crop_max), np.random.randint(crop_min,crop_max), np.random.randint(env.num_bins)]
        # action = [np.random.randint(env.env.camera_height), np.random.randint(env.env.camera_width), np.random.randint(env.num_bins)]
        if with_q:
            state_im = torch.tensor([state[0]]).type(dtype)
            goal_im = torch.tensor([state[1]]).type(dtype)
            if env.goal_type=='pixel':
                state_goal = torch.cat((state_im, goal_im[:, 0:1]), 1)
            else:
                state_goal = torch.cat((state_im, goal_im), 1)
            q_value = fc_qnet(state_goal, True)
            q_raw = q_value[0].detach().cpu().numpy()
            q = np.zeros_like(q_raw)
            q[:, crop_min:crop_max, crop_min:crop_max] = q_raw[:, crop_min:crop_max, crop_min:crop_max]
    else:
        state_im = torch.tensor([state[0]]).type(dtype)
        goal_im = torch.tensor([state[1]]).type(dtype)
        if env.goal_type=='pixel':
            state_goal = torch.cat((state_im, goal_im[:, 0:1]), 1)
        else:
            state_goal = torch.cat((state_im, goal_im), 1)

        if cascade:
            q1_value = fc_qnet(state_goal, True)
            state_goal_q = torch.cat((state_im, goal_im, q1_value), 1)
            q2_value = cqn(state_goal_q, True)

            q1_raw = q1_value[0].detach().cpu().numpy() # q_raw: 8 x 96 x 96
            q2_raw = q2_value[0].detach().cpu().numpy() # q_raw: 8 x 96 x 96

            q_raw = np.concatenate([q1_raw, q2_raw]).reshape(env.num_blocks,8,96,96)
            q = np.zeros_like(q_raw[0])
            if output=='':
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
                # sampling from uniform distribution
                elif sample=='uniform':
                    prob = [1./env.num_blocks] * env.num_blocks
                    selected_obj = np.random.choice(env.num_blocks, 1, p=prob)[0]
                    q[:, crop_min:crop_max, crop_min:crop_max] += q_raw[selected_obj, :, crop_min:crop_max, crop_min:crop_max]
                # select maximum q
                elif sample=='max':
                    q[:, crop_min:crop_max, crop_min:crop_max] += q_raw.max(0)[:, crop_min:crop_max, crop_min:crop_max]
                    selected_obj = 1
            elif output=='addR':
                q[:, crop_min:crop_max, crop_min:crop_max] += q2_raw[:, crop_min:crop_max, crop_min:crop_max]
            elif output=='addQ':
                for o in range(env.num_blocks):
                    q[:, crop_min:crop_max, crop_min:crop_max] += q_raw[o, :, crop_min:crop_max, crop_min:crop_max]

        else:
            q_value = fc_qnet(state_goal, True)
            q_raw = q_value[0].detach().cpu().numpy() # q_raw: 8 x 96 x 96
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
        return action, q, q_raw
    else:
        return action


def evaluate(env, n_blocks=3, in_channel=[6,14], model1='', model2='', num_trials=10, visualize_q=False, sampling='uniform', output='addQ'):
    FCQ = FC_QNet(8, in_channel[0]).type(dtype)
    print('Loading trained FCDQN model: {}'.format(model1))
    FCQ.load_state_dict(torch.load(model1))
    CQN = FC_QNet(8, in_channel[1]).type(dtype)
    print('Loading trained PCQN model: {}'.format(model2))
    CQN.load_state_dict(torch.load(model2))

    ne = 0
    ep_len = 0
    episode_reward = 0
    log_returns = []
    log_eplen = []
    log_success = []

    state = env.reset()
    pre_action = None
    if visualize_q:
        plt.show()
        fig = plt.figure()
        if False:
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

    while ne < num_trials:
        action, q_map, q_raw = get_action(env, FCQ, CQN, state, epsilon=0.0, pre_action=pre_action, with_q=True, cascade=True, sample=sampling, output=output)
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
            if False:
                q0 = q_raw[0].transpose([1,2,0]).max(2)
                q1 = q_raw[1].transpose([1, 2, 0]).max(2)
                ax3.imshow(q0, vmax=1.8, vmin=-0.2)
                ax4.imshow(q1, vmax=1.8, vmin=-0.2)
                if num_blocks==3:
                    q2 = q_raw[2].transpose([1, 2, 0]).max(2)
                    ax5.imshow(q2)
            fig.canvas.draw()

        next_state, rewards, done, info = env.step(action)
        episode_reward += np.sum(rewards)

        ep_len += 1
        state = next_state
        pre_action = action

        if done:
            ne += 1
            log_returns.append(episode_reward)
            log_eplen.append(ep_len)
            log_success.append(int(info['success']))

            print()
            print("{} episodes.".format(ne))
            print("Ep reward: {}".format(log_returns[-1]))
            print("Ep length: {}".format(log_eplen[-1]))
            print("Success rate: {}% ({}/{})".format(100*np.mean(log_success), np.sum(log_success), len(log_success)))

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


def learning(env, 
        savename,
        n_blocks=3,
        in_channel=[6,14],
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
        model1='',
        sampling='uniform',
        output='addQ'
        ):

    FCQ = FC_QNet(8, in_channel[0]).type(dtype)
    FCQ.load_state_dict(torch.load(model1))
    CQN = FC_QNet(8, in_channel[1]).type(dtype)
    CQN_target = FC_QNet(8, in_channel[1]).type(dtype)
    CQN_target.load_state_dict(CQN.state_dict())

    criterion = nn.SmoothL1Loss(reduction=None).type(dtype)
    # criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(CQN.parameters(), lr=learning_rate, momentum=0.9, weight_decay=2e-5)
    # optimizer = torch.optim.Adam(FCQ.parameters(), lr=learning_rate)

    if per:
        if goal_type=='pixel':
            goal_ch = n_blocks
        else:
            goal_ch = 3
        replay_buffer = PER([3, env.env.camera_height, env.env.camera_width], \
                    [goal_ch, env.env.camera_height, env.env.camera_width], 1, \
                    save_goal=True, save_gripper=False, max_size=int(buff_size),\
                    dim_reward=n_blocks)
    else:
        replay_buffer = ReplayBuffer([3, env.env.camera_height, env.env.camera_width], 1, \
                 save_goal=True, save_gripper=False, max_size=int(buff_size),\
                 dim_reward=n_blocks)

    model1_parameters = filter(lambda p: p.requires_grad, FCQ.parameters())
    model2_parameters = filter(lambda p: p.requires_grad, CQN.parameters())
    params = sum([np.prod(p.size()) for p in model1_parameters])
    params += sum([np.prod(p.size()) for p in model2_parameters])
    print("# of params: %d"%params)


    if double:
        calculate_cascade_loss = calculate_cascade_loss_double_cascade_v3
    else:
        calculate_cascade_loss = calculate_cascade_loss_cascade_v3

    log_returns = []
    log_loss = []
    log_eplen = []
    log_epsilon = []
    log_out = []
    log_success = []
    log_collisions = []
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
    f, axes = plt.subplots(3, 2)
    f.set_figheight(9) #15
    f.set_figwidth(12) #10

    axes[0][0].set_title('Loss')
    axes[1][0].set_title('Episode Return')
    axes[2][0].set_title('Episode Length')
    axes[0][1].set_title('Success Rate')
    axes[1][1].set_title('Out of Range')
    axes[2][1].set_title('Num Collisions')

    lr_decay = 0.98
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)

    epsilon = 0.5 #1.0
    start_epsilon = 0.5
    min_epsilon = 0.1
    epsilon_decay = 0.97
    episode_reward = 0.0
    max_success = 0.0
    ep_len = 0
    ne = 0
    t_step = 0
    num_collisions = 0

    state = env.reset()
    pre_action = None

    if visualize_q:
        fig = plt.figure()
        ax0 = fig.add_subplot(131)
        ax1 = fig.add_subplot(132)
        ax2 = fig.add_subplot(133)

        s0 = deepcopy(state[0]).transpose([1,2,0])
        if env.goal_type=='pixel':
            s1 = np.zeros([env.env.camera_height, env.env.camera_width, 3])
            s1[:,:,:n_blocks] = state[1].transpose([1,2,0])
        else:
            s1 = deepcopy(state[1]).transpose([1, 2, 0])
        im0 = ax0.imshow(s1)
        im = ax1.imshow(s0)
        im2 = ax2.imshow(np.zeros_like(s0))
        plt.show(block=False)
        fig.canvas.draw()
        fig.canvas.draw()

    while t_step < total_steps:
        action, q_map, _ = get_action(env, FCQ, CQN, state, epsilon=epsilon, pre_action=pre_action, with_q=True, cascade=True, sample=sampling)

        if visualize_q:
            s0 = deepcopy(state[0]).transpose([1, 2, 0])
            if env.goal_type == 'pixel':
                s1 = np.zeros([env.env.camera_height, env.env.camera_width, 3])
                s1[:, :, :n_blocks] = state[1].transpose([1, 2, 0])
            else:
                s1 = deepcopy(state[1]).transpose([1, 2, 0])
            im0 = ax0.imshow(s1)
            s0[action[0], action[1]] = [1, 0, 0]
            # q_map = q_map[0]
            q_map = q_map.transpose([1,2,0]).max(2)
            im = ax1.imshow(s0)
            im2 = ax2.imshow(q_map/q_map.max())
            print('min_q:', q_map.min(), '/ max_q:', q_map.max())
            fig.canvas.draw()

        next_state, rewards, done, info = env.step(action)
        episode_reward += np.sum(rewards)

        ## save transition to the replay buffer ##
        if per:
            state_im = torch.tensor([state[0]]).type(dtype)
            goal_im = torch.tensor([state[1]]).type(dtype)
            next_state_im = torch.tensor([next_state[0]]).type(dtype)

            batch = [state_im, next_state_im, action, rewards, 1-int(done), goal_im]
            _, error = calculate_cascade_loss(batch, FCQ, CQN, CQN_target, env.goal_type)
            replay_buffer.add(error, [state[0], 0.0], action, [next_state[0], 0.0], rewards, done, state[1])

        else:
            replay_buffer.add([state[0], 0.0], action, [next_state[0], 0.0], rewards, done, state[1])
        ## HER ##
        if her and not done:
            her_sample = sample_her_transitions(info, next_state)
            ig_samples = sample_ig_transitions(info, next_state, num_samples=3)
            samples = her_sample + ig_samples
            for sample in samples:
                rewards_re, goal_image, done_re = sample
                if per:
                    goal_im_re = torch.tensor([goal_image]).type(dtype) # replaced goal
                    batch = [state_im, next_state_im, action, rewards_re, 1-int(done_re), goal_im_re]
                    _, error = calculate_cascade_loss(batch, FCQ, CQN, CQN_target, env.goal_type)
                    replay_buffer.add(error, [state[0], 0.0], action, [next_state[0], 0.0], rewards_re, done_re, goal_image)
                else:
                    replay_buffer.add([state[0], 0.0], action, [next_state[0], 0.0], rewards_re, done_re, goal_image)

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
                torch.FloatTensor(state[0]).type(dtype),
                torch.FloatTensor(next_state[0]).type(dtype),
                torch.FloatTensor(action).type(dtype),
                torch.FloatTensor(np.array(rewards)).type(dtype),
                torch.FloatTensor([1 - done]).type(dtype),
                torch.FloatTensor(state[1]).type(dtype)
                ]
        if per:
            minibatch, idxs, is_weights = replay_buffer.sample(batch_size-1)
            combined_minibatch = combine_batch(minibatch, data)
            loss, error = calculate_cascade_loss(combined_minibatch, FCQ, CQN, CQN_target, env.goal_type)
            errors = error.data.detach().cpu().numpy()[:-1]
            # update priority
            for i in range(batch_size-1):
                idx = idxs[i]
                replay_buffer.update(idx, errors[i])
        else:
            minibatch = replay_buffer.sample(batch_size-1)
            combined_minibatch = combine_batch(minibatch, data)
            loss, _ = calculate_cascade_loss(combined_minibatch, FCQ, CQN, CQN_target, env.goal_type)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        log_minibatchloss.append(loss.data.detach().cpu().numpy())

        state = next_state
        pre_action = action
        ep_len += 1
        t_step += 1
        num_collisions += int(info['collision'])

        if t_step % update_freq == 0:
            CQN_target.load_state_dict(CQN.state_dict())
            lr_scheduler.step()
            epsilon = max(epsilon_decay * epsilon, min_epsilon)

        if done:
            ne += 1
            log_returns.append(episode_reward)
            log_loss.append(np.mean(log_minibatchloss))
            log_eplen.append(ep_len)
            log_epsilon.append(epsilon)
            log_out.append(int(info['out_of_range']))
            log_success.append(int(info['success']))
            log_collisions.append(num_collisions)

            if ne % log_freq == 0:
                log_mean_returns = smoothing_log(log_returns, log_freq)
                log_mean_loss = smoothing_log(log_loss, log_freq)
                log_mean_eplen = smoothing_log(log_eplen, log_freq)
                log_mean_out = smoothing_log(log_out, log_freq)
                log_mean_success = smoothing_log(log_success, log_freq)
                log_mean_collisions = smoothing_log(log_collisions, log_freq)

                print()
                print("{} episodes. ({}/{} steps)".format(ne, t_step, total_steps))
                print("Success rate: {0:.2f}".format(log_mean_success[-1]))
                print("Mean reward: {0:.2f}".format(log_mean_returns[-1]))
                print("Mean loss: {0:.6f}".format(log_mean_loss[-1]))
                # print("Ep reward: {}".format(log_returns[-1]))
                print("Ep length: {}".format(log_mean_eplen[-1]))
                print("Epsilon: {}".format(epsilon))

                axes[0][0].plot(log_loss, color='#ff7f00', linewidth=0.5)
                axes[1][0].plot(log_returns, color='#60c7ff', linewidth=0.5)
                axes[2][0].plot(log_eplen, color='#83dcb7', linewidth=0.5)
                axes[2][1].plot(log_collisions, color='#ff33cc', linewidth=0.5)

                axes[0][0].plot(log_mean_loss, color='red')
                axes[1][0].plot(log_mean_returns, color='blue')
                axes[2][0].plot(log_mean_eplen, color='green')
                axes[0][1].plot(log_mean_success, color='red')
                axes[1][1].plot(log_mean_out, color='black')
                axes[2][1].plot(log_mean_collisions, color='#663399')

                #f.canvas.draw()
                # plt.pause(0.001)
                plt.savefig('results/graph/%s.png' % savename)
                # plt.close()

                numpy_log = np.array([
                    log_returns, #0
                    log_loss, #1
                    log_eplen, #2
                    log_epsilon, #3
                    log_success, #4
                    log_collisions, #5
                    log_out, #6
                    ])
                np.save('results/board/%s' %savename, numpy_log)

                if log_mean_success[-1] > max_success:
                    max_success = log_mean_success[-1]
                    torch.save(CQN.state_dict(), 'results/models/%s.pth' % savename)
                    print("Max performance! saving the model.")

            episode_reward = 0.
            log_minibatchloss = []
            state = env.reset()
            pre_action = None
            ep_len = 0
            num_collisions = 0

    print('Training finished.')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--num_blocks", default=2, type=int)
    parser.add_argument("--dist", default=0.08, type=float)
    parser.add_argument("--max_steps", default=30, type=int)
    parser.add_argument("--camera_height", default=96, type=int)
    parser.add_argument("--camera_width", default=96, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--bs", default=7, type=int)
    parser.add_argument("--buff_size", default=1e3, type=float)
    parser.add_argument("--total_steps", default=1e5, type=float)
    parser.add_argument("--learn_start", default=2e3, type=float)
    parser.add_argument("--update_freq", default=500, type=int)
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--per", action="store_true")
    parser.add_argument("--her", action="store_true")
    parser.add_argument("--reward", default="binary", type=str)
    parser.add_argument("--goal", default="circle", type=str)
    parser.add_argument("--fcn_ver", default=1, type=int)
    parser.add_argument("--sampling", default="uniform", type=str)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--output", default='', type=str)
    ## Evaluate ##
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--model1_path", default="0528_1928", type=str)
    parser.add_argument("--model2_path", default="####_####", type=str)
    parser.add_argument("--num_trials", default=50, type=int)
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
        from models.fcn import FC_QNet_half as FC_QNet
    else:
        from models.fcn import FC_QNet
    output = args.output

    # evaluate configuration #
    evaluation = args.evaluate
    model1_path = os.path.join("results/models/FCDQN_%s.pth"%args.model1_path)
    model2_path = os.path.join("results/models/PCQN_%s.pth"%args.model2_path)
    num_trials = args.num_trials
    visualize_q = args.show_q
    if visualize_q:
        render = True

    now = datetime.datetime.now()
    savename = "PCQN_%s" % (now.strftime("%m%d_%H%M"))
    if not evaluation:
        if not os.path.exists("results/config/"):
            os.makedirs("results/config/")
        with open("results/config/%s.json" % savename, 'w') as cf:
            json.dump(args.__dict__, cf, indent=2)

    env = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
            control_freq=5, data_format='NCHW', xml_ver=0)
    env = pushpixel_env(env, num_blocks=num_blocks, mov_dist=mov_dist, max_steps=max_steps, \
            task=1, reward_type=reward_type, goal_type=goal_type, seperate=True)

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
    sampling = args.sampling  # 'sum' / 'choice' / 'max'

    if goal_type=="pixel":
        in_channel1 = 4 # 3+1
        in_channel2 = 13 # 3+2+8
    else:
        in_channel1 = 6 # 3+3
        in_channel2 = 14 # 3+3+8
    in_channel = [in_channel1, in_channel2]
            
    if evaluation:
        evaluate(env=env, n_blocks=num_blocks, in_channel=in_channel, model1=model1_path, \
                model2=model2_path, num_trials=num_trials, visualize_q=visualize_q, output=output)
    else:
        learning(env=env, savename=savename, n_blocks=num_blocks, in_channel=in_channel, \
                learning_rate=learning_rate, batch_size=batch_size, buff_size=buff_size, \
                total_steps=total_steps, learn_start=learn_start, update_freq=update_freq, \
                log_freq=log_freq, double=double, her=her, per=per, visualize_q=visualize_q, \
                goal_type=goal_type, model1=model1_path, sampling=sampling, output=output)
