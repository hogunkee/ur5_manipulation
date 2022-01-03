import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../../ur5_mujoco'))
from object_env import *

from training_utils import *

import torch
import torch.nn as nn
import argparse
import json

import time
import datetime
import random

from sdf_module import SDFModule
from replay_buffer import ReplayBuffer, PER
from matplotlib import pyplot as plt

#dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pad_sdf(sdf, nmax):
    h, w = 96, 96
    nsdf = len(sdf)
    padded = np.zeros([nmax, h, w])
    if nsdf > 0:
        padded[:nsdf] = sdf
    return padded

def get_action(env, qnet, sdf_raw, sdfs, epsilon, with_q=False):
    if np.random.random() < epsilon:
        #print('Random action')
        obj = np.random.randint(len(sdf_raw))
        theta = np.random.randint(env.num_bins)
        if with_q:
            nsdf = sdfs[0].shape[0]
            s = pad_sdf(sdfs[0], env.num_blocks+2)
            s = torch.FloatTensor(s).to(device).unsqueeze(0)
            g = pad_sdf(sdfs[1], env.num_blocks+2)
            g = torch.FloatTensor(g).to(device).unsqueeze(0)
            nsdf = torch.LongTensor([nsdf]).to(device)
            q_value = qnet([s, g], nsdf)
            q = q_value[0][:nsdf].detach().cpu().numpy()
    else:
        nsdf = sdfs[0].shape[0]
        s = pad_sdf(sdfs[0], env.num_blocks+2)
        s = torch.FloatTensor(s).to(device).unsqueeze(0)
        g = pad_sdf(sdfs[1], env.num_blocks+2)
        g = torch.FloatTensor(g).to(device).unsqueeze(0)
        nsdf = torch.LongTensor([nsdf]).to(device)
        q_value = qnet([s, g], nsdf)
        q = q_value[0][:nsdf].detach().cpu().numpy()

        obj = q.max(1).argmax()
        theta = q.max(0).argmax()

    action = [obj, theta]
    sdf_target = sdf_raw[obj]
    px, py = np.where(sdf_target==sdf_target.max())
    px = px[0]
    py = py[0]
    #print(px, py, theta)

    if with_q:
        return action, [px, py, theta], q
    else:
        return action, [px, py, theta]

def learning(env, 
        savename,
        sdf_module,
        n_actions=8,
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
        ig=True,
        visualize_q=False,
        pretrain=False,
        continue_learning=False,
        model_path=''
        ):

    qnet = QNet(env.num_blocks + 2, n_actions).to(device)
    if pretrain:
        qnet.load_state_dict(torch.load(model_path))
        print('Loading pre-trained model: {}'.format(model_path))
    elif continue_learning:
        qnet.load_state_dict(torch.load(model_path))
        print('Loading trained model: {}'.format(model_path))
    qnet_target = QNet(env.num_blocks + 2, n_actions).to(device)
    qnet_target.load_state_dict(qnet.state_dict())

    #optimizer = torch.optim.SGD(qnet.parameters(), lr=learning_rate, momentum=0.9, weight_decay=2e-5)
    optimizer = torch.optim.Adam(qnet.parameters(), lr=learning_rate)

    if per:
        replay_buffer = PER([env.num_blocks+2, 96, 96], [env.num_blocks+2, 96, 96], max_size=int(buff_size))
    else:
        replay_buffer = ReplayBuffer([env.num_blocks+2, 96, 96], [env.num_blocks+2, 96, 96], max_size=int(buff_size))

    model_parameters = filter(lambda p: p.requires_grad, qnet.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("# of params: %d"%params)


    if double:
        calculate_loss = calculate_loss_gcn_double
    else:
        calculate_loss = calculate_loss_gcn_origin

    if continue_learning and not pretrain:
        numpy_log = np.load(model_path.replace('models/', 'board/').replace('.pth', '.npy'), allow_pickle=True)
        log_returns = list(numpy_log[0])
        log_loss = list(numpy_log[1])
        log_eplen = list(numpy_log[2])
        log_epsilon = list(numpy_log[3])
        log_success = list(numpy_log[4])
        #log_collisions = list(numpy_log[5])
        log_sdf_mismatch = list(numpy_log[5])
        log_out = list(numpy_log[6])
        log_success_block = list(numpy_log[7])
    else:
        log_returns = []
        log_loss = []
        log_eplen = []
        log_epsilon = []
        log_success = []
        #log_collisions = []
        log_sdf_mismatch= []
        log_out = []
        log_success_block = [[], [], []]
        log_mean_success_block = [[], [], []]
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
    #axes[2][2].set_title('Num Collisions')  # 9
    axes[2][2].set_title('SDF mismatch')  # 9

    #lr_decay = 0.98
    #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)

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
    st = time.time()

    (state_img, goal_img) = env.reset()
    sdf_st, sdf_raw, feature_st = sdf_module.get_sdf_features(state_img)
    sdf_g, _, feature_g = sdf_module.get_sdf_features(goal_img)
    matching = sdf_module.object_matching(feature_st, feature_g)
    sdf_st_align = sdf_st[matching]
    sdf_raw = sdf_raw[matching]

    mismatch = len(sdf_st_align)!=env.num_blocks or len(sdf_g)!=env.num_blocks
    num_mismatch = int(mismatch) 

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

        s0 = deepcopy(state_goal[0]).transpose([1, 2, 0])
        s1 = deepcopy(state_goal[1]).reshape(96, 96)
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
        action, pixel_action, q_map = get_action(env, qnet, sdf_raw, [sdf_st_align, sdf_g], epsilon=epsilon, with_q=True)
        if visualize_q:
            s0 = deepcopy(state_goal[0]).transpose([1, 2, 0])
            s1 = deepcopy(state_goal[1]).reshape(96, 96)
            ax0.imshow(s1)
            ax3.imshow(s0[:, :, 0])
            ax4.imshow(s0[:, :, 1])
            ax5.imshow(s0[:, :, 2])

            s0[pixel_action[0], pixel_action[1]] = [1, 0, 0]
            s0[s0[:, :, 0].astype(bool)] = [1, 0, 0]
            s0[s0[:, :, 1].astype(bool)] = [0, 1, 0]
            ax1.imshow(s0)
            q_map = q_map.transpose([1,2,0]).max(2)
            ax2.imshow(q_map/q_map.max())
            #print('min_q:', q_map.min(), '/ max_q:', q_map.max())
            fig.canvas.draw()

        (next_state_img, _), reward, done, info = env.step(pixel_action)
        episode_reward += reward
        sdf_ns, sdf_raw, feature_ns = sdf_module.get_sdf_features(next_state_img)
        matching = sdf_module.object_matching(feature_ns, feature_g)
        sdf_ns_align = sdf_ns[matching]
        sdf_raw = sdf_raw[matching]

        ## save transition to the replay buffer ##
        mismatch = len(sdf_st_align)!=env.num_blocks or len(sdf_ns_align)!=env.num_blocks
        num_mismatch += int(mismatch) 
        if per:
            trajectories = []
            replay_tensors = []

            trajectories.append([sdf_st_align, action, sdf_ns_align, reward, done, sdf_g])

            traj_tensor = [
                torch.FloatTensor(pad_sdf(sdf_st_align, env.num_blocks+2)).to(device),
                torch.FloatTensor(pad_sdf(sdf_ns_align, env.num_blocks+2)).to(device),
                torch.FloatTensor(action).to(device),
                torch.FloatTensor([reward]).to(device),
                torch.FloatTensor([1 - done]).to(device),
                torch.FloatTensor(pad_sdf(sdf_g, env.num_blocks+2)).to(device),
                torch.LongTensor([len(sdf_st_align)]).to(device),
                torch.LongTensor([len(sdf_ns_align)]).to(device),
            ]
            replay_tensors.append(traj_tensor)

            ## HER ##
            if not done:
                her_sample = sample_her_transitions(env, info)
                for sample in her_sample:
                    reward_re, goal_re, done_re, block_success_re = sample

                    trajectories.append([sdf_st_align, action, sdf_ns_align, reward_re, done_re, sdf_ns_align])
                    traj_tensor = [
                        torch.FloatTensor(pad_sdf(sdf_st_align, env.num_blocks+2)).to(device),
                        torch.FloatTensor(pad_sdf(sdf_ns_align, env.num_blocks+2)).to(device),
                        torch.FloatTensor(action).to(device),
                        torch.FloatTensor([reward_re]).to(device),
                        torch.FloatTensor([1 - done_re]).to(device),
                        torch.FloatTensor(pad_sdf(sdf_ns_align, env.num_blocks+2)).to(device),
                        torch.LongTensor([len(sdf_st_align)]).to(device),
                        torch.LongTensor([len(sdf_ns_align)]).to(device),
                    ]
                    replay_tensors.append(traj_tensor)

            minibatch = None
            for data in replay_tensors:
                minibatch = combine_batch(minibatch, data)
            _, error = calculate_loss(minibatch, qnet, qnet_target)
            error = error.data.detach().cpu().numpy()
            for i, traj in enumerate(trajectories):
                replay_buffer.add(error[i], *traj)

        else:
            trajectories = []
            trajectories.append([sdf_st_align, action, sdf_ns_align, reward, done, sdf_g])

            ## HER ##
            if her and not done:
                her_sample = sample_her_transitions(env, info)
                for sample in her_sample:
                    reward_re, goal_re, done_re, block_success_re = sample
                    trajectories.append([sdf_st_align, action, sdf_ns_align, reward_re, done_re, sdf_ns_align])

            for traj in trajectories:
                replay_buffer.add(*traj)

        if t_step < learn_start:
            if done:
                (state_img, goal_img) = env.reset()
                sdf_st, sdf_raw, feature_st = sdf_module.get_sdf_features(state_img)
                sdf_g, _, feature_g = sdf_module.get_sdf_features(goal_img)
                matching = sdf_module.object_matching(feature_st, feature_g)
                sdf_st_align = sdf_st[matching]
                sdf_raw = sdf_raw[matching]

                mismatch = len(sdf_st_align)!=env.num_blocks or len(sdf_ns_align)!=env.num_blocks
                num_mismatch = int(mismatch) 
                episode_reward = 0.
            else:
                sdf_st_align = sdf_ns_align
                #sdf_state_goal = next_sdf_state_goal
            learn_start -= 1
            if learn_start==0:
                epsilon = start_epsilon
            continue

        ## sample from replay buff & update networks ##
        data = [
                torch.FloatTensor(pad_sdf(sdf_st_align, env.num_blocks+2)).to(device),
                torch.FloatTensor(pad_sdf(sdf_ns_align, env.num_blocks+2)).to(device),
                torch.FloatTensor(action).to(device),
                torch.FloatTensor([reward]).to(device),
                torch.FloatTensor([1 - done]).to(device),
                torch.FloatTensor(pad_sdf(sdf_g, env.num_blocks+2)).to(device),
                torch.LongTensor([len(sdf_st_align)]).to(device),
                torch.LongTensor([len(sdf_ns_align)]).to(device),
                ]
        if per:
            minibatch, idxs, is_weights = replay_buffer.sample(batch_size-1)
            combined_minibatch = combine_batch(minibatch, data)
            loss, error = calculate_loss(combined_minibatch, qnet, qnet_target)
            errors = error.data.detach().cpu().numpy()[:-1]
            # update priority
            for i in range(batch_size-1):
                idx = idxs[i]
                replay_buffer.update(idx, errors[i])
        else:
            minibatch = replay_buffer.sample(batch_size-1)
            combined_minibatch = combine_batch(minibatch, data)
            loss, _ = calculate_loss(combined_minibatch, qnet, qnet_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        log_minibatchloss.append(loss.data.detach().cpu().numpy())

        sdf_st_align = sdf_ns_align
        #sdf_state_goal = sdf_next_state_goal
        ep_len += 1
        t_step += 1
        #num_collisions += int(info['collision'])
        num_mismatch += 1

        if done:
            ne += 1
            log_returns.append(episode_reward)
            log_loss.append(np.mean(log_minibatchloss))
            log_eplen.append(ep_len)
            log_epsilon.append(epsilon)
            log_out.append(int(info['out_of_range']))
            log_success.append(int(info['success']))
            #log_collisions.append(num_collisions)
            log_sdf_mismatch.append(num_mismatch)

            for o in range(env.num_blocks):
                log_success_block[o].append(int(info['block_success'][o]))

            if ne % log_freq == 0:
                log_mean_returns = smoothing_log_same(log_returns, log_freq)
                log_mean_loss = smoothing_log_same(log_loss, log_freq)
                log_mean_eplen = smoothing_log_same(log_eplen, log_freq)
                log_mean_out = smoothing_log_same(log_out, log_freq)
                log_mean_success = smoothing_log_same(log_success, log_freq)
                for o in range(env.num_blocks):
                    log_mean_success_block[o] = smoothing_log_same(log_success_block[o], log_freq)
                #log_mean_collisions = smoothing_log_same(log_collisions, log_freq)
                log_mean_sdf_mismatch = smoothing_log_same(log_sdf_mismatch, log_freq)

                et = time.time()
                print()
                print("{} episodes. ({}/{} steps) - {} seconds".format(ne, t_step, total_steps, et - st))
                print("Success rate: {0:.2f}".format(log_mean_success[-1]))
                for o in range(env.num_blocks):
                    print("Block {0}: {1:.2f}".format(o+1, log_mean_success_block[o][-1]))
                print("Mean reward: {0:.2f}".format(log_mean_returns[-1]))
                print("Mean loss: {0:.6f}".format(log_mean_loss[-1]))
                print("Ep length: {}".format(log_mean_eplen[-1]))
                print("Epsilon: {}".format(epsilon))

                axes[1][2].plot(log_loss, color='#ff7f00', linewidth=0.5)  # 3->6
                axes[1][1].plot(log_returns, color='#60c7ff', linewidth=0.5)  # 5
                axes[2][0].plot(log_eplen, color='#83dcb7', linewidth=0.5)  # 7
                #axes[2][2].plot(log_collisions, color='#ff33cc', linewidth=0.5)  # 8->9

                for o in range(env.num_blocks):
                    axes[0][o].plot(log_mean_success_block[o], color='red')  # 1,2,3

                axes[1][2].plot(log_mean_loss, color='red')  # 3->6
                axes[1][1].plot(log_mean_returns, color='blue')  # 5
                axes[2][0].plot(log_mean_eplen, color='green')  # 7
                axes[1][0].plot(log_mean_success, color='red')  # 4
                axes[2][1].plot(log_mean_out, color='black')  # 6->8
                #axes[2][2].plot(log_mean_collisions, color='#663399')  # 8->9
                axes[2][2].plot(log_mean_sdf_mismatch, color='#663399')  # 8->9

                f.savefig('results/graph/%s.png' % savename)

                log_list = [
                        log_returns,  # 0
                        log_loss,  # 1
                        log_eplen,  # 2
                        log_epsilon,  # 3
                        log_success,  # 4
                        log_sdf_mismatch, #log_collisions,  # 5
                        log_out,  # 6
                        log_success_block, #7
                        ]
                numpy_log = np.array(log_list)
                np.save('results/board/%s' %savename, numpy_log)

                if log_mean_success[-1] > max_success:
                    max_success = log_mean_success[-1]
                    torch.save(qnet.state_dict(), 'results/models/%s.pth' % savename)
                    print("Max performance! saving the model.")

            (state_img, goal_img) = env.reset()
            sdf_st, sdf_raw, feature_st = sdf_module.get_sdf_features(state_img)
            sdf_g, _, feature_g = sdf_module.get_sdf_features(goal_img)
            matching = sdf_module.object_matching(feature_st, feature_g)
            sdf_st_align = sdf_st[matching]
            sdf_raw = sdf_raw[matching]
            mismatch = len(sdf_st_align)!=env.num_blocks or len(sdf_g)!=env.num_blocks

            episode_reward = 0.
            log_minibatchloss = []
            ep_len = 0
            num_mismatch = int(mismatch) 

            if ne % update_freq == 0:
                qnet_target.load_state_dict(qnet.state_dict())
                #lr_scheduler.step()
                epsilon = max(epsilon_decay * epsilon, min_epsilon)


    print('Training finished.')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--num_blocks", default=3, type=int)
    parser.add_argument("--dist", default=0.06, type=float)
    parser.add_argument("--max_steps", default=100, type=int)
    parser.add_argument("--camera_height", default=480, type=int)
    parser.add_argument("--camera_width", default=480, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--bs", default=6, type=int)
    parser.add_argument("--buff_size", default=1e3, type=float)
    parser.add_argument("--total_steps", default=2e5, type=float)
    parser.add_argument("--learn_start", default=300, type=float)
    parser.add_argument("--update_freq", default=100, type=int)
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument("--double", action="store_false") # default: True
    parser.add_argument("--per", action="store_false") # default: True
    parser.add_argument("--her", action="store_false") # default: True
    parser.add_argument("--ig", action="store_false") # default: True
    parser.add_argument("--ver", default=1, type=int)
    parser.add_argument("--reward", default="new", type=str)
    parser.add_argument("--pretrain", action="store_true")
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
    num_blocks = args.num_blocks
    mov_dist = args.dist
    max_steps = args.max_steps
    camera_height = args.camera_height
    camera_width = args.camera_width
    reward_type = args.reward

    model_path = os.path.join("results/models/SDF_%s.pth"%args.model_path)
    visualize_q = args.show_q
    if visualize_q:
        render = True

    now = datetime.datetime.now()
    savename = "SDF_%s" % (now.strftime("%m%d_%H%M"))
    if not os.path.exists("results/config/"):
        os.makedirs("results/config/")
    with open("results/config/%s.json" % savename, 'w') as cf:
        json.dump(args.__dict__, cf, indent=2)

    sdf_module = SDFModule()
    env = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
            control_freq=5, data_format='NHWC', xml_ver=0)
    env = objectwise_env(env, num_blocks=num_blocks, mov_dist=mov_dist,max_steps=max_steps,\
            conti=False, detection=True, reward_type=reward_type)

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
    ig = args.ig
    ver = args.ver

    pretrain = args.pretrain
    continue_learning = args.continue_learning
    if ver==1:
        from models.sdf_gcn import SDFGCNQNet as QNet
    elif ver==2:
        # ver2: separate edge
        from models.sdf_gcn import SDFGCNQNetV2 as QNet
    elif ver==3:
        # ver3: block flags - 1 for block's sdf, 0 for goal's sdf
        from models.sdf_gcn import SDFGCNQNetV3 as QNet

    learning(env=env, savename=savename, sdf_module=sdf_module, n_actions=8, \
            learning_rate=learning_rate, batch_size=batch_size, buff_size=buff_size, \
            total_steps=total_steps, learn_start=learn_start, update_freq=update_freq, \
            log_freq=log_freq, double=double, her=her, ig=ig, per=per, visualize_q=visualize_q, \
            continue_learning=continue_learning, model_path=model_path, pretrain=pretrain)
