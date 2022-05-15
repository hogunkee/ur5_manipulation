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
            state_tensor = torch.tensor([state[0]]).cuda()
            goal_tensor = torch.tensor([state[1]]).cuda()
            state_goal = torch.cat((state_tensor, goal_tensor), 1)
            q_value = fc_qnet(state_goal)
            q_raw = q_value[0].detach().cpu().numpy()
            q = np.zeros_like(q_raw)
            q[:, crop_min:crop_max, crop_min:crop_max] = q_raw[:, crop_min:crop_max, crop_min:crop_max]
    else:
        state_tensor = torch.tensor([state[0]]).cuda()
        goal_tensor = torch.tensor([state[1]]).cuda()
        state_goal = torch.cat((state_tensor, goal_tensor), 1)
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


def evaluate(env, n_actions=8, ver=0, model_path='', num_trials=10, visualize_q=False, n1=1, n2=1):
    if ver==0:
        s_dim = 3
    elif ver==1:
        s_dim = 1
    elif ver==2:
        s_dim = 1
    elif ver==3:
        s_dim = 2
    elif ver==4:
        s_dim = 5
    FCQ = FCQNet(n_actions, s_dim).cuda()
    print('Loading trained model: {}'.format(model_path))
    FCQ.load_state_dict(torch.load(model_path))

    log_returns = []
    log_eplen = []
    log_out = []
    log_success = []
    log_success_1block = []
    log_collisions = []

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
        s1 = deepcopy(state[1]).transpose([1, 2, 0])
        im0 = ax0.imshow(s1)
        im = ax1.imshow(s0)
        im2 = ax2.imshow(np.zeros_like(s0))
        plt.show(block=False)
        fig.canvas.draw()
        fig.canvas.draw()

    for ne in range(num_trials):
        env.set_num_blocks(np.random.choice(range(n1, n2+1)))
        ep_len = 0
        episode_reward = 0.
        num_collisions = 0

        state, info = env.reset()
        pre_action = None

        for t_step in range(env.max_steps):
            ep_len += 1
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
            num_collisions += int(info['collision'])

            if done:
                break
            else:
                state = next_state
                pre_action = action

        log_returns.append(episode_reward)
        log_eplen.append(ep_len)
        log_out.append(int(info['out_of_range']))
        log_success.append(int(info['success']))
        log_success_1block.append(np.mean(info['block_success']))
        log_collisions.append(num_collisions)

        print("EP{}".format(ne+1), end=" / ")
        print("reward:{0:.2f}".format(log_returns[-1]), end=" / ")
        print("eplen:{0:.1f}".format(log_eplen[-1]), end=" / ")
        print("SR:{0:.2f} ({1}/{2})".format(np.mean(log_success),
                np.sum(log_success), len(log_success)), end="")
        print(" / 1BSR:{0:.2f}".format(np.mean(log_success_1block)), end="")
        print(" / mean reward:{0:.1f}".format(np.mean(log_returns)), end="")
        print(" / mean eplen:{0:.1f}".format(np.mean(log_eplen)), end="")
        print(" / oor:{0:.2f}".format(np.mean(log_out)), end="")
        print(" / collisions:{0:.1f}".format(np.mean(log_collisions)))

    print()
    print("="*80)
    print("Evaluation Done.")
    print("Mean reward: {0:.2f}".format(np.mean(log_returns)))
    print("Mean episode length: {}".format(np.mean(log_eplen)))
    print("Success rate: {}".format(100*np.mean(log_success)))
    print("1BSR: {}".format(100*np.mean(log_success_1block)))

def learning(env, 
        savename,
        n_actions=8,
        ver=0,
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
        continue_learning=False,
        model_path='',
        wandb_off=False,
        n1=1,
        n2=1,
        ):

    if ver==0:
        s_dim = 3
    elif ver==1:
        s_dim = 1
    elif ver==2:
        s_dim = 1
    elif ver==3:
        s_dim = 2
    elif ver==4:
        s_dim = 5
    FCQ = FCQNet(n_actions, s_dim).cuda()
    if continue_learning:
        FCQ.load_state_dict(torch.load(model_path))
    FCQ_target = FCQNet(n_actions, s_dim).cuda()
    FCQ_target.load_state_dict(FCQ.state_dict())

    # criterion = nn.SmoothL1Loss(reduction=None).cuda()
    # criterion = nn.MSELoss(reduction='mean')
    #optimizer = torch.optim.SGD(FCQ.parameters(), lr=learning_rate, momentum=0.9, weight_decay=2e-5)
    optimizer = torch.optim.Adam(FCQ.parameters(), lr=learning_rate)

    if per:
        replay_buffer = PER([s_dim, env.env.camera_height, env.env.camera_width], \
                    [s_dim, env.env.camera_height, env.env.camera_width], dim_action=3, \
                    max_size=int(buff_size))
    else:
        replay_buffer = ReplayBuffer([s_dim, env.env.camera_height, env.env.camera_width], \
                [s_dim, env.env.camera_height, env.env.camera_width], dim_action=3, \
                max_size=int(buff_size))

    model_parameters = filter(lambda p: p.requires_grad, FCQ.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("# of params: %d"%params)

    if double:
        calculate_loss = calculate_loss_double_fcdqn
    else:
        calculate_loss = calculate_loss_fcdqn

    if continue_learning:
        numpy_log = np.load(model_path.replace('models/', 'board/').replace('.pth', '.npy'))
        log_returns = numpy_log[0].tolist()
        log_loss = numpy_log[1].tolist()
        log_eplen = numpy_log[2].tolist()
        log_epsilon = numpy_log[3].tolist()
        log_success_total = numpy_log[4].tolist()
        log_collisions = numpy_log[5].tolist()
        log_out = numpy_log[6].tolist()
        log_success_1block = numpy_log[7].tolist()
        log_success = numpy_log[8].tolist()
    else:
        log_returns = []
        log_loss = []
        log_eplen = []
        log_epsilon = []
        log_success_total = []
        log_collisions = []
        log_out = []
        log_success_1block = []
        log_success = []

    if not os.path.exists("results/models/"):
        os.makedirs("results/models/")
    if not os.path.exists("results/board/"):
        os.makedirs("results/board/")

    plt.show(block=False)
    plt.rc('axes', labelsize=6)
    plt.rc('font', size=6)

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
    max_success = 0.0
    st = time.time()

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

    count_steps = 0
    for ne in range(total_episodes):
        _env = env[ne%len(env)]
        if mujoco_py.__version__=='2.0.2.13':
            _env.env.reset_viewer()
        _env.set_num_blocks(np.random.choice(range(n1, n2+1)))
        ep_len = 0
        episode_reward = 0.
        num_collisions = 0
        log_minibatchloss = []

        state, info = env.reset()
        pre_action = None

        for t_step in range(_env.max_steps):
            count_steps += 1
            ep_len += 1
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
            num_collisions += int(info['collision'])

            ## save transition to the replay buffer ##
            if per:
                state_tensor = torch.tensor([state[0]]).cuda()
                goal_tensor = torch.tensor([state[1]]).cuda()
                next_state_tensor = torch.tensor([next_state[0]]).cuda()
                action_tensor = torch.tensor([action]).cuda()

                batch = [state_tensor, next_state_tensor, action_tensor, reward, 1-int(done), goal_tensor]
                _, error = calculate_loss(batch, FCQ, FCQ_target)
                error = error.data.detach().cpu().numpy()
                replay_buffer.add(error, [state[0], 0.0], action, [next_state[0], 0.0], reward, done, state[1])

            else:
                replay_buffer.add([state[0], 0.0], action, [next_state[0], 0.0], reward, done, state[1])
            ## HER ##
            if her and not done:
                her_sample = sample_her_transitions(env, info, next_state)
                for sample in her_sample:
                    reward_re, achieved_goal, done_re, block_success_re = sample
                    if per:
                        goal_tensor_re = torch.tensor([achieved_goal]).cuda() # replaced goal
                        batch = [state_tensor, next_state_tensor, action_tensor, reward_re, 1-int(done_re), goal_tensor_re]
                        _, error = calculate_loss(batch, FCQ, FCQ_target)
                        error = error.data.detach().cpu().numpy()
                        replay_buffer.add(error, [state[0], 0.0], action, [next_state[0], 0.0], reward_re, done_re, achieved_goal)
                    else:
                        replay_buffer.add([state[0], 0.0], action, [next_state[0], 0.0], reward_re, done_re, achieved_goal)

            if replay_buffer.size < learn_start:
                if done:
                    break
                else:
                    state = next_state
                    pre_action = action
                    continue
            elif replay_buffer.size == learn_start:
                epsilon = start_epsilon
                count_steps = 0
                break

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

            if done:
                break
            else:
                state = next_state
                pre_action = action

        if replay_buffer.size <= learn_start:
            continue

        log_returns.append(episode_reward)
        log_loss.append(np.mean(log_minibatchloss))
        log_eplen.append(ep_len)
        log_epsilon.append(epsilon)
        log_out.append(int(info['out_of_range']))
        log_success_total.append(int(info['success']))
        log_success_1block.append(np.mean(info['block_success']))
        if not _env.num_blocks in log_success:
            log_success[_env.num_blocks] = []
        log_success[_env.num_blocks].append(int(info['success']))
        log_collisions.append(num_collisions)

        eplog = {
                '%dB Reward'%_env.num_blocks: episode_reward,
                'loss': np.mean(log_minibatchloss),
                '%dB EP Len'%_env.num_blocks: ep_len,
                'epsilon': epsilon,
                '%dB OOR'%_env.num_blocks: int(info['out_of_range']),
                'success rate': int(info['success']),
                '1block success': np.mean(info['block_success']),
                '%dB SR'%_env.num_blocks: int(info['success']),
                'collision': num_collisions,
                }
        if not wandb_off:
            wandb.log(eplog, count_steps)

        if ne % log_freq == 0:
            log_mean_returns = smoothing_log(log_returns, log_freq)
            log_mean_loss = smoothing_log(log_loss, log_freq)
            log_mean_eplen = smoothing_log(log_eplen, log_freq)
            log_mean_out = smoothing_log(log_out, log_freq)
            log_mean_success = smoothing_log(log_success, log_freq)
            log_mean_success_block = smoothing_log_same(log_success_1block, log_freq)
            log_mean_collisions = smoothing_log(log_collisions, log_freq)

            et = time.time()
            now = datetime.datetime.now().strftime("%m/%d %H:%M")
            interval = str(datetime.timedelta(0, int(et-st)))
            st = et
            print(f"{now}({interval}) / ep{ne} ({count_steps} steps)", end=" / ")
            print(f"SR:{log_mean_success[-1]:.2f}", end=" / ")
            print("1BSR:{0:.2f}".format(log_mean_success_block[-1]), end=" ")
            print("/ Reward:{0:.2f}".format(log_mean_returns[-1]), end="")
            print(" / Loss:{0:.5f}".format(log_mean_loss[-1]), end="")
            print(" / Eplen:{0:.1f}".format(log_mean_eplen[-1]), end="")
            print(" / OOR:{0:.2f}".format(log_mean_out[-1]), end="")
            print(" / Collision:{0:.1f}".format(log_mean_collisions[-1]), end="")

            log_list = [
                    log_returns,  # 0
                    log_loss,  # 1
                    log_eplen,  # 2
                    log_epsilon,  # 3
                    log_success_total,  # 4
                    log_collisions,  # 5
                    log_out,  # 6
                    log_success_b1 #7
                    log_success, #8
                    ]
            numpy_log = np.array(log_list, dtype=object)
            np.save('results/board/%s' %savename, numpy_log)

            if log_mean_success[-1] > max_success:
                max_success = log_mean_success[-1]
                torch.save(FCQ.state_dict(), 'results/models/%s.pth' % savename)
                print(" <- Highest SR. Saving the model.")
            else:
                print("")

        if ne % update_freq == 0:
            FCQ_target.load_state_dict(FCQ.state_dict())
            #lr_scheduler.step()
            epsilon = max(epsilon_decay * epsilon, min_epsilon)

    print('Training finished.')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    ## env ##
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--n1", default=3, type=int)
    parser.add_argument("--n2", default=3, type=int)
    parser.add_argument("--dist", default=0.08, type=float)
    parser.add_argument("--max_steps", default=30, type=int)
    parser.add_argument("--camera_height", default=96, type=int)
    parser.add_argument("--camera_width", default=96, type=int)
    ## ablation##
    # ver.0: RGB (sdim: 3)
    # ver.1: Depth (sdim: 1)
    # ver.2: SDF (sdim: 1)
    # ver.3: SDF+Depth (sdim: 2)
    # ver.4: SDF+RGB+Depth (sdim: 5)
    parser.add_argument("--ver", default=0, type=int)
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
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--continue_learning", action="store_true")
    ## Evaluate ##
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--model_path", default="####_####", type=str)
    parser.add_argument("--num_trials", default=50, type=int)
    # etc #
    parser.add_argument("--show_q", action="store_true")
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--wandb_off", action="store_true")
    args = parser.parse_args()

    # env configuration #
    render = args.render
    n1 = args.n1
    n2 = args.n2
    mov_dist = args.dist
    max_steps = args.max_steps
    camera_height = args.camera_height
    camera_width = args.camera_width
    reward_type = args.reward

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
    env = pushpixel_env(env, num_blocks=n1, mov_dist=mov_dist, max_steps=max_steps, \
            task=1, reward_type=reward_type, goal_type='block')

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

    ver = args.ver
            
    if evaluation:
        evaluate(env=env, n_actions=8, ver=ver, model_path=model_path, \
                num_trials=num_trials, visualize_q=visualize_q, n1=n1, n2=n2)
    else:
        learning(env=env, savename=savename, n_actions=8, ver=ver, \
                learning_rate=learning_rate, batch_size=batch_size, buff_size=buff_size, \
                total_steps=total_steps, learn_start=learn_start, update_freq=update_freq, \
                log_freq=log_freq, double=double, her=her, per=per, visualize_q=visualize_q, \
                continue_learning=continue_learning, \
                model_path=model_path, wandb_off=wandb_off, n1=n1, n2=n2)
