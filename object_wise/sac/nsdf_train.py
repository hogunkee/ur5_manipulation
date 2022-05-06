import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../../ur5_mujoco'))
sys.path.append(os.path.join(FILE_PATH, '..'))
from object_env import *
from training_utils import *
from skimage import color
from PIL import Image

import torch
import torch.nn as nn
import argparse
import json

import copy
import time
import datetime
import random
import pylab

from sac import SAC
from sdf_module import SDFModule
from replay_buffer import ReplayBuffer
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import wandb

#dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def norm_npy(array):
    positive = array - array.min()
    return positive / positive.max()

def pad_sdf(sdf, nmax, res=96):
    nsdf = len(sdf)
    padded = np.zeros([nmax, res, res])
    if nsdf > nmax:
        padded[:] = sdf[:nmax]
    elif nsdf > 0:
        padded[:nsdf] = sdf
    return padded

def get_sdf_center_mask(env, sdf_raw, sidx):
    cx, cy = env.get_center_from_sdf(sdf_raw[sidx], None)
    mask = None
    masks = []
    for s in sdf_raw:
        m = copy.deepcopy(s)
        m[m<0] = 0
        m[m>0] = 1
        masks.append(m)
    mask = np.sum(masks, 0)
    return (cx, cy), mask

def learning(env, agent, sdf_module, savename, args):
    if sdf_module.resize:
        sdf_res = 96
    else:
        sdf_res = 480

    replay_buffer = ReplayBuffer([args.max_blocks, sdf_res, sdf_res], [args.max_blocks, sdf_res, sdf_res], max_size=int(args.buff_size))

    actor_parameters = filter(lambda p: p.requires_grad, agent.policy.parameters())
    actor_params = sum([np.prod(p.size()) for p in actor_parameters])
    print("# of actor params: %d"%actor_params)
    critic_parameters = filter(lambda p: p.requires_grad, agent.critic.parameters())
    critic_params = sum([np.prod(p.size()) for p in critic_parameters])
    print("# of critic params: %d"%critic_params)

    if args.continue_learning and not args.pretrain:
        numpy_log = np.load(model_path.replace('models/', 'board/').replace('.pth', '.npy'), allow_pickle=True)
        log_returns = list(numpy_log[0])
        log_critic_loss = list(numpy_log[1])
        log_actor_loss = list(numpy_log[2])
        log_eplen = list(numpy_log[3])
        log_success_total = list(numpy_log[4])
        log_out = list(numpy_log[5])
        log_success_1block = list(numpy_log[6])
        log_success = dict(numpy_log[7])
    else:
        log_returns = []
        log_critic_loss = []
        log_actor_loss = []
        log_eplen = []
        log_success_total = []
        log_out = []
        log_success_1block = []
        log_success = {}

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

    #lr_decay = 0.98
    #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)

    updates = 0
    max_success = 0.0
    st = time.time()

    if visualize_q:
        cm = pylab.get_cmap('gist_rainbow')
        fig = plt.figure()
        ax0 = fig.add_subplot(221)
        ax1 = fig.add_subplot(222)
        ax2 = fig.add_subplot(223)
        ax3 = fig.add_subplot(224)
        ax0.set_title('Goal')
        ax1.set_title('Observation')
        ax2.set_title('Goal SDF')
        ax3.set_title('Current SDF')
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax3.set_xticks([])
        ax3.set_yticks([])

        plt.show(block=False)
        fig.canvas.draw()

    count_steps = 0
    for ne in range(int(args.total_episodes)):
        _env = env[ne%len(env)]
        _env.set_num_blocks(np.random.choice(range(args.n1, args.n2+1)))
        ep_len = 0
        episode_reward = 0.
        log_minibatch_critic_loss = []
        log_minibatch_actor_loss = []

        check_env_ready = False
        while not check_env_ready:
            (state_img, goal_img), info = _env.reset()
            sdf_st, sdf_raw, feature_st = sdf_module.get_sdf_features_with_ucn(state_img[0], state_img[1], _env.num_blocks, clip=args.clip_sdf)
            sdf_g, _, feature_g = sdf_module.get_sdf_features_with_ucn(goal_img[0], goal_img[1], _env.num_blocks, clip=args.clip_sdf)
            if args.round_sdf:
                sdf_g = sdf_module.make_round_sdf(sdf_g)
            check_env_ready = (len(sdf_g)==_env.num_blocks) & (len(sdf_st)==_env.num_blocks)

        n_detection = len(sdf_st)
        # target: st / source: g
        if args.oracle:
            sdf_st_align = sdf_module.oracle_align(sdf_st, info['pixel_poses'])
            sdf_raw = sdf_module.oracle_align(sdf_raw, info['pixel_poses'], scale=1)
            sdf_g = sdf_module.oracle_align(sdf_g, info['pixel_goals'])
        else:
            matching = sdf_module.object_matching(feature_st, feature_g)
            sdf_st_align = sdf_module.align_sdf(matching, sdf_st, sdf_g)
            sdf_raw = sdf_module.align_sdf(matching, sdf_raw, np.zeros([_env.num_blocks, *sdf_raw.shape[1:]]))

        masks = []
        for s in sdf_raw:
            masks.append(s>0)
        sdf_module.init_tracker(state_img[0], masks)

        if visualize_q:
            if _env.env.camera_depth:
                ax0.imshow(goal_img[0])
                ax1.imshow(state_img[0])
            else:
                ax0.imshow(goal_img)
                ax1.imshow(state_img)
            # goal sdfs
            vis_g = norm_npy(sdf_g + 50*(sdf_g>0).astype(float))
            goal_sdfs = np.zeros([sdf_res, sdf_res, 3])
            for _s in range(len(vis_g)):
                goal_sdfs += np.expand_dims(vis_g[_s], 2) * np.array(cm(_s/5)[:3])
            ax2.imshow(norm_npy(goal_sdfs))
            # current sdfs
            vis_c = norm_npy(sdf_st_align + 50*(sdf_st_align>0).astype(float))
            current_sdfs = np.zeros([sdf_res, sdf_res, 3])
            for _s in range(len(vis_c)):
                current_sdfs += np.expand_dims(vis_c[_s], 2) * np.array(cm(_s/5)[:3])
            ax3.imshow(norm_npy(current_sdfs))
            fig.canvas.draw()

        for t_step in range(_env.max_steps):
            count_steps += 1
            ep_len += 1
            if replay_buffer.size < args.learn_start:
                sidx, action = agent.random_action([sdf_st_align, sdf_g])
            else:
                sidx, action = agent.select_action([sdf_st_align, sdf_g], evaluate=False)
            (cx, cy), sdf_mask = get_sdf_cengter_mask(env, sdf_raw, sidx)
            dx, dy = action
            pose_action = (cx, cy, dx, dy)

            # update networks #
            if replay_buffer.size > args.bs:
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(replay_buffer, args.bs, updates)
                updates += 1
                log_minibatch_critic_loss.append((critic_1_loss + critic_2_loss)/2.)
                log_minibatch_actor_loss.append(policy_loss)

            (next_state_img, _), reward, done, info = _env.step(pose_action, sdf_mask)
            sdf_ns, sdf_raw, feature_ns = sdf_module.get_sdf_features(next_state_img[0], next_state_img[1], _env.num_blocks, clip=args.clip_sdf)
            pre_n_detection = n_detection
            n_detection = len(sdf_ns)
            if args.oracle:
                sdf_ns_align = sdf_module.oracle_align(sdf_ns, info['pixel_poses'])
                sdf_raw = sdf_module.oracle_align(sdf_raw, info['pixel_poses'], scale=1)
            else:
                matching = sdf_module.object_matching(feature_ns, feature_g)
                sdf_ns_align = sdf_module.align_sdf(matching, sdf_ns, sdf_g)
                sdf_raw = sdf_module.align_sdf(matching, sdf_raw, np.zeros([_env.num_blocks, *sdf_raw.shape[1:]]))

            # sdf reward #
            reward += sdf_module.add_sdf_reward(sdf_st_align, sdf_ns_align, sdf_g)
            episode_reward += reward

            # detection failed #
            if n_detection == 0:
                done = True

            ## check GT poses and SDF centers ##
            if info['block_success'].all():
                info['success'] = True
            else:
                info['success'] = False

            if visualize_q:
                if _env.env.camera_depth:
                    ax1.imshow(next_state_img[0])
                else:
                    ax1.imshow(next_state_img)

                # goal sdfs
                vis_g = norm_npy(sdf_g + 50*(sdf_g>0).astype(float))
                goal_sdfs = np.zeros([sdf_res, sdf_res, 3])
                for _s in range(len(vis_g)):
                    goal_sdfs += np.expand_dims(vis_g[_s], 2) * np.array(cm(_s/5)[:3])
                ax2.imshow(norm_npy(goal_sdfs))
                # current sdfs
                vis_c = norm_npy(sdf_ns_align + 50*(sdf_ns_align>0).astype(float))
                current_sdfs = np.zeros([sdf_res, sdf_res, 3])
                for _s in range(len(vis_c)):
                    current_sdfs += np.expand_dims(vis_c[_s], 2) * np.array(cm(_s/5)[:3])
                ax3.imshow(norm_npy(current_sdfs))
                fig.canvas.draw()

            ## save transition to the replay buffer ##
            trajectories = []
            trajectories.append([sdf_st_align, action, sdf_ns_align, reward, done, sdf_g, sdf_g])

            ## HER ##
            if args.her and not done:
                her_sample = sample_her_transitions(_env, info)
                for sample in her_sample:
                    reward_re, goal_re, done_re, block_success_re = sample
                    reward_re += sdf_module.add_sdf_reward(sdf_st_align, sdf_ns_align, sdf_ns_align)
                    if args.round_sdf:
                        sdf_ns_align_round = sdf_module.make_round_sdf(sdf_ns_align)
                        trajectories.append([sdf_st_align, action, sdf_ns_align, reward_re, done_re, sdf_ns_align_round, sdf_ns_align_round])
                    else:
                        trajectories.append([sdf_st_align, action, sdf_ns_align, reward_re, done_re, sdf_ns_align, sdf_ns_align])

            for traj in trajectories:
                replay_buffer.add(*traj)

            if done:
                break
            else:
                sdf_st_align = sdf_ns_align
            if replay_buffer.size == args.learn_start:
                count_steps = 0
                break

        if replay_buffer.size <= learn_start:
            continue

        log_returns.append(episode_reward)
        log_critic_loss.append(np.mean(log_minibatch_critic_loss))
        log_actor_loss.append(np.mean(log_minibatch_actor_loss))
        log_eplen.append(ep_len)
        log_out.append(int(info['out_of_range']))
        log_success_total.append(int(info['success']))
        log_success_1block.append(np.mean(info['block_success']))
        if not _env.num_blocks in log_success:
            log_success[_env.num_blocks] = []
        log_success[_env.num_blocks].append(int(info['success']))

        if args.n1==args.n2:
            eplog = {
                    'reward': episode_reward,
                    'critic loss': np.mean(log_minibatch_critic_loss),
                    'actor loss': np.mean(log_minibatch_actor_loss),
                    'episode length': ep_len,
                    'out of range': int(info['out_of_range']),
                    'success rate': int(info['success']),
                    '1block success': np.mean(info['block_success']),
                    }
        else:
            eplog = {
                    '%dB reward'%_env.num_blocks: episode_reward,
                    'critic loss': np.mean(log_minibatch_critic_loss),
                    'actor loss': np.mean(log_minibatch_actor_loss),
                    'episode length': ep_len,
                    'out of range': int(info['out_of_range']),
                    'success rate': int(info['success']),
                    '1block success': np.mean(info['block_success']),
                    '%dB SR'%_env.num_blocks: int(info['success']),
                    }
        if not args.wandb_off:
            wandb.log(eplog, count_steps)

        if ne % args.log_freq == 0:
            log_mean_returns = smoothing_log_same(log_returns, args.log_freq)
            log_mean_critic_loss = smoothing_log_same(log_critic_loss, args.log_freq)
            log_mean_actor_loss = smoothing_log_same(log_actor_loss, args.log_freq)
            log_mean_eplen = smoothing_log_same(log_eplen, args.log_freq)
            log_mean_out = smoothing_log_same(log_out, args.log_freq)
            log_mean_success = smoothing_log_same(log_success_total, args.log_freq)
            log_mean_success_block = smoothing_log_same(log_success_1block, args.log_freq)

            et = time.time()
            now = datetime.datetime.now().strftime("%m/%d %H:%M")
            interval = str(datetime.timedelta(0, int(et-st)))
            st = et
            print(f"{now}({interval}) / ep{ne} ({count_steps} steps)", end=" / ")
            print(f"SR:{log_mean_success[-1]:.2f}", end=" / ")
            print("1BSR:{0:.2f}".format(log_mean_success_block[-1]), end=" ")
            print("/ Reward:{0:.2f}".format(log_mean_returns[-1]), end="")
            print(" / CriticLoss:{0:.5f}".format(log_mean_critic_loss[-1]), end="")
            print(" / ActorLoss:{0:.5f}".format(log_mean_actor_loss[-1]), end="")
            print(" / Eplen:{0:.1f}".format(log_mean_eplen[-1]), end="")
            print(" / OOR:{0:.2f}".format(log_mean_out[-1]), end="")

            log_list = [
                    log_returns,  # 0
                    log_critic_loss,  # 1
                    log_actor_loss,  # 2
                    log_eplen,  # 3
                    log_success_total,  # 4
                    log_out,  # 5
                    log_success_1block, #6
                    log_success,  # 7
                    ]
            numpy_log = np.array(log_list, dtype=object)
            np.save('results/board/%s' %savename, numpy_log)

            if log_mean_success[-1] > max_success:
                max_success = log_mean_success[-1]
                torch.save(qnet.state_dict(), 'results/models/%s.pth' % savename)
                print(" <- Highest SR. Saving the model.")
            else:
                print("")

    print('Training finished.')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # env config #
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--camera_height", default=480, type=int)
    parser.add_argument("--camera_width", default=480, type=int)
    parser.add_argument("--n1", default=3, type=int)
    parser.add_argument("--n2", default=5, type=int)
    parser.add_argument("--max_blocks", default=8, type=int)
    parser.add_argument("--dist", default=0.06, type=float)
    parser.add_argument("--threshold", default=0.10, type=float)
    parser.add_argument("--real_object", action="store_false")
    parser.add_argument("--dataset", default="train", type=str)
    parser.add_argument("--max_steps", default=100, type=int)
    parser.add_argument("--reward", default="linear_maskpenalty", type=str)
    # sdf #
    parser.add_argument("--convex_hull", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--tracker", default="medianflow", type=str)
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--clip", action="store_true")
    parser.add_argument("--round_sdf", action="store_false")
    # learning params #
    parser.add_argument("--resize", action="store_false") # defalut: True
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--bs", default=12, type=int)
    parser.add_argument("--buff_size", default=1e4, type=float)
    parser.add_argument("--total_episodes", default=1e4, type=float)
    parser.add_argument("--learn_start", default=3e3, type=float)
    parser.add_argument("--log_freq", default=50, type=int)
    parser.add_argument("--double", action="store_false")
    parser.add_argument("--her", action="store_false")
    # gcn #
    parser.add_argument("--ver", default=0, type=int)
    parser.add_argument("--adj_ver", default=1, type=int)
    parser.add_argument("--selfloop", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--separate", action="store_true")
    parser.add_argument("--bias", action="store_false")
    # SAC #
    parser.add_argument("--policy", default="Gaussian", type=str, help="Gaussian | Deterministic")
    parser.add_argument("--n_hidden", default=8, type=int)
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    # model #
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--continue_learning", action="store_true")
    parser.add_argument("--model_path", default="", type=str)
    # etc #
    parser.add_argument("--show_q", action="store_true")
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--wandb_off", action="store_true")
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
    n1 = args.n1
    n2 = args.n2
    max_blocks = args.max_blocks
    real_object = args.real_object
    dataset = args.dataset
    depth = args.depth
    threshold = args.threshold
    mov_dist = args.dist
    max_steps = args.max_steps
    camera_height = args.camera_height
    camera_width = args.camera_width
    reward_type = args.reward
    gpu = args.gpu

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if str(gpu) in visible_gpus:
            gpu_idx = visible_gpus.index(str(gpu))
            torch.cuda.set_device(gpu_idx)

    model_path = os.path.join("results/models/SAC_%s.pth"%args.model_path)
    visualize_q = args.show_q

    now = datetime.datetime.now()
    savename = "SAC_%s" % (now.strftime("%m%d_%H%M"))
    if not os.path.exists("results/config/"):
        os.makedirs("results/config/")
    with open("results/config/%s.json" % savename, 'w') as cf:
        json.dump(args.__dict__, cf, indent=2)
    
    sdf_module = SDFModule(rgb_feature=True, resnet_feature=True, convex_hull=args.convex_hull, 
            binary_hole=True, using_depth=depth, tracker=args.tracker, resize=args.resize)

    if real_object:
        from realobjects_env import UR5Env
    else:
        from ur5_env import UR5Env
    if dataset=="train":
        urenv1 = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
                control_freq=5, data_format='NHWC', gpu=gpu, camera_depth=True, dataset="train1")
        env1 = objectwise_env(urenv1, num_blocks=n1, mov_dist=mov_dist, max_steps=max_steps, \
                threshold=threshold, conti=False, detection=True, reward_type=reward_type, \
                delta_action=True)
        urenv2 = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
                control_freq=5, data_format='NHWC', gpu=gpu, camera_depth=True, dataset="train2")
        env2 = objectwise_env(urenv2, num_blocks=n1, mov_dist=mov_dist, max_steps=max_steps, \
                threshold=threshold, conti=False, detection=True, reward_type=reward_type, \
                delta_action=True)
        env = [env1, env2]
    else:
        urenv = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
                control_freq=5, data_format='NHWC', gpu=gpu, camera_depth=True, dataset="test")
        env = [objectwise_env(urenv, num_blocks=n1, mov_dist=mov_dist, max_steps=max_steps, \
                threshold=threshold, conti=False, detection=True, reward_type=reward_type, \
                delta_action=True)]

    # wandb model name #
    if not args.wandb_off:
        log_name = savename
        if n1==n2:
            log_name += '_%db' %n1
        else:
            log_name += '_%d-%db' %(n1, n2)
        log_name += '_v%d' %ver
        log_name += 'a%d' %adj_ver
        wandb.init(project="TrackGCN")
        wandb.run.name = log_name
        wandb.config.update(args)
        wandb.run.save()

    # Agent
    agent = SAC(max_blocks, args)

    learning(env, agent, sdf_module, savename, args)
