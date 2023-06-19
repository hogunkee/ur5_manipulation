import cv2
import imageio
import types
import time
import numpy as np
import os
file_path = os.path.dirname(os.path.abspath(__file__))

from copy import deepcopy
from matplotlib import pyplot as plt
from reward_functions import *
from transform_utils import euler2quat, quat2mat, mat2quat, mat2euler

import sys
sys.path.append('/home/gun/Desktop/contact_graspnet/contact_graspnet')
import config_utils
from contact_grasp_estimator import GraspEstimator
from visualization_utils import visualize_grasps, show_image

from pcd_gen import *
from cpd import *
from scipy.optimize import linear_sum_assignment

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class picknplace_env(object):
    def __init__(self, ur5_env, num_blocks=1, mov_dist=0.05, max_steps=50, threshold=0.10, \
            angle_threshold=30, reward_type='binary'):
        self.env = ur5_env 
        self.num_blocks = num_blocks
        self.mov_dist = mov_dist
        self.block_spawn_range_x = [-0.25, 0.25] #[-0.20, 0.20] #[-0.25, 0.25]
        self.block_spawn_range_y = [-0.12, 0.3] #[-0.10, 0.30] #[-0.15, 0.35]
        self.block_range_x = [-0.31, 0.31] #[-0.25, 0.25]
        self.block_range_y = [-0.18, 0.36] #[-0.15, 0.35]
        self.eef_range_x = [-0.35, 0.35]
        self.eef_range_y = [-0.22, 0.40]
        self.z_pick = 1.045
        self.z_prepick = self.z_pick + 0.12 #2.5 * self.mov_dist
        self.z_min = 1.04

        self.time_penalty = 0.02 #0.1
        self.max_steps = max_steps
        self.step_count = 0
        self.reward_type = reward_type

        self.init_pos = [0.0, -0.23, 1.4]

        self.threshold = threshold
        self.angle_threshold = angle_threshold
        self.depth_bg = np.load(os.path.join(file_path, 'depth_bg_480.npy'))
        self.cam_id = 2
        self.cam_theta = 0 * np.pi / 180
        self.set_camera()

        self.pre_selected_objects = None
        self.PCG = PointCloudGen()
        self.init_env()

    def init_env(self, scenario=-1):
        self.env._init_robot()
        self.env.selected_objects = self.env.selected_objects[:self.num_blocks]
        range_x = self.block_spawn_range_x
        range_y = self.block_spawn_range_y
        threshold = 0.10

        # mesh grid 
        x = np.linspace(-0.4, 0.4, 7)
        y = np.linspace(0.4, -0.2, 5)
        xx, yy = np.meshgrid(x, y, sparse=False)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)

        # init all blocks
        if self.pre_selected_objects is None:
            for obj_idx in range(self.env.num_objects):
                self.env.sim.data.qpos[7*obj_idx+12: 7*obj_idx+15] = [xx[obj_idx], yy[obj_idx], 0]
        else:
            for obj_idx in self.pre_selected_objects:
                self.env.sim.data.qpos[7*obj_idx+12: 7*obj_idx+15] = [xx[obj_idx], yy[obj_idx], 0]

        check_feasible = False
        while not check_feasible:
            goals, inits = self.generate_scene(scenario)
            for i, obj_idx in enumerate(self.env.selected_objects):
                if i>=self.num_blocks:
                    self.env.sim.data.qpos[7*obj_idx+12: 7*obj_idx+15] = [xx[obj_idx], yy[obj_idx], 0]
                    continue
                gx, gy = goals[i]
                gz = 0.95
                euler = np.zeros(3) 
                euler[2] = 2*np.pi * np.random.random()
                if self.env.real_object:
                    if obj_idx in self.env.obj_orientation:
                        euler[:2] = np.pi * np.array(self.env.obj_orientation[obj_idx])
                    else:
                        euler[:2] = [0, 0]
                x, y, z, w = euler2quat(euler)
                self.env.sim.data.qpos[7*obj_idx+12: 7*obj_idx+15] = [gx, gy, gz]
                self.env.sim.data.qpos[7*obj_idx+15: 7*obj_idx+19] = [w, x, y, z]
            for i in range(50):
                self.env.sim.step()
                if self.env.render: self.env.sim.render(mode='window')
            check_goal_feasible = self.check_blocks_in_range()
            self.goal_image = self.env.move_to_pos(self.init_pos, grasp=1.0, get_img=True)
            self.goal_poses, self.goal_rotations = self.get_poses()

            for i, obj_idx in enumerate(self.env.selected_objects):
                if i>=self.num_blocks:
                    continue
                tx, ty = inits[i]
                tz = 0.95
                euler = np.zeros(3) 
                euler[2] = 2*np.pi * np.random.random()
                if self.env.real_object:
                    if obj_idx in self.env.obj_orientation:
                        euler[:2] = np.pi * np.array(self.env.obj_orientation[obj_idx])
                    else:
                        euler[:2] = [0, 0]
                x, y, z, w = euler2quat(euler)
                self.env.sim.data.qpos[7*obj_idx+12: 7*obj_idx+15] = [tx, ty, tz]
                self.env.sim.data.qpos[7*obj_idx+15: 7*obj_idx+19] = [w, x, y, z]
            for i in range(50):
                self.env.sim.step()
                if self.env.render: self.env.sim.render(mode='window')
            check_init_feasible = self.check_blocks_in_range()
            im_state = self.env.move_to_pos(self.init_pos, grasp=1.0, get_img=True)
            self.pre_poses, self.pre_rotations = self.get_poses()

            check_feasible = check_goal_feasible and check_init_feasible

        self.goals = goals
        self.pre_selected_objects = self.env.selected_objects
        self.step_count = 0
        return im_state

    def generate_scene(self, scene):
        threshold = 0.1
        range_x = self.block_spawn_range_x
        range_y = self.block_spawn_range_y
        # randon scene
        goals = []
        inits = []
        if scene==-1:
            check_feasible = False
            while not check_feasible:
                nb = self.num_blocks
                goal_x = np.random.uniform(*range_x, size=self.num_blocks)
                goal_y = np.random.uniform(*range_y, size=self.num_blocks)
                goals = np.concatenate([goal_x, goal_y]).reshape(2, -1).T
                init_x = np.random.uniform(*range_x, size=self.num_blocks)
                init_y = np.random.uniform(*range_y, size=self.num_blocks)
                inits = np.concatenate([init_x, init_y]).reshape(2, -1).T
                dist_g = np.linalg.norm(goals.reshape(nb, 1, 2) - goals.reshape(1, nb, 2), axis=2) + 1
                dist_i = np.linalg.norm(inits.reshape(nb, 1, 2) - inits.reshape(1, nb, 2), axis=2) + 1
                if not((dist_g < threshold).any() or (dist_i < threshold).any()):
                    check_feasible = True

        # random grid #
        elif scene==0:
            num_grid = 4
            #x = np.linspace(range_x[0], range_x[1], num_grid)
            #y = np.linspace(range_y[0], range_y[1], num_grid)
            offset_x = (range_x[1] - range_x[0]) / (3*num_grid) # 2*num_grid
            offset_y = (range_y[1] - range_y[0]) / (3*num_grid) # 2*num_grid
            x = np.linspace(range_x[0] + offset_x, range_x[1] - offset_x, num_grid)
            y = np.linspace(range_y[0] + offset_y, range_y[1] - offset_y, num_grid)
            xx, yy = np.meshgrid(x, y, sparse=False)
            xx = xx.reshape(-1)
            yy = yy.reshape(-1)
            indices = np.arange(len(xx))

            check_scene = False
            while not check_scene:
                selected_grid = np.random.choice(indices, 2*self.num_blocks, replace=False)
                num_collisions = 0
                for idx1 in range(self.num_blocks):
                    for idx2 in range(self.num_blocks):
                        sx1, sy1 = xx[idx1], yy[idx1]
                        gx1, gy1 = xx[self.num_blocks+idx1], yy[self.num_blocks+idx1]
                        sx2, sy2 = xx[idx2], yy[idx2]
                        gx2, gy2 = xx[self.num_blocks+idx2], yy[self.num_blocks+idx2]
                        intersection = self.line_intersection([[sx1, sy1], [gx1, gy1]], [[sx2, sy2], [gx2, gy2]])
                        if intersection is None:
                            continue
                        else:
                            check_x1 = min(sx1, gx1) <= intersection[0] <= max(sx1, gx1)
                            check_y1 = min(sy1, gy1) <= intersection[1] <= max(sy1, gy1)
                            check_x2 = min(sx2, gx2) <= intersection[0] <= max(sx2, gx2)
                            check_y2 = min(sy2, gy2) <= intersection[1] <= max(sy2, gy2)
                            if check_x1 and check_y1 and check_x2 and check_y2:
                                num_collisions += 1
                init_x = xx[selected_grid[:self.num_blocks]]
                init_y = yy[selected_grid[:self.num_blocks]]
                goal_x = xx[selected_grid[self.num_blocks:]]
                goal_y = yy[selected_grid[self.num_blocks:]]
                check_scene = True
            goals = np.concatenate([goal_x, goal_y]).reshape(2, -1).T
            inits = np.concatenate([init_x, init_y]).reshape(2, -1).T

        # no blocking #
        elif scene==1:
            range_x = self.block_spawn_range_x
            range_y = self.block_spawn_range_y
            num_grid = 4
            x = np.linspace(range_x[0], range_x[1], num_grid)
            y = np.linspace(range_y[0], range_y[1], num_grid)
            #offset_x = (range_x[1] - range_x[0]) / (2*num_grid)
            #offset_y = (range_y[1] - range_y[0]) / (2*num_grid)
            #x = np.linspace(range_x[0] + offset_x, range_x[1] - offset_x, num_grid)
            #y = np.linspace(range_y[0] + offset_y, range_y[1] - offset_y, num_grid)
            xx, yy = np.meshgrid(x, y, sparse=False)
            xx = xx.reshape(-1)
            yy = yy.reshape(-1)
            indices = np.arange(len(xx))

            check_scene = False
            while not check_scene:
                selected_grid = np.random.choice(indices, 2*self.num_blocks, replace=False)
                num_collisions = 0
                for idx1 in range(self.num_blocks):
                    for idx2 in range(self.num_blocks):
                        sx1, sy1 = xx[idx1], yy[idx1]
                        gx1, gy1 = xx[self.num_blocks+idx1], yy[self.num_blocks+idx1]
                        sx2, sy2 = xx[idx2], yy[idx2]
                        gx2, gy2 = xx[self.num_blocks+idx2], yy[self.num_blocks+idx2]
                        intersection = self.line_intersection([[sx1, sy1], [gx1, gy1]], [[sx2, sy2], [gx2, gy2]])
                        if intersection is None:
                            continue
                        else:
                            check_x1 = min(sx1, gx1) <= intersection[0] <= max(sx1, gx1)
                            check_y1 = min(sy1, gy1) <= intersection[1] <= max(sy1, gy1)
                            check_x2 = min(sx2, gx2) <= intersection[0] <= max(sx2, gx2)
                            check_y2 = min(sy2, gy2) <= intersection[1] <= max(sy2, gy2)
                            if check_x1 and check_y1 and check_x2 and check_y2:
                                num_collisions += 1
                init_x = xx[selected_grid[:self.num_blocks]]
                init_y = yy[selected_grid[:self.num_blocks]]
                goal_x = xx[selected_grid[self.num_blocks:]]
                goal_y = yy[selected_grid[self.num_blocks:]]
                if num_collisions==0:
                    check_scene = True
            goals = np.concatenate([goal_x, goal_y]).reshape(2, -1).T
            inits = np.concatenate([init_x, init_y]).reshape(2, -1).T

        # crossover #
        elif scene==2:
            range_x = self.block_spawn_range_x
            range_y = self.block_spawn_range_y
            num_grid = 4
            x = np.linspace(range_x[0], range_x[1], num_grid)
            y = np.linspace(range_y[0], range_y[1], num_grid)
            #offset_x = (range_x[1] - range_x[0]) / (2*num_grid)
            #offset_y = (range_y[1] - range_y[0]) / (2*num_grid)
            #x = np.linspace(range_x[0] + offset_x, range_x[1] - offset_x, num_grid)
            #y = np.linspace(range_y[0] + offset_y, range_y[1] - offset_y, num_grid)
            xx, yy = np.meshgrid(x, y, sparse=False)
            xx = xx.reshape(-1)
            yy = yy.reshape(-1)
            indices = np.arange(len(xx))

            check_scene = False
            while not check_scene:
                selected_grid = np.random.choice(indices, 2*self.num_blocks, replace=False)
                num_collisions = 0
                for idx1 in range(self.num_blocks):
                    for idx2 in range(self.num_blocks):
                        sx1, sy1 = xx[idx1], yy[idx1]
                        gx1, gy1 = xx[self.num_blocks+idx1], yy[self.num_blocks+idx1]
                        sx2, sy2 = xx[idx2], yy[idx2]
                        gx2, gy2 = xx[self.num_blocks+idx2], yy[self.num_blocks+idx2]
                        intersection = self.line_intersection([[sx1, sy1], [gx1, gy1]], [[sx2, sy2], [gx2, gy2]])
                        if intersection is None:
                            continue
                        else:
                            check_x1 = min(sx1, gx1) <= intersection[0] <= max(sx1, gx1)
                            check_y1 = min(sy1, gy1) <= intersection[1] <= max(sy1, gy1)
                            check_x2 = min(sx2, gx2) <= intersection[0] <= max(sx2, gx2)
                            check_y2 = min(sy2, gy2) <= intersection[1] <= max(sy2, gy2)
                            if check_x1 and check_y1 and check_x2 and check_y2:
                                num_collisions += 1
                init_x = xx[selected_grid[:self.num_blocks]]
                init_y = yy[selected_grid[:self.num_blocks]]
                goal_x = xx[selected_grid[self.num_blocks:]]
                goal_y = yy[selected_grid[self.num_blocks:]]
                if num_collisions==1:
                    check_scene = True
            goals = np.concatenate([goal_x, goal_y]).reshape(2, -1).T
            inits = np.concatenate([init_x, init_y]).reshape(2, -1).T

        return np.array(goals), np.array(inits)

    def set_camera(self, fovy=45):
        f = 0.5 * self.env.camera_height / np.tan(fovy * np.pi / 360)
        self.cam_K = np.array([[f, 0, 240],
                          [0, f, 240],
                          [0, 0, 1]])

        x, y, z, w = self.env.sim.model.cam_quat[self.cam_id]
        cam_rotation = quat2mat([w, x, y, z])
        cam_pose = self.env.sim.model.cam_pos[self.cam_id]
        T_cam = np.eye(4)
        T_cam[:3, :3] = cam_rotation
        T_cam[:3, 3] = cam_pose
        self.T_cam = T_cam
        cam_mat = np.eye(4)
        cam_mat[:3, :3] = cam_rotation
        cam_mat[:3, 3] = - cam_rotation.dot(cam_pose)
        self.cam_mat = cam_mat

    def load_contactgraspnet(self, ckpt_dir, arg_configs):
        self.z_range = [0.2, 1.0]
        self.local_regions = False
        self.filter_grasps = False
        self.skip_border_objects = False
        self.forward_passes = 1

        global_config = config_utils.load_config(ckpt_dir, batch_size=self.forward_passes,
                                                    arg_configs=arg_configs)
        self.grasp_estimator = GraspEstimator(global_config)
        self.grasp_estimator.build_network()

        saver = tf.train.Saver(save_relative_paths=True)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        self.grasp_estimator.load_weights(self.sess, saver, ckpt_dir, mode='test')


    def reset(self, sidx=-1, scenario=-1):
        if self.env.real_object:
            self.env.select_objects(self.num_blocks, sidx)
        im_state = self.init_env(scenario)
        poses, rotations = self.get_poses()
        goals = np.array(self.goals)

        info = {}
        info['num_blocks'] = self.num_blocks
        info['target'] = -1
        info['goal_poses'] = np.array(self.goal_poses)
        info['goal_rotations'] = np.array(self.goal_rotations)
        info['poses'] = np.array(poses)
        info['rotations'] = np.array(rotations)
        if self.num_blocks>0:
            info['dist_diff'] = np.linalg.norm(info['goal_poses']-info['poses'], axis=1)
            info['dist_flags'] = info['dist_diff'] < self.threshold
            info['angle_diff'] = self.get_angles(info['goal_rotations'], info['rotations'])
            info['angle_flags'] = info['angle_diff'] < self.angle_threshold
        info['out_of_range'] = not self.check_blocks_in_range()
        return [im_state, self.goal_image], info

    def get_force(self):
        force = self.env.sim.data.sensordata
        return force

    def step(self, action):
        pre_poses, pre_rotations = self.get_poses()
        #self.pick
        #self.place
        poses, rotations = self.get_poses()

        info = {}
        info['num_blocks'] = self.num_blocks
        info['action'] = action
        info['goals'] = np.array(self.goals)
        #info['contact'] = contact
        #info['collision'] = collision
        info['pre_poses'] = np.array(pre_poses)
        info['pre_rotations'] = np.array(pre_rotations)
        info['poses'] = np.array(poses)
        info['rotations'] = np.array(rotations)
        #info['dist'] = np.linalg.norm(info['goals']-info['poses'], axis=1)
        #info['goal_flags'] = np.linalg.norm(info['goals']-info['poses'], axis=1) < self.threshold
        #info['out_of_range'] = not self.check_blocks_in_range()
        pixel_poses = []
        for p in poses:
            _y, _x = self.pos2pixel(*p)
            pixel_poses.append([_x, _y])
        info['pixel_poses'] = np.array(pixel_poses)
        info['pixel_goals'] = self.pixel_goals

        reward, done, block_success = self.get_reward(info)
        #info['success'] = np.all(block_success)
        info['block_success'] = block_success

        self.step_count += 1
        if self.step_count==self.max_steps:
            done = True

        return None, reward, done, info
        #return [im_state, self.goal_image], reward, done, info

    def get_grasps(self, rgb, depth, segmap=None):
        depth[:20] = depth[:20, :100].mean()
        rgb[:20] = rgb[:20, :100].mean(axis=0).mean(axis=0)

        pc_full, pc_segments, pc_colors = self.grasp_estimator.extract_point_clouds(depth, \
                self.cam_K, segmap=segmap, rgb=(rgb*255).astype(np.uint8), \
                skip_border_objects=self.skip_border_objects, z_range=self.z_range)

        grasps, scores, contact_pts, _ = self.grasp_estimator.predict_scene_grasps(self.sess, \
                pc_full, pc_segments=pc_segments, local_regions=self.local_regions, \
                filter_grasps=self.filter_grasps, forward_passes=self.forward_passes)

        visualize_grasps(pc_full, grasps, scores, plot_opencv_cam=True, pc_colors=pc_colors)
        
        return grasps[-1], scores[-1]

    def get_grasp_pixel(self, grasp):
        eef_grasp = grasp.copy()
        eef_grasp[:3, 3] = eef_grasp[:3, 3] + eef_grasp[:3, :3].dot(np.array([0, 0, 0.04]))
        t = self.T_cam.dot(eef_grasp)[:3, 3]
        u, v = self.projection(t)
        return u, v

    def extract_grasps(self, grasps, scores, masks):
        candidates = {}
        for grasp, score in zip(grasps, scores):
            u, v = self.get_grasp_pixel(grasp)
            for i, m in enumerate(masks):
                m_blur = cv2.blur(m, (7, 7)).astype(bool).astype(int)
                if not i in candidates:
                    candidates[i] = []
                if m_blur[v, u] != 0:
                    candidates[i].append([grasp, score])
        for c in candidates:
            candidates[c].sort(key=lambda x: x[1], reverse=True)
        return candidates

    def picknplace(self, grasps, R, t):
        exist_feasible_grasp = False
        for g in grasps:
            grasp = g[0]
            R_place = grasp[:3, :3].dot(R)
            if R_place[2, 2]<=0:
                continue

            exist_feasible_grasp = True
            self.pick(grasp)
            placement = self.place(grasp, np.dot(grasp[:3, :3].T, R_place), t)
            #self.place(grasp, R, t)
            break

        if exist_feasible_grasp:
            return placement
        else:
            print('No feasible grasps..')
            return None

    def pick(self, grasp):
        real_grasp = grasp.copy()
        real_grasp[:3, 3] = real_grasp[:3, 3] - real_grasp[:3, :3].dot(np.array([0, 0, 0.04]))
        P = self.T_cam.dot(real_grasp)
        R = P[:3, :3]
        t = P[:3, 3]
        t[2] = max(t[2], self.z_min)
        quat = mat2quat(R)      # quat=[w,x,y,z]

        pre_grasp = grasp.copy()
        pre_grasp[:3, 3] = pre_grasp[:3, 3] - pre_grasp[:3, :3].dot(np.array([0, 0, 0.10]))
        P_pre = self.T_cam.dot(pre_grasp)

        self.env.move_to_pos(grasp=0.0)
        self.env.move_to_pos(P_pre[:3, 3], [quat[3], quat[0], quat[1], quat[2]], grasp=0.0)
        self.env.move_to_pos(t, [quat[3], quat[0], quat[1], quat[2]], grasp=0.0)
        self.env.move_to_pos(t, [quat[3], quat[0], quat[1], quat[2]], grasp=1.0)
        self.env.move_to_pos_slow(P_pre[:3, 3], [quat[3], quat[0], quat[1], quat[2]], grasp=1.0)
        self.env.move_to_pos_slow(quat=[quat[3], quat[0], quat[1], quat[2]], grasp=1.0)

    def place(self, grasp, R, t):
        real_grasp = grasp.copy()
        real_grasp[:3, 3] = real_grasp[:3, 3] - real_grasp[:3, :3].dot(np.array([0, 0, 0.04]))

        P = self.T_cam.dot(real_grasp)
        #P = self.cam_mat.dot(grasp)
        P[:3, :3] = P[:3, :3].dot(R)
        P[:3, 3] = P[:3, 3].dot(R) + t
        P[2, 3] = max(P[2, 3], self.z_min + 0.04)
        quat = mat2quat(P[:3, :3])

        P_pre = P.copy()
        P_pre[:3, 3] = P_pre[:3, 3] + np.array([0, 0, 0.1])
        #pre_place[:3, 3] = pre_place[:3, 3] - pre_place[:3, :3].dot(np.array([0, 0, 0.10]))

        self.env.move_to_pos_slow(P_pre[:3, 3], [quat[3], quat[0], quat[1], quat[2]], grasp=1.0)
        self.env.move_to_pos_slow(P[:3, 3], [quat[3], quat[0], quat[1], quat[2]], grasp=1.0)
        self.env.move_to_pos(P[:3, 3], [quat[3], quat[0], quat[1], quat[2]], grasp=0.0)
        self.env.move_to_pos(P_pre[:3, 3], [quat[3], quat[0], quat[1], quat[2]], grasp=0.0)
        self.env.move_to_pos(grasp=0.0)
        placement = P_pre[:3, 3]
        return placement

    def apply_cpd(self, state, goal, masks):
        state_rgb, state_depth = state
        goal_rgb, goal_depth = goal
        pcd_s = self.PCG.pcd_from_rgbd(255*state_rgb, state_depth)
        pcd_g = self.PCG.pcd_from_rgbd(255*goal_rgb, goal_depth)
        print(np.array(pcd_s.points).shape)
        pcd_s = pcd_s.random_down_sample(sampling_ratio=0.3)
        pcd_g = pcd_g.random_down_sample(sampling_ratio=0.3)
        print(np.array(pcd_s.points).shape)

        K = 3
        reg = ArtRegistrationColor(pcd_s, pcd_g, K, max_iterations=40, tolerance=1e-5, gpu=False)
        TY, (R, t) = reg.register(visualize_color)
        Z = reg.reg.Z

        M = len(masks)
        count_points = np.zeros([K, M])
        select_K = np.argmax(Z, axis=1)
        for i, k in enumerate(select_K):
            tx, ty, tz = reg.reg.Y[i, :3]
            u, v = self.projection([tx, ty, tz])
            count_points[k, :] += np.array(masks)[:, v, u]
            #cv2.circle(state_rgb, (u, v), 5, (255, 0, 0), 2)
            # print(np.array([tx, ty, tz]).dot(R[k]) + t[k])
            # tx, ty, tz = TY[k, i, :3]
            # print(tx, ty, tz)
            # print()
            # u, v = self.projection([tx, ty, tz])
            #cv2.circle(goal_rgb, (u, v), 5, (255, 0, 0), 2)
        #plt.imshow(state_rgb)
        #plt.imshow(goal_rgb)
        #plt.show()
        #print('count:', count_points)
        idx_m, idx_k = linear_sum_assignment(-count_points.T)
        return R[idx_k], t[idx_k]

    def projection(self, pose_3d):
        cam_K = deepcopy(self.cam_K)
        # cam_K[0, 0] = -cam_K[0, 0]
        x_world = np.ones(4)
        x_world[:3] = pose_3d
        p = cam_K.dot(self.cam_mat[:3].dot(x_world))
        u = p[0] / p[2]
        v = p[1] / p[2]
        return int(np.round(u)), int(np.round(v))

    def get_angle(self, R1, R2):
        R = np.multiply(R1.T, R2)
        cos_theta = (np.trace(R)-1)/2
        cos_theta = np.clip(cos_theta, -1, 1)
        theta = np.arccos(cos_theta)

        R_ = np.multiply(R1, R2.T)
        cos_theta_ = (np.trace(R_)-1)/2
        cos_theta_ = np.clip(cos_theta_, -1, 1)
        theta_ = np.arccos(cos_theta_)
        #print(R)
        #print(R_)
        return theta * 180 / np.pi

    def get_angles(self, Rs1, Rs2):
        angles = []
        for r1, r2 in zip(Rs1, Rs2):
            theta = self.get_angle(r1, r2)
            angles.append(theta)
        return np.array(angles)
    

    def test(self, pos, img):
        img = deepcopy(img)
        u, v = self.projection(pos)
        cv2.circle(img, (u, v), 5, (255, 0, 0), 2)
        plt.imshow(img)
        plt.show()

    def pick2(self, pos, quat):
        rot_mat = quat2mat(quat)
        pos_before = pos + np.dot(rot_mat, np.array([0, 0, 0.1]))
        self.env.move_to_pos(pos_before, quat, grasp=0.0)
        print('1.', self.get_force())
        self.env.move_to_pos(pos, quat, grasp=0.0)
        print('2.', self.get_force())
        self.env.move_to_pos(pos, quat, grasp=1.0)
        print('3.', self.get_force())
        self.env.move_to_pos(self.init_pos, grasp=1.0)
        print('4.', self.get_force())

        force = self.get_force()
        if False and force[0]>1.0:
            print("Failed.")

    def place2(self, pos, quat):
        rot_mat = quat2mat(quat)
        pos_before = pos + np.dot(rot_mat, np.array([0, 0, 0.1]))
        self.env.move_to_pos(pos_before, quat, grasp=1.0)
        self.env.move_to_pos(pos, quat, grasp=1.0)
        self.env.move_to_pos(pos, quat, grasp=0.0)
        rgb, depth = self.env.move_to_pos(self.init_pos, grasp=0.0, get_img=True)
        return rgb, depth

    def get_poses(self):
        poses = []
        rotations = []
        for obj_idx in self.env.selected_objects:
            pos = deepcopy(self.env.sim.data.get_body_xpos(self.env.object_names[obj_idx])[:2])
            poses.append(pos)
            quat = deepcopy(self.env.sim.data.get_body_xquat(self.env.object_names[obj_idx]))
            rotation_mat = quat2mat(np.concatenate([quat[1:],quat[:1]]))
            rotations.append(rotation_mat)
        return poses, rotations

    def check_blocks_in_range(self):
        poses, _ = self.get_poses()
        if len(poses)==0:
            return True
        x_max, y_max = np.concatenate(poses).reshape(-1, 2).max(0)
        x_min, y_min = np.concatenate(poses).reshape(-1, 2).min(0)
        if x_max > self.block_range_x[1] or x_min < self.block_range_x[0]:
            return False
        if y_max > self.block_range_y[1] or y_min < self.block_range_y[0]:
            return False
        return True

                                  
