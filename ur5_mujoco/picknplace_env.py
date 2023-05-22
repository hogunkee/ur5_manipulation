from pushpixel_env import *
from reward_functions import *
import cv2
import imageio
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

class picknplace_env(pushpixel_env):
    def __init__(self, ur5_env, num_blocks=1, mov_dist=0.05, max_steps=50, threshold=0.10, \
            reward_type='binary'):
        self.threshold = threshold
        self.depth_bg = np.load(os.path.join(file_path, 'depth_bg_480.npy'))
        super().__init__(ur5_env, num_blocks, mov_dist, max_steps, 1, reward_type, 'block', False, False)
        self.cam_id = 2
        self.cam_theta = 0 * np.pi / 180
        self.set_camera()
        self.PCG = PointCloudGen()
        self.z_min = 1.04

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
        info['goals'] = np.array(self.goals)
        info['poses'] = np.array(poses)
        info['rotations'] = np.array(rotations)
        if self.num_blocks>0:
            info['dist'] = np.linalg.norm(info['goals']-info['poses'], axis=1)
            info['goal_flags'] = np.linalg.norm(info['goals']-info['poses'], axis=1) < self.threshold
        info['out_of_range'] = not self.check_blocks_in_range()
        pixel_poses = []
        for p in poses:
            _y, _x = self.pos2pixel(*p)
            pixel_poses.append([_x, _y])
        info['pixel_poses'] = np.array(pixel_poses)
        pixel_goals = []
        for g in self.goals:
            _y, _x = self.pos2pixel(*g)
            pixel_goals.append([_x, _y])
        self.pixel_goals = np.array(pixel_goals)
        info['pixel_goals'] = self.pixel_goals

        return [im_state, self.goal_image], info

    def get_force(self):
        force = self.env.sim.data.sensordata
        return force

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

    def push_pixel2pixel(self, pixel_before, pixel_target, theta):
        bx, by = pixel_before
        tx, ty = pixel_target
        pos_before = np.array(self.pixel2pos(bx, by))
        pos_before[:2] = self.clip_pos(pos_before[:2])
        pos_after = np.array(self.pixel2pos(tx, ty))
        pos_after[:2] = self.clip_pos(pos_after[:2])

        x, y, z, w = euler2quat([np.pi, 0, -theta+np.pi/2])
        quat = [w, x, y, z]
        self.env.move_to_pos([pos_before[0], pos_before[1], self.z_prepush], quat, grasp=1.0)
        self.env.move_to_pos([pos_before[0], pos_before[1], self.z_collision_check], quat, grasp=1.0)
        force = self.env.sim.data.sensordata
        if np.abs(force[2]) > 1.0 or np.abs(force[5]) > 1.0:
            #print("Collision!")
            self.env.move_to_pos([pos_before[0], pos_before[1], self.z_prepush], quat, grasp=1.0)
            if self.env.camera_depth:
                im_state, depth_state = self.env.move_to_pos(self.init_pos, grasp=1.0, get_img=True)
            else:
                im_state = self.env.move_to_pos(self.init_pos, grasp=1.0, get_img=True)
                depth_state = None
            return im_state, True, np.zeros(self.num_blocks), depth_state
        self.env.move_to_pos([pos_before[0], pos_before[1], self.z_push], quat, grasp=1.0)
        self.env.move_to_pos_slow([pos_after[0], pos_after[1], self.z_push], quat, grasp=1.0)
        contacts = self.check_block_contact()
        self.env.move_to_pos_slow([pos_after[0], pos_after[1], self.z_prepush], quat, grasp=1.0)
        if self.env.camera_depth:
            im_state, depth_state = self.env.move_to_pos(self.init_pos, grasp=1.0, get_img=True)
        else:
            im_state = self.env.move_to_pos(self.init_pos, grasp=1.0, get_img=True)
            depth_state = None
        return im_state, False, contacts, depth_state

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

    def picknplace(self, grasp, R, t):
        self.pick(grasp)
        #self.place(grasp, R, t)

        R_base = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
        R_place = grasp[:3, :3].dot(R)
        R1 = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
        R2 = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        print('test')
        print(mat2euler(R1.dot(R_base)))
        roll, pitch, yaw = mat2euler(R1.dot(R_place))
        #quat = euler2quat([roll, pitch, yaw])   # quat=[x,y,z,w]
        #R_place_recon = R2.dot(quat2mat(quat))
        #quat = euler2quat([np.pi/2, pitch, yaw])   # quat=[x,y,z,w]
        quat = euler2quat([roll, np.pi/2, yaw])   # quat=[x,y,z,w]
        #quat = euler2quat([roll, pitch, 0.0])   # quat=[x,y,z,w]
        R_place_removez = R2.dot(quat2mat(quat))
        R_place_remove_gamma = self.remove_gamma(R_place)

        if False:
            R_place = grasp[:3, :3].dot(R)
            roll, pitch, yaw = mat2euler(R_place)
            quat = euler2quat([roll, pitch, yaw])   # quat=[x,y,z,w]
            R_place_recon = quat2mat(quat)
            quat = euler2quat([roll, pitch, 0.0])   # quat=[x,y,z,w]
            #quat = euler2quat([roll, 0.0, yaw])   # quat=[x,y,z,w]
            R_place_removez = quat2mat(quat)
            R_z = np.array([[np.cos(gamma), -np.sin(gamma), 0.],
                            [np.sin(gamma), np.cos(gamma), 0.],
                            [0., 0., 1.]])
        print('-'*50)
        print('roll:', roll)
        print('pitch:', pitch)
        print('yaw:', yaw)
        print('R:')
        print(R_place)
        print('R remove-z:')
        print(R_place_removez)
        theta = self.get_angle(R_place, R_place_removez)
        print(theta)
        print('R remove-gamma:')
        print(R_place_remove_gamma)
        theta = self.get_angle(R_place, R_place_remove_gamma)
        print(theta)
        print()

        self.place(grasp, np.dot(grasp[:3, :3].T, R_place), t)
        self.place_removez(grasp, np.dot(grasp[:3, :3].T, R_place), \
                np.dot(grasp[:3, :3].T, R_place_remove_gamma), t)
        self.place_removez(grasp, np.dot(grasp[:3, :3].T, R_place), \
                np.dot(grasp[:3, :3].T, R_place_removez), t)
        input()

        #self.pick(grasp)
        #self.place(grasp, R, t)

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

    def place_removez(self, grasp, R, R_removez, t):
        real_grasp = grasp.copy()
        real_grasp[:3, 3] = real_grasp[:3, 3] - real_grasp[:3, :3].dot(np.array([0, 0, 0.04]))

        P = self.T_cam.dot(real_grasp)
        #P = self.cam_mat.dot(grasp)
        P[:3, :3] = P[:3, :3].dot(R_removez)
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

    def remove_gamma(self, R):
        a = R.reshape(-1)
        sin_beta = -a[6]
        cos_beta = np.sqrt(a[7]**2 + a[8]**2)
        R_remove_gamma = np.array([[a[0], -a[3]/cos_beta, -a[0]*a[6]/cos_beta],
                                  [a[3], a[0]/cos_beta, -a[3]*a[6]/cos_beta],
                                  [a[6], 0., cos_beta]])
        return R_remove_gamma

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
        print(R)
        print(R_)
        return theta * 180 / np.pi
    
    def remove_z_axis(self, R):
        R_flat = R.reshape(-1)
        R_wz = np.zeros_like(R_flat)
        xLen = np.sqrt(R_flat[0] * R_flat[0] + R_flat[1] * R_flat[1])
        yLen = np.sqrt(R_flat[3] * R_flat[3] + R_flat[4] * R_flat[4])
        R_wz[0] = R_flat[0] / xLen
        R_wz[1] = R_flat[1] / xLen
        R_wz[3] = R_flat[3] / yLen
        R_wz[4] = R_flat[4] / yLen
        R_wz[8] = 1
        R_wz = R_wz.reshape(3, 3)
        return R_wz 

        

    def test(self, pos, img):
        img = deepcopy(img)
        u, v = self.projection(pos)
        cv2.circle(img, (u, v), 5, (255, 0, 0), 2)
        plt.imshow(img)
        plt.show()

