import cv2
from ur5_env import *
from transform_utils import euler2quat, quat2mat

class pushdiscrete_env(object):
    def __init__(self, ur5_env, mov_dist=0.03, max_steps=50):
        self.env = ur5_env
        self.mov_dist = mov_dist

        self.block_spawn_range_x = [-0.20, 0.20]
        self.block_spawn_range_y = [-0.10, 0.30]
        self.block_range_x = [-0.25, 0.25]
        self.block_range_y = [-0.15, 0.35]
        self.eef_range_x = [-0.35, 0.35]
        self.eef_range_y = [-0.22, 0.40]
        self.z_min = 1.05
        self.z_max = self.z_min + 3 * self.mov_dist
        self.time_penalty = 1e-2
        self.max_steps = max_steps
        self.step_count = 0
        self.threshold = 0.05

        self.init_pos = [0.0, 0.0, self.z_min + 0.01] # + self.mov_dist]
        self.init_env()

    def init_env(self):
        self.env._init_robot()
        range_x = self.block_spawn_range_x
        range_y = self.block_spawn_range_y
        # object pose #
        tx = np.random.uniform(*range_x)
        ty = np.random.uniform(*range_y)
        tz = 0.9
        self.env.sim.data.qpos[12: 15] = [tx, ty, tz]
        # robot pose #
        theta = 2 * np.pi * np.random.rand()
        radius = np.random.uniform(0.05, 0.1)
        rx = tx + radius * np.cos(theta)
        ry = ty + radius * np.sin(theta)
        self.init_pos = [rx, ry, self.z_min + 0.01] # + self.mov_dist]
        # goal pose #
        self.goal = [-0.27, 0.35]
        #self.env.sim.data.qpos[19:21] = [0.2, 0.2] #, 0.9]

        pre_init_pos = self.init_pos + np.array([0., 0., 0.05])
        self.env.move_to_pos(pre_init_pos, grasp=1.0, get_img=False)
        im_state = self.env.move_to_pos(self.init_pos, grasp=1.0, get_img=True)
        self.step_count = 0
        return im_state

    def reset(self):
        im_state = self.init_env()
        gripper_pose, grasp = self.get_gripper_state()
        pose, _ = self.get_poses()
        state = np.concatenate([gripper_pose[:2], pose, self.goal])
        return im_state, state

    def step(self, action):
        # action: (delta_x, delta_y)
        pre_gripper_pose, grasp = self.get_gripper_state()
        pre_pose, _ = self.get_poses()

        gripper_pose = deepcopy(pre_gripper_pose)
        gx = np.clip(gripper_pose[0] + action[0], self.eef_range_x[0], self.eef_range_x[1])
        gy = np.clip(gripper_pose[1] + action[1], self.eef_range_y[0], self.eef_range_y[1])
        gripper_pose[:2] = [gx, gy]

        im_state = self.env.move_to_pos(gripper_pose, grasp=grasp, get_img=True)
        pose, _ = self.get_poses()

        info = {}
        info['out_of_range'] = not self.check_blocks_in_range()
        info['goal'] = np.array(self.goal)
        info['pre_pose'] = np.array(pre_pose)
        info['pose'] = np.array(pose)
        #info['rotations'] = np.array(rotations)
        info['goal_flag'] = np.linalg.norm(info['goal']-info['pose']) < self.threshold
        info['pre_gripper_pose'] = np.array(pre_gripper_pose[:2])
        info['gripper_pose'] = np.array(gripper_pose[:2])

        reward, done, success = self.get_reward(info)
        info['success'] = success

        self.step_count += 1
        if self.step_count==self.max_steps:
            done = True

        pose = info['pose']
        goal = info['goal']
        state = np.concatenate([info['gripper_pose'], pose, goal])
        return [im_state, state], reward, done, info

    def get_poses(self):
        obj_idx = 0
        pos = deepcopy(self.env.sim.data.get_body_xpos('target_body_%d'%(obj_idx+1))[:2])
        quat = deepcopy(self.env.sim.data.get_body_xquat('target_body_%d'%(obj_idx+1)))
        rotation_mat = quat2mat(np.concatenate([quat[1:],quat[:1]]))
        return pos, rotation_mat

    def check_blocks_in_range(self):
        pose, _ = self.get_poses()
        x_max, y_max = np.array([pose]).reshape(-1, 2).max(0)
        x_min, y_min = np.array([pose]).reshape(-1, 2).min(0)
        if x_max > self.block_range_x[1] or x_min < self.block_range_x[0]:
            return False
        if y_max > self.block_range_y[1] or y_min < self.block_range_y[0]:
            return False
        return True

    def get_gripper_state(self):
        # get gripper_pose, grasp_close #
        return deepcopy(self.env.sim.data.mocap_pos[0]), deepcopy(int(bool(sum(self.env.sim.data.ctrl))))

    def get_reward(self, info):
        reward_scale_1 = 30
        reward_scale_2 = 3
        min_reward = -1

        goal = info['goal']
        pose = info['pose']
        pre_pose = info['pre_pose']
        oor = info['out_of_range']
        gripper_pose = info['gripper_pose']
        pre_gripper_pose = info['pre_gripper_pose']

        dist = np.linalg.norm(pose - goal)
        pre_dist = np.linalg.norm(pre_pose - goal)
        gripper_dist = np.linalg.norm(pose - gripper_pose)
        pre_gripper_dist = np.linalg.norm(pre_pose - pre_gripper_pose)

        reward = 0.0
        reward += reward_scale_1 * (pre_dist - dist)
        reward += reward_scale_2 * (pre_gripper_dist - gripper_dist)
        reward -= self.time_penalty

        done = False
        success = (dist < self.threshold)
        if success:
            reward = 10.0
            done = True
        elif oor:
            reward = -2.0
            done = True

        reward = max(reward, min_reward)
        return reward, done, success


if __name__=='__main__':
    env = UR5Env(render=True, camera_height=64, camera_width=64, control_freq=5, xml_ver='1bpush')
    env = pushdiscrete_env(env, mov_dist=0.03, max_steps=100)
    frame, state = env.reset()

    for i in range(100):
        #action = [np.random.randint(6), np.random.randint(2)]
        try:
            action = int(input("action? "))
        except KeyboardInterrupt:
            exit()
        except:
            continue
        if action > 10:
            continue
        print('{} steps. action: {}'.format(env.step_count, action))
        states, reward, done, info = env.step(action)

        print(states[0].shape)
        print(states[1].shape)
        plt.imshow(states[0])
        plt.show()
        print('Reward: {}. Done: {}'.format(reward, done))

        if done:
            print('Done. New episode starts.')
            env.reset()
