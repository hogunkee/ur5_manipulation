import cv2
from ur5_env import *
from transform_utils import euler2quat, quat2mat

class reachdiscrete_env(object):
    def __init__(self, ur5_env, mov_dist=0.03, max_steps=50):
        self.env = ur5_env
        self.action_range = 8

        self.mov_dist = mov_dist

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

        # object pose #
        self.env.sim.data.qpos[12: 15] = [0., 0., 0.]

        # robot pose #
        rx = np.random.uniform(*self.eef_range_x)
        ry = np.random.uniform(*self.eef_range_y)
        self.init_pos = [rx, ry, self.z_min + 0.01] # + self.mov_dist]

        # goal pose #
        fixed_goal = False
        if fixed_goal:
            self.goal = [-0.27, 0.35]
        else:
            gx = np.random.uniform(*self.eef_range_x)
            gy = np.random.uniform(*self.eef_range_y)
            self.goal = [gx, gy]
        self.env.sim.data.qpos[19:21] = self.goal

        pre_init_pos = self.init_pos + np.array([0., 0., 0.05])
        self.env.move_to_pos(pre_init_pos, grasp=1.0, get_img=False)
        im_state = self.env.move_to_pos(self.init_pos, grasp=1.0, get_img=True)
        self.step_count = 0
        return im_state

    def reset(self):
        im_state = self.init_env()
        gripper_pose, grasp = self.get_gripper_state()
        state = np.concatenate([gripper_pose[:2], self.goal])
        return im_state, state

    def step(self, action):
        assert action < self.action_range
        pre_gripper_pose, grasp = self.get_gripper_state()

        dist = self.mov_dist
        dist2 = dist/np.sqrt(2)
        gripper_pose = deepcopy(pre_gripper_pose)
        if action==0:
            gripper_pose[1] = np.min([gripper_pose[1] + dist, self.eef_range_y[1]])
        elif action==1:
            gripper_pose[0] = np.min([gripper_pose[0] + dist2, self.eef_range_x[1]])
            gripper_pose[1] = np.min([gripper_pose[1] + dist2, self.eef_range_y[1]])
        elif action==2:
            gripper_pose[0] = np.min([gripper_pose[0] + dist, self.eef_range_x[1]])
        elif action==3:
            gripper_pose[0] = np.min([gripper_pose[0] + dist2, self.eef_range_x[1]])
            gripper_pose[1] = np.max([gripper_pose[1] - dist2, self.eef_range_y[0]])
        elif action==4:
            gripper_pose[1] = np.max([gripper_pose[1] - dist, self.eef_range_y[0]])
        elif action==5:
            gripper_pose[0] = np.max([gripper_pose[0] - dist2, self.eef_range_x[0]])
            gripper_pose[1] = np.max([gripper_pose[1] - dist2, self.eef_range_y[0]])
        elif action==6:
            gripper_pose[0] = np.max([gripper_pose[0] - dist, self.eef_range_x[0]])
        elif action==7:
            gripper_pose[0] = np.max([gripper_pose[0] - dist2, self.eef_range_x[0]])
            gripper_pose[1] = np.min([gripper_pose[1] + dist2, self.eef_range_y[1]])

        im_state = self.env.move_to_pos(gripper_pose, grasp=grasp, get_img=True)

        info = {}
        info['goal'] = np.array(self.goal)
        info['pre_gripper_pose'] = np.array(pre_gripper_pose[:2])
        info['gripper_pose'] = np.array(gripper_pose[:2])

        reward, done, success = self.get_reward(info)
        info['success'] = success

        self.step_count += 1
        if self.step_count==self.max_steps:
            done = True

        goal = info['goal']
        state = np.concatenate([info['gripper_pose'], goal])
        return [im_state, state], reward, done, info

    def get_gripper_state(self):
        # get gripper_pose, grasp_close #
        return deepcopy(self.env.sim.data.mocap_pos[0]), deepcopy(int(bool(sum(self.env.sim.data.ctrl))))

    def get_reward(self, info):
        reward_scale_1 = 30
        reward_scale_2 = 3
        min_reward = -1

        goal = info['goal']
        gripper_pose = info['gripper_pose']
        pre_gripper_pose = info['pre_gripper_pose']

        dist = np.linalg.norm(gripper_pose - goal)
        pre_dist = np.linalg.norm(pre_gripper_pose - goal)

        reward = 0.0
        reward += reward_scale_1 * (pre_dist - dist)
        reward -= self.time_penalty

        done = False
        success = (dist < self.threshold)
        if success:
            reward = 10.0
            done = True

        reward = max(reward, min_reward)
        return reward, done, success


if __name__=='__main__':
    env = UR5Env(render=True, camera_height=64, camera_width=64, control_freq=5, xml_ver='1bpush')
    env = reachdiscrete_env(env, mov_dist=0.03, max_steps=100)
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
