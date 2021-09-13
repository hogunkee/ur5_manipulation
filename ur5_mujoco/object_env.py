from pushpixel_env import *
from reward_functions import *
import cv2
import imageio
from transform_utils import euler2quat, quat2mat

class objectwise_env(pushpixel_env):
    def __init__(self, ur5_env, num_blocks=1, mov_dist=0.05, max_steps=50, reward_type='binary', conti=False, detection=False):
        self.conti = conti
        self.detection = detection
        super().__init__(ur5_env, num_blocks, mov_dist, max_steps, 0, reward_type, 'block', False, False)

    def reset(self):
        im_state = self.init_env()
        poses, rotations = self.get_poses()
        goals = np.array(self.goals)

        if self.detection:
            if self.env.camera_depth:
                return [im_state, depth]
            else:
                return [im_state]
        else:
            state_goal = [poses, goals]
            return state_goal

    def step(self, action):
        poses, _ = self.get_poses()

        push_obj, theta = action
        if theta_idx >= self.num_bins:
            print("Error! theta_idx cannot be bigger than number of angle bins.")
            exit()
        if not self.conti:
            theta = theta * (2*np.pi / self.num_bins)
        push_center = poses[push_obj]
        pos_before = push_center - self.mov_dist * np.array([np.sin(theta), np.cos(theta)])
        px, py = self.pos2pixel(*pos_before)

        im_state, collision, contact, depth = self.push_from_pixel(px, py, theta)
        pre_poses = deepcopy(poses)
        poses, rotations = self.get_poses()

        info = {}
        info['target'] = -1
        info['goals'] = np.array(self.goals)
        info['contact'] = contact
        info['collision'] = collision
        info['pre_poses'] = np.array(pre_poses)
        info['poses'] = np.array(poses)
        info['rotations'] = np.array(rotations)
        info['goal_flags'] = np.linalg.norm(info['goals']-info['poses'], axis=1) < self.threshold
        info['out_of_range'] = not self.check_blocks_in_range()

        reward, done, block_success = self.get_reward(info)
        info['success'] = np.all(block_success)
        info['block_success'] = block_success

        self.step_count += 1
        if self.step_count==self.max_steps:
            done = True

        if self.detection:
            if self.env.camera_depth:
                return [im_state, depth], reward, done, info
            else:
                return [im_state], reward, done, info
        else:
            poses = info['poses']
            goals = info['goals']
            state_goal = [poses, goals]
            return state_goal, reward, done, info
