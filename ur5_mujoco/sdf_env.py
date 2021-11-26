from pushpixel_env import *
from reward_functions import *
import cv2
import imageio
from transform_utils import euler2quat, quat2mat

class SDFEnv(pushpixel_env):
    def __init__(self, ur5_env, num_blocks=1, mov_dist=0.05, max_steps=50, reward_type='binary', conti=False):
        self.conti = conti
        super().__init__(ur5_env, num_blocks, mov_dist, max_steps, 1, reward_type, 'block', False, False)

    def get_sdf(self, image):
        seg_masks, colors, features = get_seg(image) # TODO
        sdfs = []
        for seg in seg_masks:
            sd = skfmm.distance(seg.astype(int) - 0.5, dx=1)
            sdfs.append(sd)
        sdf_info = {
                'sdfs': np.array(sdfs),
                'masks': seg_masks, 
                'colors': colors,
                'features': features
                }
        return sdf_info
    
    def get_state(self, image):
        sdf_info = self.get_sdf(image)
        poses = []
        for sdf in sdf_info['sdfs']:
            pixels = np.where(sdf==sdf.max())
            cx = np.mean(pixels[0])
            cy = np.mean(pixels[1])
            poses.append([cx, cy])
        sdf_info['poses'] = poses
        return sdf_info

    def reset(self):
        im_state = self.init_env()
        state = self.get_state(im_state)
        goal_state = self.get_state(self.goal_image)
        return [state, goal_state]

    def step(self, action): # action: (px, py, theta)
        px, py, theta = action
        if theta >= self.num_bins:
            print("Error! theta_idx cannot be bigger than number of angle bins.")
            exit()
        if not self.conti:
            theta = theta * (2*np.pi / self.num_bins)

        im_state, collision, contact, depth = self.push_from_pixel(px, py, theta)
        pre_poses = deepcopy(poses)
        poses, rotations = self.get_poses()

        info = {}
        info['target'] = -1
        info['action'] = action
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
                return [im_state, self.goal_image], reward, done, info
        else:
            poses = info['poses']
            goals = info['goals']
            state_goal = [poses, goals]
            return [state_goal, im_state], reward, done, info
