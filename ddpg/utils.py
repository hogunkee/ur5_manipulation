import numpy as np
import torch
import shutil
import torch.autograd as Variable
from copy import deepcopy


def soft_update(target, source, tau):
	"""
	Copies the parameters from source network (x) to target network (y) using the below update
	y = TAU*x + (1 - TAU)*y
	:param target: Target network (PyTorch)
	:param source: Source network (PyTorch)
	:return:
	"""
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(
			target_param.data * (1.0 - tau) + param.data * tau
		)


def hard_update(target, source):
	"""
	Copies the parameters from source network to target network
	:param target: Target network (PyTorch)
	:param source: Source network (PyTorch)
	:return:
	"""
	for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(param.data)


def save_training_checkpoint(state, is_best, episode_count):
	"""
	Saves the models, with all training parameters intact
	:param state:
	:param is_best:
	:param filename:
	:return:
	"""
	filename = str(episode_count) + 'checkpoint.path.rar'
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'model_best.pth.tar')


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:

	def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2):
		self.action_dim = action_dim
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.X = np.ones(self.action_dim) * self.mu

	def reset(self):
		self.X = np.ones(self.action_dim) * self.mu

	def sample(self):
		dx = self.theta * (self.mu - self.X)
		dx = dx + self.sigma * np.random.randn(len(self.X))
		self.X = self.X + dx
		return self.X

def get_action_near_blocks(env, pad=0.06):
    poses, _ = env.get_poses()
    obj = np.random.randint(len(poses))
    pose = poses[obj]
    x = np.random.uniform(pose[0]-pad, pose[0]+pad)
    y = np.random.uniform(pose[1]-pad, pose[1]+pad)
    py, px = env.pos2pixel(x, y)
    a0 = 2 * px / env.env.camera_width - 1
    a1 = 2 * py / env.env.camera_width - 1
    a2 = 50 * (- x + pose[0]) + np.random.normal(0, 0.1)
    a3 = 50 * (- y + pose[1]) + np.random.normal(0, 0.1)
    action = np.array([a0, a1, a2, a3])
    return action

def sample_her_transitions(env, info):
    _info = deepcopy(info)
    move_threshold = 0.005
    range_x = env.block_range_x
    range_y = env.block_range_y

    pre_poses = info['pre_poses']
    poses = info['poses']
    pos_diff = np.linalg.norm(poses - pre_poses, axis=1)
    if np.linalg.norm(poses - pre_poses) < move_threshold:
        return []

    for i in range(env.num_blocks):
        if pos_diff[i] < move_threshold:
            continue
        ## 1. archived goal ##
        archived_goal = poses[i]

        ## clipping goal pose ##
        x, y = archived_goal
        _info['goals'][i] = np.array([x, y])

    ## recompute reward  ##
    reward_recompute, done_recompute, block_success_recompute = env.get_reward(_info)
    if _info['out_of_range']:
        if env.seperate:
            reward_recompute = [-1.] * env.num_blocks
        else:
            reward_recompute = -1.

    return [[reward_recompute, _info['goals'].flatten(), done_recompute, block_success_recompute]]

def sample_ig_transitions(env, info, num_samples=1):
    move_threshold = 0.005
    range_x = env.block_range_x
    range_y = env.block_range_y
    n_blocks = env.num_blocks

    pre_poses = info['pre_poses']
    poses = info['poses']
    pos_diff = np.linalg.norm(poses - pre_poses, axis=1)
    if np.linalg.norm(poses - pre_poses) < move_threshold:
        return []

    transitions = []
    for s in range(num_samples):
        _info = deepcopy(info)
        goal_image = deepcopy(env.background_img)
        for i in range(n_blocks):
            if pos_diff[i] < move_threshold:
                continue
            ## 1. archived goal ##
            gx = np.random.uniform(*range_x)
            gy = np.random.uniform(*range_y)
            archived_goal = np.array([gx, gy])
            _info['goals'][i] = archived_goal

        ## recompute reward  ##
        reward_recompute, done_recompute, block_success_recompute = env.get_reward(_info)
        if _info['out_of_range']:
            if env.seperate:
                reward_recompute = [-1.] * env.num_blocks
            else:
                reward_recompute = -1.
        transitions.append([reward_recompute, _info['goals'].flatten(), done_recompute, block_success_recompute])

    return transitions

# use this to plot Ornstein Uhlenbeck random motion
if __name__ == '__main__':
	ou = OrnsteinUhlenbeckActionNoise(1)
	states = []
	for i in range(1000):
		states.append(ou.sample())
	import matplotlib.pyplot as plt

	plt.plot(states)
	plt.show()
