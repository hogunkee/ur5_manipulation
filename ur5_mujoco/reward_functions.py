import numpy as np

def reward_touch(self, info):
    pre_poses = np.array(info['pre_poses'])
    poses = np.array(info['poses'])
    if np.linalg.norm(pre_poses - poses) > 1e-3:
        reward = 1.0
        done = False
    else:
        reward = -self.time_penalty
        done = False
    return reward, done

def reward_targetpush(self, info):
    target_idx = list(info['obj_indices']).index(info['target_obj'])
    poses = np.array(info['poses'])
    check_near = np.linalg.norm(poses, axis=1) < 0.05
    info['success'] = False
    if check_near.any():
        done = True
        reached_idx = list(check_near).index(True)
        if reached_idx==target_idx:
            reward = 1.0
            info['success'] = True
        else:
            reward = 0.0
        reached_obj = info['obj_indices'][reached_idx]
        info['reached_obj'] = reached_obj
    else:
        reward = -self.time_penalty
        done = False
        info['reached_obj'] = -1
    return reward, done

def reward_reach(self):
    target_pos = self.env.sim.data.get_body_xpos('target_body_1')
    if np.linalg.norm(target_pos - self.pre_target_pos) > 1e-3:
        reward = 1.0
        done = False #True
    else:
        reward = -self.time_penalty
        done = False
    return reward, done

def reward_push_sparse(self):
    done = False
    reward = 0.0
    if self.num_blocks >= 1:
        pos1 = self.env.sim.data.get_body_xpos('target_body_1')[:2]
        if np.linalg.norm(pos1 - self.goal1) < self.threshold:
            reward += 1.0
    if self.num_blocks >= 2:
        pos2 = self.env.sim.data.get_body_xpos('target_body_2')[:2]
        if np.linalg.norm(pos2 - self.goal2) < self.threshold:
            reward += 1.0
    if self.num_blocks >= 3:
        pos3 = self.env.sim.data.get_body_xpos('target_body_3')[:2]
        if np.linalg.norm(pos3 - self.goal3) < self.threshold:
            reward += 1.0

    if reward >= self.num_blocks:
        done = True
    reward += -self.time_penalty
    return reward, done

def reward_push_dense(self, info):
    reward_scale = 10
    min_reward = -2
    goals = info['goals']
    poses = info['poses']
    pre_poses = info['pre_poses']

    reward = 0.0
    success = []
    for obj_idx in range(self.num_blocks):
        dist = np.linalg.norm(poses[obj_idx] - self.goals[obj_idx])
        pre_dist = np.linalg.norm(pre_poses[obj_idx] - goals[obj_idx])
        if dist < self.threshold:
            reward += 1.0
        else:
            reward += reward_scale * (pre_dist - dist)
        success.append(int(dist<self.threshold))

    reward -= self.time_penalty
    reward = max(reward, min_reward)
    done = (np.sum(success) >= self.num_blocks)
    return reward, done


def reward_push_reverse(self, info):
    reward_scale = 0.5
    min_reward = -2
    goals = info['goals']
    poses = info['poses']
    pre_poses = info['pre_poses']

    reward = 0.0
    success = []
    for obj_idx in range(self.num_blocks):
        dist = np.linalg.norm(poses[obj_idx] - self.goals[obj_idx])
        pre_dist = np.linalg.norm(pre_poses[obj_idx] - goals[obj_idx])
        reward += reward_scale * min(10, (1/dist - 1/pre_dist))
        success.append(int(dist<self.threshold))

    reward -= self.time_penalty
    reward = max(reward, min_reward)
    done = (np.sum(success) >= self.num_blocks)
    return reward, done


def reward_push_binary(self, info):
    goals = info['goals']
    poses = info['poses']
    pre_poses = info['pre_poses']

    reward = 0.0
    success = []
    for obj_idx in range(self.num_blocks):
        dist = np.linalg.norm(poses[obj_idx] - goals[obj_idx])
        pre_dist = np.linalg.norm(pre_poses[obj_idx] - goals[obj_idx])
        if dist < pre_dist - 0.001:
            reward += 1
        if dist > pre_dist + 0.001:
            reward -= 1
        success.append(int(dist<self.threshold))

    reward -= self.time_penalty
    done = (np.sum(success) >= self.num_blocks)
    return reward, done
