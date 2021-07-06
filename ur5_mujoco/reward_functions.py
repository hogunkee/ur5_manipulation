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


###################################################################

def reward_push_sparse(self, info):
    goals = info['goals']
    poses = info['poses']
    pre_poses = info['pre_poses']
    target = info['target']

    reward = 0.0
    success = []
    for obj_idx in range(self.num_blocks):
        dist = np.linalg.norm(poses[obj_idx] - goals[obj_idx])
        pre_dist = np.linalg.norm(pre_poses[obj_idx] - goals[obj_idx])
        if target==-1 or target==obj_idx:
            if dist < self.threshold:
                reward += 1.0
        success.append(dist<self.threshold)

    reward -= self.time_penalty
    done = np.array(success).all()
    return reward, done, success


def reward_push_linear(self, info):
    reward_scale = 100
    min_reward = -2
    goals = info['goals']
    poses = info['poses']
    pre_poses = info['pre_poses']
    target = info['target']

    reward = 0.0
    success = []
    for obj_idx in range(self.num_blocks):
        dist = np.linalg.norm(poses[obj_idx] - goals[obj_idx])
        pre_dist = np.linalg.norm(pre_poses[obj_idx] - goals[obj_idx])
        if target==-1 or target==obj_idx:
            reward += reward_scale * (pre_dist - dist)
            if dist < self.threshold:
                reward += 1.0
        success.append(dist<self.threshold)

    reward -= self.time_penalty
    reward = max(reward, min_reward)
    done = np.array(success).all()
    return reward, done, success


def reward_push_inverse(self, info):
    reward_scale = 20
    min_reward = -2
    goals = info['goals']
    poses = info['poses']
    pre_poses = info['pre_poses']
    target = info['target']

    reward = 0.0
    success = []
    for obj_idx in range(self.num_blocks):
        dist = np.linalg.norm(poses[obj_idx] - goals[obj_idx])
        pre_dist = np.linalg.norm(pre_poses[obj_idx] - goals[obj_idx])
        if target==-1 or target==obj_idx:
            reward += reward_scale * min(10, (1/dist - 1/pre_dist))
            if dist < self.threshold:
                reward += 1.0
        success.append(dist<self.threshold)

    reward -= self.time_penalty
    reward = max(reward, min_reward)
    done = np.array(success).all()
    return reward, done, success


def reward_push_binary(self, info):
    goals = info['goals']
    poses = info['poses']
    pre_poses = info['pre_poses']
    target = info['target']

    reward = 0.0
    success = []
    for obj_idx in range(self.num_blocks):
        dist = np.linalg.norm(poses[obj_idx] - goals[obj_idx])
        pre_dist = np.linalg.norm(pre_poses[obj_idx] - goals[obj_idx])
        if target==-1 or target==obj_idx:
            if dist < pre_dist - 0.001:
                reward += 1
            if dist > pre_dist + 0.001:
                reward -= 1
        success.append(dist<self.threshold)

    reward -= self.time_penalty
    done = np.array(success).all()
    return reward, done, success


def reward_push_new(self, info):
    goals = info['goals']
    poses = info['poses']
    pre_poses = info['pre_poses']
    target = info['target']

    reward = 0.0
    pre_success = []
    success = []
    for obj_idx in range(self.num_blocks):
        dist = np.linalg.norm(poses[obj_idx] - goals[obj_idx])
        pre_dist = np.linalg.norm(pre_poses[obj_idx] - goals[obj_idx])
        success.append(dist < self.threshold)
        pre_success.append(pre_dist < self.threshold)
        '''
        if target==-1 or target==obj_idx:
            if dist < pre_dist - 0.001:
                reward += 1
            if dist > pre_dist + 0.001:
                reward -= 1
        '''
        if np.linalg.norm(poses[obj_idx] - pre_poses[obj_idx]) > 0.01:
            reward += 1

    # fail to touch
    if reward == 0:
        reward -= 1

    # reach reward
    reward += 10. * (sum(success) - sum(pre_success))

    # collision
    for i in range(self.env.sim.data.ncon):
        contact = self.env.sim.data.contact[i]
        geom1 = self.env.sim.model.geom_id2name(contact.geom1)
        geom2 = self.env.sim.model.geom_id2name(contact.geom2)
        if geom1 is None or geom2 is None: continue
        if 'target_' in geom1 and 'target_' in geom2:
            if max(int(geom1[-1]), int(geom2[-1])) > self.num_blocks:
                continue
            reward -= 1.0 #self.contact_penalty

    done = np.array(success).all()
    return reward, done, success

####################### seperate rewards ###############################

def reward_sparse_seperate(self, info):
    goals = info['goals']
    poses = info['poses']
    pre_poses = info['pre_poses']
    target = info['target']

    rewards = []
    success = []
    for obj_idx in range(self.num_blocks):
        reward = 0.0
        dist = np.linalg.norm(poses[obj_idx] - goals[obj_idx])
        pre_dist = np.linalg.norm(pre_poses[obj_idx] - goals[obj_idx])
        if target==-1 or target==obj_idx:
            if dist < self.threshold:
                reward += 1.0
            if pre_dist < self.threshold and dist > self.threshold:
                reward -= 1.0
        reward -= self.time_penalty
        rewards.append(reward)
        success.append(dist<self.threshold)

    done = np.array(success).all()
    return rewards, done, success


def reward_linear_seperate(self, info):
    reward_scale = 20
    min_reward = -2
    goals = info['goals']
    poses = info['poses']
    pre_poses = info['pre_poses']
    target = info['target']

    rewards = []
    success = []
    for obj_idx in range(self.num_blocks):
        reward = 0.0
        dist = np.linalg.norm(poses[obj_idx] - goals[obj_idx])
        pre_dist = np.linalg.norm(pre_poses[obj_idx] - goals[obj_idx])
        if target==-1 or target==obj_idx:
            reward += reward_scale * np.clip(pre_dist - dist, -0.1, 0.1)
            if dist < self.threshold:
                reward += 1.0
            if pre_dist < self.threshold and dist > self.threshold:
                reward -= 1.0
        reward -= self.time_penalty
        reward = max(reward, min_reward)
        rewards.append(reward)
        success.append(dist<self.threshold)

    done = np.array(success).all()
    return rewards, done, success


def reward_inverse_seperate(self, info):
    reward_scale = 5
    min_reward = -2
    goals = info['goals']
    poses = info['poses']
    pre_poses = info['pre_poses']
    target = info['target']

    rewards = []
    success = []
    for obj_idx in range(self.num_blocks):
        reward = 0.0
        dist = np.linalg.norm(poses[obj_idx] - goals[obj_idx])
        pre_dist = np.linalg.norm(pre_poses[obj_idx] - goals[obj_idx])
        if target==-1 or target==obj_idx:
            reward += reward_scale * np.clip(1/(dist+0.1) - 1/(pre_dist+0.1), -2, 2)
            if dist < self.threshold:
                reward += 1.0
            if pre_dist < self.threshold and dist > self.threshold:
                reward -= 1.0
        reward -= self.time_penalty
        reward = max(reward, min_reward)
        rewards.append(reward)
        success.append(dist<self.threshold)

    done = np.array(success).all()
    return rewards, done, success


def reward_binary_seperate(self, info):
    goals = info['goals']
    poses = info['poses']
    pre_poses = info['pre_poses']
    target = info['target']

    rewards = []
    success = []
    for obj_idx in range(self.num_blocks):
        reward = 0.0
        dist = np.linalg.norm(poses[obj_idx] - goals[obj_idx])
        pre_dist = np.linalg.norm(pre_poses[obj_idx] - goals[obj_idx])
        if target==-1 or target==obj_idx:
            if dist < pre_dist - 0.001:
                reward += 1
            if dist > pre_dist + 0.001:
                reward -= 1
        reward -= self.time_penalty
        rewards.append(reward)
        success.append(dist<self.threshold)

    done = np.array(success).all()
    return rewards, done, success


def reward_new_seperate(self, info):
    goals = info['goals']
    poses = info['poses']
    pre_poses = info['pre_poses']
    target = info['target']

    rewards = []
    pre_success = []
    success = []
    for obj_idx in range(self.num_blocks):
        reward = 0.0
        dist = np.linalg.norm(poses[obj_idx] - goals[obj_idx])
        pre_dist = np.linalg.norm(pre_poses[obj_idx] - goals[obj_idx])
        success.append(dist < self.threshold)
        pre_success.append(pre_dist < self.threshold)
        '''
        if target==-1 or target==obj_idx:
            if dist < pre_dist - 0.001:
                reward += 1
            if dist > pre_dist + 0.001:
                reward -= 1
        reward -= self.time_penalty
        '''
        # touch reward
        if np.linalg.norm(poses[obj_idx] - pre_poses[obj_idx]) > 0.01:
            reward += 1
        else:
            reward -= 1 / self.num_blocks
        # reach reward
        reward += 10. * (int(success[-1]) - int(pre_success[-1]))

        rewards.append(reward)

    # collision
    for i in range(self.env.sim.data.ncon):
        contact = self.env.sim.data.contact[i]
        geom1 = self.env.sim.model.geom_id2name(contact.geom1)
        geom2 = self.env.sim.model.geom_id2name(contact.geom2)
        if geom1 is None or geom2 is None: continue
        if 'target_' in geom1 and 'target_' in geom2:
            if max(int(geom1[-1]), int(geom2[-1])) > self.num_blocks:
                continue
            rewards[int(geom1[-1])-1] -= 0.5
            rewards[int(geom2[-1])-1] -= 0.5

    done = np.array(success).all()
    return rewards, done, success
