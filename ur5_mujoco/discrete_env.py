from ur5_env import *
import cv2


class discrete_env(object):
    def __init__(self, ur5_env, num_blocks=1, mov_dist=0.03, max_steps=50, task=0):
        self.env = ur5_env 
        self.num_blocks = num_blocks

        self.task = task # 0: Reach / 1: Push
        self.mov_dist = mov_dist
        self.range_x = [-0.3, 0.3]
        self.range_y = [-0.2, 0.4]
        self.z_min = 1.05
        self.z_max = self.z_min + self.mov_dist
        self.time_penalty = 1e-2
        self.max_steps = max_steps
        self.step_count = 0
        self.threshold = 0.1

        self.init_pos = [0.0, 0.0, self.z_min + self.mov_dist]

        self.cam_id = 1
        self.cam_theta = 30 * np.pi / 180
        # cam_mat = self.env.sim.data.get_camera_xmat("rlview")
        # cam_pos = self.env.sim.data.get_camera_xpos("rlview")

        self.colors = np.array([ 
            [0.6784, 1.0, 0.1843], 
            [0.93, 0.545, 0.93], 
            [0.9686, 0.902, 0] 
            ])

        self.init_env()

    def init_env(self):
        self.env._init_robot()
        range_x = self.range_x
        range_y = self.range_y
        self.env.sim.data.qpos[12:15] = [0, 0, 0]
        self.env.sim.data.qpos[19:22] = [0, 0, 0]
        self.env.sim.data.qpos[26:29] = [0, 0, 0]
        self.goal1 = [0., 0.]
        self.goal2 = [0., 0.]
        self.goal3 = [0., 0.]
        self.goal_image = np.zeros([self.env.camera_height, self.env.camera_width, 3])
        if self.num_blocks >= 1:
            tx1 = 0.1 #np.random.uniform(*range_x)
            ty1 = 0.1 #np.random.uniform(*range_y)
            tz1 = 0.9
            self.env.sim.data.qpos[12:15] = [tx1, ty1, tz1]
            gx1 = np.random.uniform(*range_x)
            gy1 = np.random.uniform(*range_y)
            self.goal1 = [gx1, gy1]
            cv2.circle(self.goal_image, self.pos2pixel(*self.goal1), 1, self.colors[0], -1)
        if self.num_blocks >= 2:
            tx2 = np.random.uniform(*range_x)
            ty2 = np.random.uniform(*range_y)
            tz2 = 0.9
            self.env.sim.data.qpos[19:22] = [tx2, ty2, tz2]
            gx2 = np.random.uniform(*range_x)
            gy2 = np.random.uniform(*range_y)
            self.goal2 = [gx2, gy2]
            cv2.circle(self.goal_image, self.pos2pixel(*self.goal2), 1, self.colors[1], -1)
        if self.num_blocks >= 3:
            tx3 = np.random.uniform(*range_x)
            ty3 = np.random.uniform(*range_y)
            tz3 = 0.9
            self.env.sim.data.qpos[26:29] = [tx3, ty3, tz3]
            gx3 = np.random.uniform(*range_x)
            gy3 = np.random.uniform(*range_y)
            self.goal3 = [gx3, gy3]
            cv2.circle(self.goal_image, self.pos2pixel(*self.goal3), 1, self.colors[2], -1)

        im_state = self.env.move_to_pos(self.init_pos, grasp=1.0)
        self.step_count = 0

        return im_state

    def reset(self):
        glfw.destroy_window(self.env.viewer.window)
        self.env.viewer = None
        im_state = self.init_env()
        gripper_height = self.get_gripper_state()
        if self.task==0:
            return im_state, gripper_height
        else:
            return im_state, self.goal_image, gripper_height

    def step(self, action, grasp=1.0):
        self.pre_gripper_pos = deepcopy(self.env.sim.data.mocap_pos[0])
        self.pre_target_pos = deepcopy(self.env.sim.data.get_body_xpos('target_body_1'))
        z_check_collision = self.z_min + 0.025

        dist = self.mov_dist
        dist2 = dist/np.sqrt(2)
        if action==0:
            gripper_height = self.get_gripper_state()
            if gripper_height==0:
                im_state = self.env.move_pos_diff([0.0, 0.0, 0.0], grasp=grasp)
            else:
                gripper_pos = deepcopy(self.pre_gripper_pos)
                gripper_pos[2] = z_check_collision
                self.env.move_to_pos(gripper_pos, grasp=grasp)
                force = self.env.sim.data.sensordata
                if np.abs(force[2]) > 1.0 or np.abs(force[5]) > 1.0:
                    print("Collision!")
                    gripper_pos[2] = self.z_max
                    im_state = self.env.move_to_pos(gripper_pos, grasp=grasp)
                else:
                    gripper_pos[2] = self.z_min
                    im_state = self.env.move_to_pos(gripper_pos, grasp=grasp)

            '''
            if self.pre_mocap_pos[2]-dist < self.z_min:
                im_state = self.env.move_pos_diff([0.0, 0.0, -self.pre_mocap_pos[2]+self.z_min], grasp=grasp)
            else:
                im_state = self.env.move_pos_diff([0.0, 0.0, -dist], grasp=grasp)
            '''
        elif action==5:
            gripper_pos = deepcopy(self.pre_gripper_pos)
            gripper_pos[2] = self.z_max
            im_state = self.env.move_to_pos(gripper_pos, grasp=grasp)
            '''
            if self.pre_mocap_pos[2]+dist > self.z_max:
                im_state = self.env.move_pos_diff([0.0, 0.0, -self.pre_mocap_pos[2]+self.z_max], grasp=grasp)
            else:
                im_state = self.env.move_pos_diff([0.0, 0.0, dist], grasp=grasp)
            '''
        else:
            gripper_pos = deepcopy(self.pre_gripper_pos)
            if action==8:
                gripper_pos[1] = np.min([gripper_pos[1] + dist, self.range_y[1]])
                #im_state = self.env.move_pos_diff([0.0, dist, 0.0], grasp=grasp)
            elif action==9:
                gripper_pos[0] = np.min([gripper_pos[0] + dist2, self.range_x[1]])
                gripper_pos[1] = np.min([gripper_pos[1] + dist2, self.range_y[1]])
                #im_state = self.env.move_pos_diff([dist2, dist2, 0.0], grasp=grasp)
            elif action==6:
                gripper_pos[0] = np.min([gripper_pos[0] + dist, self.range_x[1]])
                #im_state = self.env.move_pos_diff([dist, 0.0, 0.0], grasp=grasp)
            elif action==3:
                gripper_pos[0] = np.min([gripper_pos[0] + dist2, self.range_x[1]])
                gripper_pos[1] = np.max([gripper_pos[1] - dist2, self.range_y[0]])
                #im_state = self.env.move_pos_diff([dist2, -dist2, 0.0], grasp=grasp)
            elif action==2:
                gripper_pos[1] = np.max([gripper_pos[1] - dist, self.range_y[0]])
                #im_state = self.env.move_pos_diff([0.0, -dist, 0.0], grasp=grasp)
            elif action==1:
                gripper_pos[0] = np.max([gripper_pos[0] - dist2, self.range_x[0]])
                gripper_pos[1] = np.max([gripper_pos[1] - dist2, self.range_y[0]])
                #im_state = self.env.move_pos_diff([-dist2, -dist2, 0.0], grasp=grasp)
            elif action==4:
                gripper_pos[0] = np.max([gripper_pos[0] - dist, self.range_x[0]])
                #im_state = self.env.move_pos_diff([-dist, 0.0, 0.0], grasp=grasp)
            elif action==7:
                gripper_pos[0] = np.max([gripper_pos[0] - dist2, self.range_x[0]])
                gripper_pos[1] = np.min([gripper_pos[1] + dist2, self.range_y[1]])
                #im_state = self.env.move_pos_diff([-dist2, dist2, 0.0], grasp=grasp)
            im_state = self.env.move_to_pos(gripper_pos, grasp=grasp)

        gripper_height = self.get_gripper_state()
        reward, done = self.get_reward()

        self.step_count += 1
        if self.step_count==self.max_steps:
            done = True

        #sim = self.env.sim
        #print('force: {}'.format(sim.data.sensordata))
        # left_finger_idx = sim.model.body_name2id('left_inner_finger')
        # right_finger_idx = sim.model.body_name2id('right_inner_finger')
        # right_contact_force = sim.data.efc_force[right_finger_idx]
        # # print('left: {} / right: {}'.format(left_contact_force, right_contact_force))
        # print(sim.data.get_sensor('left_finger_force'))

        if self.task == 0:
            return [im_state, gripper_height], reward, done, None
        else:
            return [im_state, self.goal_image, gripper_height], reward, done, None

    def get_gripper_state(self):
        # return grasp_height, gripper_close
        # return self.env.sim.data.mocap_pos[0][2], int(bool(sum(self.env.sim.data.ctrl)))
        if self.env.sim.data.mocap_pos[0][2] > self.z_min + self.mov_dist/2:
            return 1.0
        else: 
            return 0.0

    def get_reward(self):
        # 0: Reach   #
        # 1: Spread  #
        # 2: Gather  #
        # 3: Line up #
        ## Reach ##
        if self.task == 0:
            target_pos = self.env.sim.data.get_body_xpos('target_body_1')
            if np.linalg.norm(target_pos - self.pre_target_pos) > 1e-3:
                reward = 1.0
                done = True
            else:
                reward = -self.time_penalty
                done = False
        ## Push ##
        elif self.task == 1:
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

    def pixel2pos(self, u, v):
        theta = self.cam_theta
        cx, cy, cz = self.env.sim.model.cam_pos[self.cam_id]
        fovy = self.env.sim.model.cam_fovy[self.cam_id]
        f = 0.5 * self.env.camera_height / np.tan(fovy * np.pi / 360)
        u0 = 0.5 * self.env.camera_width
        v0 = 0.5 * self.env.camera_height
        z0 = 0.9  # table height
        y_cam = (cz - z0) / (np.sin(theta) + np.cos(theta) * f / (v - v0 + 1e-10))
        x_cam = (u - u0) / (v - v0 + 1e-10) * y_cam
        x = - x_cam
        y = np.tan(theta) * (z0 - cz) + cy + 1 / np.cos(theta) * y_cam
        z = z0
        # print("cam pos:", [x_cam, y_cam])
        # print("world pos:", [x, y])
        # print()
        return x, y, z

    def pos2pixel(self, x, y):
        theta = self.cam_theta
        cx, cy, cz = self.env.sim.model.cam_pos[self.cam_id]
        fovy = self.env.sim.model.cam_fovy[self.cam_id]
        f = 0.5 * self.env.camera_height / np.tan(fovy * np.pi / 360)
        u0 = 0.5 * self.env.camera_width
        v0 = 0.5 * self.env.camera_height
        z0 = 0.9  # table height
        y_cam = np.cos(theta) * (y - cy - np.tan(theta) * (z0 - cz))
        dv = f * np.cos(theta) / ((cz - z0) / y_cam - np.sin(theta))
        v = dv + v0
        u = - dv * x / y_cam + u0
        return int(u), int(v)

    def move2pixel(self, u, v):
        target_pos = np.array(self.pixel2pos(u, v))
        target_pos[2] = 1.05
        frame = self.env.move_to_pos(target_pos)
        plt.imshow(frame)
        plt.show()

if __name__=='__main__':
    env = UR5Env(render=True, camera_height=64, camera_width=64, control_freq=5)
    env = discrete_env(env, num_blocks=3, mov_dist=0.03, max_steps=100)
    frame = env.reset()
    '''
    img = cv2.circle(frame[0].astype(np.float32), (10, 50), 1, [255, 0, 0], 1)
    plt.imshow(img/255)
    plt.show()

    env.move2pixel(48, 48)
    env.move2pixel(48, 60)
    env.move2pixel(48, 70)
    env.move2pixel(20, 48)
    env.move2pixel(20, 20)
    env.move2pixel(0, 20)
    '''
    frames = []

    for i in range(100):
        #action = [np.random.randint(6), np.random.randint(2)]
        try:
            action = [int(input("action? ")), 1]
        except KeyboardInterrupt:
            exit()
        except:
            continue
        if action[0] > 9:
            continue
        print('{} steps. action: {}'.format(env.step_count, action))
        states, reward, done, info = env.step(*action)
        plt.imshow(states[0])
        plt.show()
        print('Reward: {}. Done: {}'.format(reward, done))
        #show_image(states[0])
        if done:
            print('Done. New episode starts.')
            env.reset()
