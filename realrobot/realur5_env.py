from utils_ur5 import *

class UR5Robot(object):
    X_MIN = -0.5
    X_MAX = 0.5
    Y_MIN = -0.85
    Y_MAX = -0.3
    Z_MIN = 0.18
    Z_MAX = 0.8

    X_WS_MIN = -0.3
    X_WS_MAX = 0.3
    Y_WS_MIN = -0.75
    Y_WS_MAX = -0.35
    Z_WS_MIN = 0.19
    Z_WS_MAX = 0.25
    ROBOT_WS_INIT = [0, -0.5, 0.65]

    ARM_JOINT_NAME = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    ROBOT_INIT_POS = [0.0, -0.3, 0.65]
    ROBOT_INIT_ROTATION = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])

    def __init__(self, cam_id="025222072234"):
        self.mov_dist = 0.08

        self.cam_id = cam_id
        self.realsense = None
        self.set_realsense()

        self.planUR5 = None
        self.moveUR5 = None
        self.getEEFPose = None
        self.getJointStates = None
        self.get_ur5_control_service()

        self.T_eef_to_rs = np.load('rs_extrinsic_secondUR5.npy')
        #self.T_eef_to_rs = np.load('rs_extrinsic.npy')

    def set_realsense(self):
        self.realsense = RealSenseSensor(self.cam_id)
        self.K_rs = self.realsense._color_intrinsics
        self.D_rs = 0
        return

    def get_ur5_control_service(self):
        self.planUR5 = rospy.ServiceProxy('plan_robot_arm', JointTrajectory)
        rospy.wait_for_service('plan_robot_arm')
        self.moveUR5 = rospy.ServiceProxy('move_robot_arm', JointTrajectory)
        rospy.wait_for_service('move_robot_arm')
        self.getEEFPose = rospy.ServiceProxy('get_eef_pose', EndPose)
        rospy.wait_for_service('get_eef_pose')
        self.getJointStates = rospy.ServiceProxy('get_joint_states', JointStates)
        rospy.wait_for_service('get_joint_states')
        return 

    def get_joint_states(self):
        joints_Str = self.getJointStates().joint_states.replace('(', '').replace(')', '').split(', ')
        joints = [float(j) for j in joints_Str]
        return joints

    def get_eef_pose(self):
        pose = self.getEEFPose().eef_pose
        position = [pose.position.x, pose.position.y, pose.position.z]
        quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        return position, quaternion

    def move_to_pose(self, goal_pos, quat=[1,0,0,0], grasp=0.0):
        plans = self.moveUR5(self.ARM_JOINT_NAME, None, goal_pos, quat, 1-grasp)
        return plans

    def get_view(self, goal_pos=None, quat=[1, 0, 0, 0], grasp=0.0, show_img=False):
        # quat: xyzw
        if goal_pos is not None:
            plans = self.move_to_pose(goal_pos, quat, grasp)
            if len(plans.plan.points)<=1:
                print("Failed planning to the goal.")
                return None, None
        rospy.sleep(0.5)
        color, depth = self.realsense.frames(spatial=True, hole_filling=True, temporal=True)
        if show_img:
            plt.imshow(color)
            plt.show()
        return color, depth


    def pixel2pos(self, depth, goal_pixel):
        goal_pixel = np.array(goal_pixel)
        p_rs_to_goal = inverse_projection(depth, goal_pixel, self.K_rs, self.D_rs)
        #print(p_rs_to_goal)
        
        T_base_to_initeef = form_T(self.ROBOT_INIT_ROTATION, self.ROBOT_WS_INIT)
        T_rs_to_goal = form_T(np.eye(3), p_rs_to_goal)

        T_base_to_goal = T_base_to_initeef.dot(self.T_eef_to_rs.dot(T_rs_to_goal))    
        
        goal_position = T_base_to_goal[:3, 3]
        return goal_position

    def clip_pose(self, pose):
        pose[0] = np.clip(pose[0], self.X_MIN, self.X_MAX)
        pose[1] = np.clip(pose[1], self.Y_MIN, self.Y_MAX)
        pose[2] = np.clip(pose[2], self.Z_MIN, self.Z_MAX)
        return pose

    def clip_ws_pose(self, pose):
        pose[0] = np.clip(pose[0], self.X_WS_MIN, self.X_WS_MAX)
        pose[1] = np.clip(pose[1], self.Y_WS_MIN, self.Y_WS_MAX)
        pose[2] = np.clip(pose[2], self.Z_WS_MIN, self.Z_WS_MAX)
        return pose

    def push_from_pixel(self, depth, px, py, theta):
        if depth is None:
            color, depth = self.get_view(self.ROBOT_WS_INIT)
            return color, depth
        z_prepush = 0.25
        z_push = 0.2
        pos_before = np.array(self.pixel2pos(depth, [px, py]))
        pos_before = self.clip_ws_pose(pos_before)
        pos_after = pos_before + self.mov_dist * np.array([-np.sin(theta), -np.cos(theta), 0.])
        pos_after = self.clip_ws_pose(pos_after)

        x, y, z, w = euler2quat([-theta+np.pi/2, 0., 0.])
        quat = [w, x, y, z]
        self.move_to_pose([pos_before[0], pos_before[1], z_prepush], quat, grasp=1.0)
        self.move_to_pose([pos_before[0], pos_before[1], z_push], quat, grasp=1.0)
        self.move_to_pose([pos_after[0], pos_after[1], z_push], quat, grasp=1.0)
        self.move_to_pose([pos_after[0], pos_after[1], z_prepush], quat, grasp=1.0)
        color, depth = self.get_view(self.ROBOT_WS_INIT)
        return color, depth
        
    
class RealUR5Env(object):
    def __init__(self, ur5robot):
        self.goal_type = 'pixel'
        self.num_blocks = 3
        self.seg_target = None
        self.max_steps = 20
        self.num_bins = 8


        self.ur5 = ur5robot
        self.target = None
        self.target_color = None
        self.goal_scene = None
        self.goals = None
        self.goal_centers = None
        self.goal_colors = None

        self.timestep = 0
        self.color = None
        self.depth = None

    def set_target_with_color(self, color):
        # color: [R, G, B]
        target_obj= np.argmin(np.linalg.norm(self.goal_colors - color, axis=1))
        self.set_target(target_obj)

    def set_target(self, target):
        self.seg_target = target

    def reset(self):
        color, depth = self.ur5.get_view(self.ur5.ROBOT_WS_INIT)
        self.goal_scene = [color, depth]
        self.get_goal_representations()
        self.target = None
        self.target_color = None
        self.timestep = 0

        color, depth = self.ur5.push_from_pixel(None, 0, 0, 0)
        self.depth = depth
        return [[color, depth], self.goals]

    def get_goal_representations(self):
        goal_color, goal_depth = self.goal_scene
        masks, colors, fmask = self.get_masks(goal_color, goal_depth, n_clusters=3)
        goal_ims = []
        goal_centers = []
        for _mask in masks:
            zero_array = np.zeros([96, 96])
            _x, _y = np.where(_mask)
            mx = int(np.round(_x.mean()))
            my = int(np.round(_y.mean()))
            cv2.circle(zero_array, [mx, my], 1, 1, -1)
            goal_ims.append(zero_array)
            goal_centers.append([mx, my])
        self.goals = goal_ims
        self.goal_centers = np.array(goal_centers)
        self.goal_colors = colors
        return

    # for seg_fcdqn
    def step(self, action, mapping=True):
        px, py, theta_idx = action
        theta = theta_idx * (2*np.pi / 8)

        midx, midy = self.ur5.K_rs[:2, 2]
        if mapping:
            PX, PY = inverse_raw_pixel(np.array([px, py]), midx, midy, cs=480, ih=96, iw=96)
        else:
            PX, PY = px, py
        color, depth = self.ur5.push_from_pixel(self.depth, PX, PY, theta)
        self.depth = depth

        reward = 0.0
        done = False
        info = None
        return [[color, depth], self.goals], reward, done, info


class RealSegModule(object):
    def __init__(self):
        workspace_seg = np.zeros([96, 96])
        workspace_seg[14:-14, 14:-14] = 1.0
        self.workspace_seg = workspace_seg

    def get_masks(self, color, depth, n_cluster=3):
        fmask = (depth < 0.625).astype(float)

        my, mx = np.nonzero(fmask)
        points = list(zip(mx, my, np.ones_like(mx) * 96))
        z = (np.array(points).T / np.linalg.norm(points, axis=1)).T

        im_blur = cv2.blur(image, (5, 5))
        colors = np.array([im_blur[y, x] / (10 * 255) for x, y in zip(mx, my)])
        z_color = np.concatenate([z, colors], 1)
        clusters = SpectralClustering(n_clusters=n_cluster, n_init=10).fit_predict(z_color)

        new_mask = np.zeros([fmask.shape[0], fmask.shape[1], n_cluster])
        for x, y, c in zip(mx, my, clusters):
            new_mask[y, x, c] = 1
        masks = new_mask.transpose([2,0,1]).astype(float)

        colors = []
        for mask in masks:
            color = image[mask.astype(bool)].mean(0) / 255.
            colors.append(color)

        return masks, np.array(colors), fmask

    '''
    def generate_state(self, masks, colors):
        if self.target_color is not None:
            t_obj = np.argmin(np.linalg.norm(colors - self.target_color, axis=1))
        else:
            t_obj = np.random.randint(len(masks))
        target_seg = masks[t_obj]
        obstacle_seg = np.any([masks[o] for o in range(len(masks)) if o != t_obj], 0)
        self.target_color = colors[t_obj]

        state = np.concatenate([target_seg, obstacle_seg, workspace_seg]).reshape(-1, 96, 96)
        goal = self.goals[target_idx: target_idx+1]
        return state, goal
    '''


class UR5Calibration(object):
    calib_positions = np.array([
        [0.0, -0.3, 0.65],
        [0.282, -0.413, 0.606],
        [0.249, -0.217, 0.567],
        [-0.193, -0.278, 0.495],
        [-0.249, -0.528, 0.45]
    ])
    # xyzw quaternion
    calib_quaternions = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.9917, 0.0041, -0.1246, 0.0321],
        [0.9626, 0.1529, -0.205, 0.09],
        [0.9697, -0.1399, 0.191, 0.0606],
        [0.9685, -0.1086, 0.2076, -0.0849]
    ])

    def __init__(self, ur5robot, method='leastsquares'):
        self.ur5 = ur5robot
        self.method = 'leastsquares'

        if self.method=='ukf':
            self.solve = self.calib_ukf
        elif self.method=='iekf':
            self.solve = self.calib_iekf
        else:
            self.solve = self.calib_leastsquares

    # for calibration
    def get_cam_theta_p(self, color, show_img=True):
        objpoints_rs, corners_rs = get_chessboard_corners(color)
        err_value, R_vec_cam_to_chess, p_cam_to_chess = cv2.solvePnP(objpoints_rs, corners_rs, K_rs, D_rs)
        img = cv2.drawChessboardCorners(np.array(color), CHECKERBOARD, corners_rs, True)
        if show_img:
            plt.imshow(img)
            plt.show()
        return R_vec_cam_to_chess, p_cam_to_chess

    def get_cam_R_T(self, color, show_img=True):
        objpoints_rs, corners_rs = get_chessboard_corners(color)
        img = cv2.drawChessboardCorners(np.array(color), CHECKERBOARD, corners_rs, True)
        if show_img:
            plt.imshow(img)
            plt.show()
        err_value, R_vec_cam_to_chess, p_cam_to_chess = cv2.solvePnP(objpoints_rs, corners_rs, K_rs, D_rs)
        R_cam_to_chess, _ = cv2.Rodrigues(R_vec_cam_to_chess)
        T_cam_to_chess = form_T(R_cam_to_chess, p_cam_to_chess)
        return R_cam_to_chess, T_cam_to_chess

    def get_calib_transformations(self):
        T_ur5 = []
        T_realsense = []

        for p, q in zip(self.calib_positions, self.calib_quaternions):
            img, _ = self.ur5.get_view(p, q)
            R, T_cam_to_chess = self.get_cam_R_T(img)
            T_chess_to_cam = np.linalg.inv(T_cam_to_chess)
            
            q_wxyz = [q[3], q[0], q[1], q[2]]
            T_base_to_eef = form_T(quaternion_matrix(q_wxyz)[:3, :3], p)
            
            T_ur5.append(T_base_to_eef)
            T_realsense.append(T_chess_to_cam)
            rospy.sleep(1.0)

        img, _ = self.ur5.get_view(self.ur5.ROBOT_INIT_POS)
        return T_ur5, T_realsense
    
    def form_AB_matrices(self, T_ur5, T_realsense):
        A = []
        B = []

        for i, Ti in enumerate(T_ur5):
            for j, Tj in enumerate(T_ur5):
                if i==j: continue
                A.append(np.linalg.inv(Ti).dot(Tj))
                
        for i, Ti in enumerate(T_realsense):
            for j, Tj in enumerate(T_realsense):
                if i==j: continue
                B.append(np.linalg.inv(Ti).dot(Tj))
                #A.append(Ti.dot(np.linalg.inv(Tj)))

        A = np.array(A).transpose([1,2,0])
        B = np.array(B).transpose([1,2,0])
        return A, B

    def get_calib_error(self, A, B, X):
        # A: T_ur5
        # B: T_realsense
        # X: T_eef_to_rs
        error = 0
        for i in range(A.shape[-1]):
            err = np.linalg.norm(A[:,:,i].dot(X) - X.dot(B[:,:,i]))
            error += err
        return error

    def calib_ukf(self, A, B):
        ukf=UKF()
        for i in range(len(A.shape[-1])):
            AA=A[:,:,i] 
            BB=B[:,:,i]
            ukf.Update(AA,BB)

        theta=np.linalg.norm(ukf.x[:3])
        if theta < EPS:
            k=[0,1,0] #VRML standard
        else:
            k=ukf.x[0:3]/np.linalg.norm(ukf.x[:3])
        euler_ukf=Tools.mat2euler(Tools.vec2rotmat(theta, k))
        print('.....UKF Results')

        print('Euler:', np.array(euler_ukf)*180/np.pi)
        print('Translation:', ukf.x[3:])
        print('------------------------------')
        T_eef_to_rs = form_T(Tools.vec2rotmat(theta, k), ukf.x[3:])
        print(T_eef_to_rs)

        print('------------------------------')
        print('Quat:', Rotation.from_matrix(T_eef_to_rs[:3,:3]).as_quat())
        print('Euler(degree):', Rotation.from_matrix(T_eef_to_rs[:3,:3]).as_euler('zyx', degrees=True))
        print('------------------------------')
        print('Error:', self.get_calib_error(T_eef_to_rs))
        return T_eef_to_rs

    def calib_iekf(self, A, B):
	#IEKF
        iekf=IEKF()
        for i in range(len(A.shape[-1])):
            AA=A[:,:,i] 
            BB=B[:,:,i]
            iekf.Update(AA,BB)
            
        theta=np.linalg.norm(iekf.x[:3])
        if theta < EPS:
            k=[0,1,0] #VRML standard
        else:
            k=iekf.x[0:3]/np.linalg.norm(iekf.x[:3])
        euler_iekf=Tools.mat2euler(Tools.vec2rotmat(theta, k))

        print('IEKF Results')

        print('Euler:', np.array(euler_iekf)*180/np.pi)
        print('Translation:', iekf.x[3:])
        print('------------------------------')
        T_eef_to_rs = form_T(Tools.vec2rotmat(theta, k), iekf.x[3:])
        print(T_eef_to_rs)

        print('------------------------------')
        print('Quat:', Rotation.from_matrix(T_eef_to_rs[:3,:3]).as_quat())
        print('Euler(degree):', Rotation.from_matrix(T_eef_to_rs[:3,:3]).as_euler('zyx', degrees=True))
        print('------------------------------')
        print('Error:', self.get_calib_error(T_eef_to_rs))
        return T_eef_to_rs

    def calib_leastsquares(self, A, B):
        T_eef_to_rs = LeastSquaresAXXB(A.transpose([2, 0, 1]), B.transpose([2, 0, 1]), verbose=True)

        print(T_eef_to_rs)
        print('------------------------------')
        print('Quat:', Rotation.from_matrix(T_eef_to_rs[:3,:3]).as_quat())
        print('Euler(degree):', Rotation.from_matrix(T_eef_to_rs[:3,:3]).as_euler('zyx', degrees=True))
        print('------------------------------')
        print('Error:', self.get_calib_error(T_eef_to_rs))
        return T_eef_to_rs
