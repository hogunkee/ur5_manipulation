import copy
import rospy

from calibration_helper import *
from matplotlib import pyplot as plt

from transformations import quaternion_matrix
from transformations import quaternion_from_matrix
from transformations import rotation_matrix

# ros related packages
#sys.path.append('/usr/lib/python2.7/dist-packages')
from ur_msgs.srv import JointTrajectory, EndPose, JointStates

# calibration
sys.path.append('/home/scarab5/Desktop/Pose-Estimation-for-Sensor-Calibration')
from axxb_solver import *
from Pose_Estimation_Class import *


def get_transformation_from_pose(ros_pose):
    ros_position = ros_pose['position']
    ros_orientation = ros_pose['orientation']
    quaternion = np.array([ros_orientation.x, ros_orientation.y, ros_orientation.z, ros_orientation.w])
    return form_T(quaternion_matrix(quaternion)[:3,:3],ros_position)   

def inverse_projection(depth_img, pixel, K_realsense, D_realsense):
    depth = depth_img[pixel[1], pixel[0]]
    point = pixel2cam(pixel.reshape([-1,1]), depth, K_realsense, D_realsense)
    return np.array([point[0], point[1], depth])

def get_input_rgbd_image(realsense,
                         crop_size = 430,
                         resize_width = 256,
                         resize_height = 256):
    realsense.stop()
    realsense.start()
    rospy.sleep(0.2)
    rgb_im, depth_im, hole_mask = realsense._read_color_and_depth_image(apply_hole_filling=True,apply_temporal=False)
    realsense.stop()
    
    midx, midy = realsense._intrinsics[:2,2]
    input_rgb_im = resize_image(crop_image(rgb_im, midx, midy, crop_size), resize_width, resize_height)
    input_depth_im = resize_image(crop_image(depth_im, midx, midy, crop_size), resize_width, resize_height)
    input_hole_mask = resize_image(crop_image(hole_mask, midx, midy, crop_size), resize_width, resize_height)
    return input_rgb_im, input_depth_im, input_hole_mask, rgb_im, depth_im, hole_mask

def crop_image(image, midx, midy, crop_size=430):
    cs = crop_size
    cropped_image = image[int(np.round(midy-cs/2)):int(np.round(midy+cs/2)), int(np.round(midx-cs/2)):int(np.round(midx+cs/2))]
    return cropped_image

def resize_image(image, resize_width=256, resize_height=256):
    return cv2.resize(image, (resize_width, resize_height), interpolation=cv2.INTER_AREA)

def scaling_segmask(segmask,
                        realsense,
                        crop_size = 430,
                        resize_width = 256,
                        resize_height = 256):

        midx, midy = realsense._intrinsics[:2,2]
        input_segmask = resize_image(crop_image(segmask, midx, midy, crop_size), resize_width, resize_height)
        return input_segmask

def get_mask_of_square_bin_from_rgb(realsense,rgb_im, minium_bin_pixel_area=13000, show_image=True, verbose=True):
    iw, ih = rgb_im.shape[1], rgb_im.shape[0]
    rgb_= copy.deepcopy(rgb_im)
    _, edged = cv2.threshold(rgb_[:,:,0],20,255, cv2.THRESH_BINARY_INV)
    if show_image:
        plt.imshow(rgb_im)
        plt.show()
        plt.imshow(edged)
        plt.show()

    _, contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_area_segmask = np.inf

    if verbose:
        print("Number of detected contours : %d"%len(contours))

    segmask_contour = None

    for trial in range(100):
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
            area = cv2.contourArea(approx)
            x = approx.ravel()[0]
            y = approx.ravel()[1]
            if area > minium_bin_pixel_area:
                if verbose:
                    print('Bin is detected!')
                if area < min_area_segmask:
                    min_area_segmask = area
                    segmask_contour = approx
                rgb_ = copy.deepcopy(rgb_im)
                if show_image:
                    cont = cv2.drawContours(rgb_, [approx], 0, (255, 0, 0), 10)
                    plt.imshow(cont)
                    plt.show()
        if segmask_contour is not None:
            break

    if segmask_contour is None:
        return None

    segmask_contour = cv2.convexHull(segmask_contour, clockwise=True)
    segmask_contour = np.asarray(segmask_contour)
    segmask_contour_ = segmask_contour.squeeze(axis=1)
    bin_center_pixel = np.mean(segmask_contour_, axis=0)

    segmask_contour_ = (segmask_contour_ - bin_center_pixel )*0.7 + bin_center_pixel
    segmask_contour_ = segmask_contour_.astype(np.int32)

    segmask = np.zeros_like(rgb_im[:,:,0])
    cv2.fillConvexPoly(segmask,segmask_contour_,255.)
    
    if show_image:
        rgb_bin_center = cv2.circle(rgb_, (int(bin_center_pixel[0]), int(bin_center_pixel[1])), 5, (255, 0, 0), 5)
        plt.imshow(rgb_bin_center)
        plt.show()
    
    return segmask, segmask_contour_

def inverse_raw_pixel(pixel, midx, midy, cs=430, ih=256, iw=256):
    pixel = pixel * float(cs)/float(ih)
    px = int(pixel[0] + midx - cs/2.)
    py = int(pixel[1] + midy - cs/2.)
    return np.array([px, py])

def map_pixel(pixel, midx, midy, cs=430, ih=256, iw=256):
    pixel[0] = pixel[0] - midx + cs/2.
    pixel[1] = pixel[1] - midy + cs/2.
    pixel = pixel * float(ih)/float(cs)
    px, py = int(round(pixel[0])), int(round(pixel[1]))
    return np.array([px, py])

def split_and_merge(vp_network, depth_im, mask):
    crop_image_list = []
    for j in range(11):
        for k in range(11):
            crop_image_list.append(depth_im[16 * j:96 + 16 * j, 16 * k:96 + 16 * k] - 0.1)
    crop_images = np.asarray(crop_image_list)
    crop_images = crop_images[..., np.newaxis]

    prediction = vp_network.predict(crop_images)
    
    weight = prediction[0]
    max_weight = np.argmax(weight, axis=-1)
    predictions = np.zeros((crop_images.shape[0], 5))
    for i in range(crop_images.shape[0]):
        predictions[i] = prediction[1][i, max_weight[i], :]
    predictions[:, 0] = (predictions[:, 0] + 1.0) * np.pi / 10.0 
    predictions[:, 1] = (predictions[:, 1] + 1.0) * np.pi
    predictions[:, 2] = predictions[:, 2] * np.pi / 2.0
    predictions[:, 3] = (predictions[:, 3] + 1.0) * 48
    predictions[:, 4] = (predictions[:, 4] + 1.0) * 48
    
    concat_predictions = []
    for j in range(11):
        for k in range(11):
            pose_x = np.max(int(predictions[11 * j + k, 3] + 16 * k), 0)
            pose_y = np.max(int(predictions[11 * j + k, 4] + 16 * (10 - j)), 0)
            if mask[np.clip(255 - pose_y,0,255), np.clip(pose_x,0,255)]:
                concat_predictions.append([predictions[11 * j + k, 0], predictions[11 * j + k, 1], 
                                           predictions[11 * j + k, 2], pose_x, pose_y])
                                       
    concat_predictions = np.asarray(concat_predictions)
    boxes = []
    for i in range(concat_predictions.shape[0]):
        pose_x = int(concat_predictions[i, 3])
        pose_y = int(concat_predictions[i, 4])
        boxes.append([pose_x - 25, pose_y - 25, pose_x + 25, pose_y + 25])
    boxes = np.asarray(boxes, dtype=float)

    def non_max_suppression_slow(boxes, overlapThresh):
        if len(boxes) == 0:
            return []

        pick = []
        prob = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            delete_idx = np.where(overlap > overlapThresh)[0]

            if len(delete_idx) != 0:
                xxx1 = np.minimum(x1[i], np.min(x1[idxs[delete_idx]]))
                yyy1 = np.minimum(y1[i], np.min(y1[idxs[delete_idx]]))
                xxx2 = np.maximum(x2[i], np.max(x2[idxs[delete_idx]]))
                yyy2 = np.maximum(y2[i], np.max(y2[idxs[delete_idx]]))

                boxes[i] = [xxx1, yyy1, xxx2, yyy2]
            prob.append(1 + len(delete_idx))

            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int"), prob
                                       
    concat_boxes, prob = non_max_suppression_slow(boxes, 0.9)
    
    num_theta_bin = 12
    num_phi_bin = 4
    votes = np.zeros((num_phi_bin, num_theta_bin))
    for i in range(concat_predictions.shape[0]):
        x = int(min(concat_predictions[i, 0], np.pi * 0.098) // (np.pi / (10 * num_phi_bin)))
        y = int(concat_predictions[i, 1] // (2 * np.pi / num_theta_bin))
        votes[x, y] = votes[x, y] + 1
    return concat_boxes, prob, votes              


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

    def __init__(self, cam_id="025222072234"):
        self.cam_id = cam_id
        self.realsense = None
        self.set_realsense()

        self.planUR5 = None
        self.moveUR5 = None
        self.getEEFPose = None
        self.getJointStates = None
        self.get_ur5_control_service()

        self.T_eef_to_rs = np.load('rs_extrinsic.npy')

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

    def get_view(self, goal_pos=None, quat=[1, 0, 0, 0], grasp=0.0, show_img=False):
        # quat: xyzw
        if goal_pos is not None:
            plans = self.moveUR5(self.ARM_JOINT_NAME, None, goal_pos, quat, 1-grasp)
            if len(plans.plan.points)<=1:
                print("Failed planning to the goal.")
                return None, None
        rospy.sleep(0.5)
        color, depth = self.realsense.frames(spatial=True, hole_filling=True, temporal=True)
        if show_img:
            plt.imshow(color)
            plt.show()
        return color, depth

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
        T_realsense = []
        T_ur5 = []

        for p, q in zip(self.calib_positions, self.calib_quaternions):
            img, _ = self.get_view(p, q)
            R, T_cam_to_chess = self.get_cam_R_T(img)
            T_chess_to_cam = np.linalg.inv(T_cam_to_chess)
            
            q_wxyz = [q[3], q[0], q[1], q[2]]
            T_base_to_eef = form_T(quaternion_matrix(q_wxyz)[:3, :3], p)
            
            T_realsense.append(T_chess_to_cam)
            T_ur5.append(T_base_to_eef)
            rospy.sleep(1.0)

        img, _ = self.get_view(ROBOT_INIT_POS)
        return T_realsense, T_ur5
    
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
        for i in range(len(A[1,1,:])):
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

        # print(euler_iekf)
        print('Euler:', np.array(euler_ukf)*180/np.pi)
        print('Translation:', ukf.x[3:])
        print('------------------------------')
        T_eef_to_rs = form_T(Tools.vec2rotmat(theta, k), ukf.x[3:])
        print(T_eef_to_rs)

        print('------------------------------')
        print('Quat:', Rotation.from_matrix(T_eef_to_rs[:3,:3]).as_quat())
        print('Euler(degree):', Rotation.from_matrix(T_eef_to_rs[:3,:3]).as_euler('zyx', degrees=True))
        print('------------------------------')
        print('Error:', get_error(T_eef_to_rs))
