#!/usr/bin/env python
import os
import sys
import rospy

sys.path.append(os.path.join('/home/brain3/catkin_ws/src/'))
#from moveit_tutorials.srv import *

from ur_msgs.srv import JointTrajectory
#from trajectory_msgs.msg import JointTrajectory

import moveit_commander
from geometry_msgs.msg import Pose, Quaternion
from moveit_commander import RobotCommander, MoveGroupCommander, roscpp_initialize

#group_name = "both_arms" #"left_arm", "both_arms"
#group = moveit_commander.MoveGroupCommander(group_name)

eef_pose = None

def get_eef_pose(msg):
    global eef_pose
    eef_pose = None

def send_joint_trajectory(req):
    if len(req.arm_qpos) != 0:
        print(req.arm_qpos)
        arm_pose = dict(zip(req.arm_joint_name, req.arm_qpos))
        robot_arm.set_joint_value_target(arm_pose)
    elif len(req.arm_qpos) == 0:
        print(req.eef_pose, req.eef_quat)
        pose_goal = Pose()
        pose_goal.orientation.x = req.eef_quat[0]
        pose_goal.orientation.y = req.eef_quat[1]
        pose_goal.orientation.z = req.eef_quat[2]
        pose_goal.orientation.w = req.eef_quat[3]
        pose_goal.position.x = req.eef_pose[0]
        pose_goal.position.y = req.eef_pose[1]
        pose_goal.position.z = req.eef_pose[2]

        robot_arm.set_pose_target(pose_goal)

    plan_results = robot_arm.plan()
    robot_arm.go(wait=True)
    return plan_results.joint_trajectory

def move_code():
          
    """
    robot_arm.set_named_target("home")  #go to goal state.
    robot_arm.go(wait=True)
    print("====== move plan go to home 1 ======")        
    rospy.sleep(1)  
    """      

    """robot_arm.set_named_target("up")  #go to goal state.
    robot_arm.go(wait=True)
    print("====== move plan go to up ======")        
    rospy.sleep(1)"""

    #robot_state = robot_arm.get_current_pose()
    #robot_angle = robot_arm.get_current_joint_values()

    #TARGET_JOINT_QPOS = [-1.9040511290179651, -1.9013617674456995, 1.276972770690918, -0.8737161795245569, 4.636346817016602, -3.311625067387716]
    TARGET_JOINT_QPOS = [-1.9631989637957972, -2.113516632710592, 1.476273536682129, -0.894970719014303, -1.5849650541888636, -2.9540748417318197]
    robot_arm.set_joint_value_target(TARGET_JOINT_QPOS)
    robot_arm.go(wait=True)
    rospy.sleep(1)  

    pose_goal = Pose()
    pose_goal.orientation.x = 0.96 #0.0
    pose_goal.orientation.y = -0.28 #0.7071068
    pose_goal.orientation.z = 0.0 #0.0
    pose_goal.orientation.w = -0.02 #0.7071068
    pose_goal.position.x = 0.028 #0.4
    pose_goal.position.y = -0.21 #0.0
    pose_goal.position.z = 0.6 #0.4
    robot_arm.set_pose_target(pose_goal)
    robot_arm.go(wait=True)
    print("====== move plan go to random pose ======")        
    rospy.sleep(1)


if __name__ == '__main__':
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('ur5_move', anonymous=True)
    rospy.Subscriber('/ur5_eef_pose', Pose, get_eef_pose)

    GROUP_NAME_ARM = "manipulator"
    robot_cmd = RobotCommander()
    robot_arm = MoveGroupCommander(GROUP_NAME_ARM)

    move_code()

    rospy.Service('send_joint_trajectory', JointTrajectory, send_joint_trajectory)
    rospy.spin()




    




