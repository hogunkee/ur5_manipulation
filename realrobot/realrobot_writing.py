import os
ROS_DISTRO = os.environ['ROS_DISTRO']

from __future__ import print_function
from six.moves import input

# path
import sys
ros_python_path = '/opt/ros/{}/lib/python2.7/dist-packages'.format(ROS_DISTRO)
if not ros_python_path in sys.path:
    sys.path.append(ros_python_path)

import rospy
import copy
import numpy as np

# moveit
from moveit_commander.conversions import pose_to_list
from std_msgs.msg import String
import geometry_msgs.msg
import moveit_commander
import moveit_msgs.msg
from math import pi

# gripper
from robotiq_2f_gripper_msgs.msg import CommandRobotiqGripperFeedback, CommandRobotiqGripperResult, CommandRobotiqGripperAction, CommandRobotiqGripperGoal
from robotiq_2f_gripper_control.robotiq_2f_gripper_driver import Robotiq2FingerGripperDriver as Robotiq
import actionlib


class UR5Robot(object):
    def __init__():
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_group_python_interface_tutorial', anonymous=True)

        self.robot = moveit_commander.RobotCommander()
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                       moveit_msgs.msg.DisplayTrajectory,
                                                       queue_size=20)
        group_name = "manipulator"
        self.move_group = moveit_commander.MoveGroupCommander(group_name)

        # robotiq gripper
        action_name = rospy.get_param('~action_name', 'command_robotiq_action')
        print(action_name)
        self.robotiq_client = actionlib.SimpleActionClient(action_name, CommandRobotiqGripperAction)
        self.robotiq_client.wait_for_server()
        print("Client test: Starting sending goals")


    def move_to_joints(self, joint_goal):
        self.move_group.go(joint_goal, wait=True)
        self.move_group.stop()
        
        current_joints = self.move_group.get_current_joint_values()
        return current_joints

        
    def get_eef_pose(self):
        return self.move_group.get_current_pose().pose


    def move_to_init_pose(self):
        init_joints = [-1.9631989637957972, -2.113516632710592, 1.476273536682129, -0.894970719014303, -1.5849650541888636, -2.9540748417318197]
        self.move_to_joints(init_joints)


    def custom_path(self):
        delta_x = 0.01
        delta_y = 0.015
        delta_z = 0.015

        waypoints = []
        wpose = self.move_group.get_current_pose().pose

        wpose.position.x += delta_x
        waypoints.append(copy.deepcopy(wpose))
        wpose.position.y += delta_y
        waypoints.append(copy.deepcopy(wpose))
        wpose.position.z += delta_z
        waypoints.append(copy.deepcopy(wpose))

        wpose.position.x -= delta_x
        wpose.position.y -= delta_y
        wpose.position.z -= delta_z
        waypoints.append(copy.deepcopy(wpose))

        (plan, fraction) = self.move_group.compute_cartesian_path(
                                           waypoints,   # waypoints to follow
                                           0.01,        # eef_step
                                           0.0)         # jump_threshold

        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = robot.get_current_state()
        display_trajectory.trajectory.append(plan)

        self.display_trajectory_publisher.publish(display_trajectory);
        self.move_group.execute(plan, wait=True)


    def gripper_open(self):
        goal = CommandRobotiqGripperGoal()
        goal.emergency_release = False
        goal.stop = False
        goal.position = 0.8
        goal.speed = 0.1
        goal.force = 1.0
        self.robotiq_client.send_goal(goal)
        self.robotiq_client.wait_for_result()


    def gripper_close(self):
        goal = CommandRobotiqGripperGoal()
        goal.emergency_release = False
        goal.stop = False
        goal.position = 0.0
        goal.speed = 0.1
        goal.force = 5.0
        self.robotiq_client.send_goal(goal)
        self.robotiq_client.wait_for_result()

