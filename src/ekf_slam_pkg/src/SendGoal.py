#!/usr/bin/env python
import rospy
import actionlib
from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import quaternion_from_euler
import os
import math

GOAL_FILE_PATH = '/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/data/Goals.txt'
ROTATION_TIMEOUT = 30  # Timeout for rotation in seconds
MAX_RETRIES = 5  # Maximum number of retries for a goal

def move_to_goal(client, x_goal, y_goal, yaw_goal):
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = 'map'  # Replace with your frame ID
    goal.target_pose.header.stamp = rospy.Time.now()

    goal.target_pose.pose.position.x = x_goal
    goal.target_pose.pose.position.y = y_goal

    q_angle = quaternion_from_euler(0, 0, yaw_goal)
    goal.target_pose.pose.orientation.z = q_angle[2]
    goal.target_pose.pose.orientation.w = q_angle[3]

    client.send_goal(goal)
    wait = client.wait_for_result(rospy.Duration(ROTATION_TIMEOUT))

    if not wait:
        rospy.logerr("Action server not available!")
        return False
    else:
        state = client.get_state()
        if state == actionlib.GoalStatus.SUCCEEDED:
            return True
        else:
            rospy.loginfo("Failed to reach the goal within the timeout.")
            return False

def initialize_action_client():
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    rospy.loginfo("Waiting for move_base action server...")
    client.wait_for_server()
    rospy.loginfo("Connected to move_base action server.")
    return client

def stop_robot():
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    rospy.sleep(1)
    stop_msg = Twist()
    pub.publish(stop_msg)
    rospy.loginfo("Robot stopped.")

def stop_ros_master():
    rospy.loginfo("Stopping ROS master and all nodes.")
    os.system("rosnode kill -a")
    os.system("killall -9 rosmaster")
    os.system("killall -9 roscore")

def save_goal_to_file(x_goal, y_goal, yaw_goal):
    with open(GOAL_FILE_PATH, 'a') as file:
        file.write(f"{x_goal},{y_goal},{yaw_goal}\n")

def follow_path(client, waypoints):
    for idx, waypoint in enumerate(waypoints):
        waypoint_label = f"P{idx + 2}"
        for retry in range(MAX_RETRIES):
            x_goal, y_goal, yaw_goal = waypoint
            save_goal_to_file(x_goal, y_goal, yaw_goal)
            rospy.loginfo("Moving to waypoint %s: x = %s, y = %s meters", waypoint_label, x_goal, y_goal)
            result = move_to_goal(client, x_goal, y_goal, yaw_goal)
            if result:
                rospy.loginfo("Reached waypoint %s successfully.", waypoint_label)
                break
            else:
                rospy.loginfo("Failed to reach waypoint %s. Retrying...", waypoint_label)
                rospy.sleep(1)
        else:
            rospy.loginfo("Failed to reach waypoint %s after maximum retries. Continuing to next waypoint.", waypoint_label)
            continue

def main():
    try:
        rospy.init_node('sendGoal_py')
        client = initialize_action_client()

        # First path
        # waypoints = [
        #     # P1
        #     (-0.9834, 1.9400, 1.0517),
        #     # P2
        #     (-3.7368, 2.8272, 1.1072),
        #     # P3
        #     (-1.4227, 5.0595, 0.5162),
        #     # P4
        #     (1.1790, 3.9684, -0.7855),
        #     # P5
        #     (3.6892, 2.1652, -0.8643),
        #     # P6
        #     (4.0100, -0.4734, -0.2387),
        #     # P7
        #     (5.5542, -1.8953, -1.1788),
        #     # P8
        #     (2.8486, -3.9326, 0.0334),
        #     # P9
        #     (1.9288, -6.0383, 0.2575),
        #     # P10
        #     (-0.5112, -5.1600, 0.2213),
        #     # P11
        #     (-2.4035, -5.2799, -0.6313),
        #     # P12
        #     (-4.7121, -5.7601, 1.0867),
        #     # P13
        #     (-3.3396, -1.8661, 0.9080),
        #     # P14
        #     (-4.2382, -0.8942, 0.2248),
        #     # P15
        #     (-0.6600, -0.2257, -0.3587)
        # ]
        
        # Second path
        waypoints = [
            # P1
            (0.8132, 0.0196, 0.0004),
            # P2
            (2.2985, -1.1061, -0.5716),
            # P3
            (1.5988, -2.4078, -0.9951),
            # P4
            (-1.1393, -2.0232, 0.7214),
            # P5
            (-0.0636, 0.6228, 0.4397),
            # P6
            (2.1695, 2.5829, -0.0494),
            # P7
            (3.7611, 1.1996, -0.9020),
            # P8
            (3.0506, -0.9589, -0.9450),
            # P9
            (3.0726, -3.5360, -0.6925),
        ]

        follow_path(client, waypoints)

        stop_robot()
        stop_ros_master()

    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation sequence interrupted.")

if __name__ == '__main__':
    main()
