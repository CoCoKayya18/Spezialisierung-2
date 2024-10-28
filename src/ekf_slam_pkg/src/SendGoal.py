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

        # Define your custom snake path here around the columns
        waypoints = [
            # P1
            (0.3071, 1.3245, 1.3802), 
            # P2
            (-2.3131, -0.0423, -1.2761), 
            # P3
            (-4.2755, 1.5453, 1.1396), 
            # P4
            (-2.3377, 4.1598, 0.3736), 
            # P5
            (0.0355, 3.0975, -0.8200), 
            # P6
            (2.2565, 0.7089, -0.1164), 
            # P7
            (3.7533, 2.3387, 1.2745), 
            # P10
            (4.4890, -3.4088, 1.0032), 
            # P11
            (2.2995, -1.9500, 0.0989), 
            # P12
            (0.9298, -4.4423, -1.1085), 
            # P13
            (3.1496, -5.2670, -0.4960), 
            # P16
            (1.3999, -6.9431, 0.1347), 
            # P17
            (-0.2188, -4.7697, 1.1162), 
            # P18
            (-2.4132, -3.4897, 3.0992), 
            # P19
            (-2.6674, -2.5219, 0.9041), 
            # P20
            (-1.0336, -1.4038, -0.1317), 
            # P21
            (0.5955, -0.8447, 0.4238), 
            # P22
            (2.6163, 0.0997, 0.8365)
        ]

        follow_path(client, waypoints)

        stop_robot()
        stop_ros_master()

    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation sequence interrupted.")

if __name__ == '__main__':
    main()
