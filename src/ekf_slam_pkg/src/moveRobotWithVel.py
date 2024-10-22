#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import time

def move_robot():
    # Initialize the ROS node
    rospy.init_node('robot_driver', anonymous=True)
    
    # Create a publisher for the /cmd_vel topic
    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    
    # Define the Twist message for movement
    move_cmd = Twist()

    # Function to drive the robot forward
    def drive_forward(duration, speed):
        move_cmd.linear.x = speed  # Move forward with the specified speed
        move_cmd.angular.z = 0.0   # No rotation
        cmd_vel_pub.publish(move_cmd)
        time.sleep(duration)
        stop_robot()  # Stop after the duration
    
    # Function to rotate the robot
    def rotate_robot(duration, angular_speed):
        move_cmd.linear.x = 0.0   # No forward movement
        move_cmd.angular.z = angular_speed  # Rotate at the specified angular speed
        cmd_vel_pub.publish(move_cmd)
        time.sleep(duration)
        stop_robot()  # Stop after the duration

    # Function to stop the robot
    def stop_robot():
        move_cmd.linear.x = 0.0
        move_cmd.angular.z = 0.0
        cmd_vel_pub.publish(move_cmd)
    
    # Give time for the connection to establish
    rospy.sleep(2)

    # Drive in a square pattern (adjust timings for your robot)
    for _ in range(4):
        drive_forward(duration=10, speed=0.5)   # Drive forward for 2 seconds
        # rotate_robot(duration=1, angular_speed=0.5)  # Rotate 90 degrees (adjust timing for exact angle)

    # Stop the robot before exiting
    stop_robot()

if __name__ == '__main__':
    try:
        move_robot()
    except rospy.ROSInterruptException:
        pass
