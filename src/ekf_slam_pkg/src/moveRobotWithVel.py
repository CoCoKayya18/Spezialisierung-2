#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import time

def move_robot():
    # Initialize the ROS node
    rospy.init_node('robot_driver', anonymous=True)
    
    # Create a publisher for the /cmd_vel topic
    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=100)
    
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
    rospy.sleep(5)

    # 1. Move forward toward the first line (for 8 seconds)
    drive_forward(duration=8, speed=0.4)
    
    # 2. Rotate to avoid first detected circle
    rotate_robot(duration=1.5, angular_speed=0.6)
    
    # 3. Move forward to bypass the first detected line and circle
    drive_forward(duration=3, speed=0.4)
    
    # 4. Rotate again to follow along the path without intersecting lines
    rotate_robot(duration=1.2, angular_speed=-0.6)
    
    # 5. Move forward to avoid second detected circle and line intersection
    drive_forward(duration=5, speed=0.5)
    
    # 6. Rotate to align towards the top right part of the map
    rotate_robot(duration=1.5, angular_speed=-0.8)
    
    # 7. Move forward to reach the top right corner
    drive_forward(duration=6, speed=0.5)
    
    # 8. Rotate to the right
    rotate_robot(duration=5, angular_speed=0.6)

    # Stop the robot before exiting
    stop_robot()
    
    rospy.loginfo("Movement sequence complete. Keeping the node alive.")
    rospy.spin()

if __name__ == '__main__':
    try:
        move_robot()
    except rospy.ROSInterruptException:
        pass
