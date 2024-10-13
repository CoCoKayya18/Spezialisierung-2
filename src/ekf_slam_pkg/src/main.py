#!/usr/bin/env python

import os
import sys
import rospy
from ekf_slam_pkg import EKFSLAM, Sensor, Map, Robot, Utils
import time  # Import the time module for frequency checking

def main():

    rospy.init_node('ekf_slam_launcher', anonymous=True)

    # Initialize components
    config = {
        'initial_position': rospy.get_param("robot/initial_position"),
        'sensor_noise': rospy.get_param("sensor/noise"),
        'process_noise': rospy.get_param("ekf/process_noise"),
        'measurement_noise': rospy.get_param("ekf/measurement_noise")
    }
    
    sensor = Sensor(config)
    map = Map(config)
    utils = Utils()
    ekf_slam = EKFSLAM(None, sensor, map, config, utils)  # Initialize without robot first
    
    robot = Robot(config, ekf_slam, utils)
    ekf_slam.robot = robot  # Set the robot reference after initialization

    # Initialize timing for frequency monitoring
    loop_rate = 30  # Desired frequency in Hz
    rate = rospy.Rate(loop_rate)
    last_time = time.time()

    rospy.loginfo("Started EKF SLAM launch")

    # Main loop where your SLAM steps occur
    while not rospy.is_shutdown():
        current_time = time.time()
        loop_duration = current_time - last_time
        last_time = current_time

        # Calculate current frequency
        frequency = 1 / loop_duration if loop_duration > 0 else float('inf')
        rospy.loginfo(f"Current loop frequency: {frequency:.2f} Hz")

        # Call your EKF SLAM processing steps here
        # For example:
        # ekf_slam.predict()  # Prediction step
        # ekf_slam.correct()  # Correction step

        # Sleep to maintain loop rate of 30 Hz
        rate.sleep()

if __name__ == '__main__':
    main()
