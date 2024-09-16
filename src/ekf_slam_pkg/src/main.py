#!/usr/bin/env python

# import rospy
import os
import sys
import rospy
from ekf_slam_pkg import EKFSLAM
from ekf_slam_pkg import Sensor
from ekf_slam_pkg import Map
from ekf_slam_pkg import Robot
from ekf_slam_pkg import Utils


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

    rospy.loginfo("Started EKF SLAM launch")
    rospy.spin()

if __name__ == '__main__':
    main()
