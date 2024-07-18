#!/usr/bin/env python

import rospy
import os
import sys

print("PYTHONPATH:", sys.path)

from ekf_slam_pkg.include.ekf_slam_pkg import EKFSLAM

def main():
    rospy.init_node('ekf_slam_launcher', anonymous=True)

    # Load parameters from the YAML file
    param_file = os.path.join(rospy.get_param('/ekf_slam_pkg_path'), 'config', 'ekf_slam_params.yaml')
    with open(param_file, 'r') as f:
        yaml_params = rospy.load_param(f)

    # Set parameters
    for key, value in yaml_params.items():
        rospy.set_param(key, value)

    # Get the initial position parameters
    x_pos = rospy.get_param("robot/initial_position/x")
    y_pos = rospy.get_param("robot/initial_position/y")
    z_pos = rospy.get_param("robot/initial_position/z")
    model = rospy.get_param("TURTLEBOT3_MODEL", "burger")

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
    
    robot = Robot(config, ekf_slam)
    ekf_slam.robot = robot  # Set the robot reference after initialization

    rospy.loginfo("Started EKF SLAM launch")
    rospy.spin()

if __name__ == '__main__':
    print("PYTHONPATH:", sys.path)
    # main()
