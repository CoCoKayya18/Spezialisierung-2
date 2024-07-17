#!/usr/bin/env python

import rospy
from ekf_slam_pkg.ekf import EKFSLAM
from ekf_slam_pkg.robot import Robot
from ekf_slam_pkg.sensor import Sensor
from ekf_slam_pkg.map import Map

def main():
    rospy.init_node('ekf_slam')
    
    # Load parameters
    initial_state = rospy.get_param("robot/initial_state")
    sensor_noise = rospy.get_param("sensor/noise")
    process_noise = rospy.get_param("ekf/process_noise")
    measurement_noise = rospy.get_param("ekf/measurement_noise")
    
    config = {
        'initial_state': initial_state,
        'sensor_noise': sensor_noise,
        'process_noise': process_noise,
        'measurement_noise': measurement_noise
    }
    
    # Initialize components
    robot = Robot(config)
    sensor = Sensor(config)
    map = Map(config)
    ekf_slam = EKFSLAM(robot, sensor, map, config)
    
    rate = rospy.Rate(10)
    
    while not rospy.is_shutdown():
        ekf_slam.run()
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
