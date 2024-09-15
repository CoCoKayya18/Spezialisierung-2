import numpy as np
import rospy
import csv
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_from_euler
import pickle
import os

# Add more utility functions as needed

# def scaleInput(odometryData):
#     with open(os.path.join(scalerFilePath, scaler_filenameX), 'rb') as file:
#         scaler_X = pickle.load(file)

#     standardizedInput = scaler_X.transform(odometryData)

#     return standardizedInput

# def rescaleOutput(output):
#     with open(os.path.join(scalerFilePath, scaler_filenameY), 'rb') as file:
#         scaler_Y = pickle.load(file)
    
#     rescaledOuput = scaler_Y.inverse_transform(output)

#     return rescaledOuput

class Utils:
    
    def __init__(self):
        rospy.loginfo("Utils class initialized")

        # For the csv files
        self.ground_truth_csv_path = '../Spezialisierung-2/src/ekf_slam_pkg/data/ground_truth_path.csv'
        self.ekf_path_csv_path = '../Spezialisierung-2/src/ekf_slam_pkg/data/ekf_path.csv'
        self.odom_velocities_csv_path = '../Spezialisierung-2/src/ekf_slam_pkg/data/odom_velocities.csv'

    def update_pose_from_state(self, pose, x, y, theta):
        # Helper function to update the robot's pose from state vector
        pose.position.x = x
        pose.position.y = y
        quaternion = quaternion_from_euler(0, 0, theta)
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]

    def wrap_angle(angle):
        # Wrap angle between -pi and pi
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def transform_odometry_to_world(self, odometry_msg: Odometry) -> np.ndarray:
        # Extract velocities from odometry
        linear_x = odometry_msg.twist.twist.linear.x
        linear_y = odometry_msg.twist.twist.linear.y
        angular_z = odometry_msg.twist.twist.angular.z

        # Extract quaternion from odometry orientation
        orientation_q = odometry_msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        
        # Convert quaternion to Euler angles to get yaw
        _, _, yaw = euler_from_quaternion(orientation_list)
        
        # Apply rotation matrix to transform velocities to the world frame
        v_x_world = linear_x * np.cos(yaw) - linear_y * np.sin(yaw)
        v_y_world = linear_x * np.sin(yaw) + linear_y * np.cos(yaw)

        # Create the transformed velocity vector (including angular velocity)
        transformed_velocities = np.array([[v_x_world, v_y_world, angular_z]])

        return transformed_velocities
    
    def initialize_csv_files(self, ground_truth_csv_path, ekf_path_csv_path, odom_velocities_csv_path):
        # Initialize ground truth CSV file
        with open(ground_truth_csv_path, 'w', newline='') as csvfile:
            self.ground_truth_csv_path = ground_truth_csv_path
            writer = csv.writer(csvfile)
            writer.writerow(['timestamp', 'x', 'y', 'z'])  # Header for ground truth path

        # Initialize EKF path CSV file
        with open(ekf_path_csv_path, 'w', newline='') as csvfile:
            self.ekf_path_csv_path = ekf_path_csv_path
            writer = csv.writer(csvfile)
            writer.writerow(['timestamp', 'x', 'y', 'z'])  # Header for EKF path

        # Initialize odom velocities CSV file
        with open(odom_velocities_csv_path, 'w', newline='') as csvfile:
            self.odom_velocities_csv_path = odom_velocities_csv_path
            writer = csv.writer(csvfile)
            writer.writerow(['timestamp', 'linear_x', 'linear_y', 'angular_z'])  # Header for odom velocities

    def save_odom_velocities_to_csv(self, msg):
        with open(self.odom_velocities_csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            timestamp = rospy.Time.now().to_sec()
            writer.writerow([timestamp, msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z])
    
    def save_ground_truth_path_to_csv(self, pose):
        with open(self.ground_truth_csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            timestamp = rospy.Time.now().to_sec()
            writer.writerow([timestamp, pose.position.x, pose.position.y, pose.position.z])
        
    def save_ekf_path_to_csv(self, pose):
        with open(self.ekf_path_csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            timestamp = rospy.Time.now().to_sec()
            writer.writerow([timestamp, pose.position.x, pose.position.y, pose.position.z])

        