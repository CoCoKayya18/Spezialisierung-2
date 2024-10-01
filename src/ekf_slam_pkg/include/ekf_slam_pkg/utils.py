import numpy as np
import rospy
import csv
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_from_euler
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

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

        return pose

    def normalize_angle(self, angle):
        # Wrap angle between -pi and pi
        return np.arctan2(np.sin(angle), np.cos(angle))

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
            writer.writerow(['timestamp', 'linear_x', 'linear_y', 'angular_z', 'Orientation_Quat_x', 'Orientation_Quat_y', 'Orientation_Quat_z', 'Orientation_Quat_w'])  # Header for odom velocities

    def save_odom_velocities_to_csv(self, msg):
        with open(self.odom_velocities_csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            timestamp = rospy.Time.now().to_sec()
            writer.writerow([timestamp, msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
    
    def save_ground_truth_path_to_csv(self, pose):
        with open(self.ground_truth_csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            timestamp = rospy.Time.now().to_sec()
            writer.writerow([timestamp, pose.position.x, pose.position.y, pose.position.z])
        
    def save_ekf_path_to_csv(self, pose):
        with open(self.ekf_path_csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            timestamp = rospy.Time.now().to_sec()
            writer.writerow([timestamp, pose[0].item() if isinstance(pose[0], np.ndarray) else pose[1], pose[1].item() if isinstance(pose[1], np.ndarray) else pose[1], pose[2].item() if isinstance(pose[2], np.ndarray) else pose[2]])

    def clear_directory(self, directory):
        files = glob.glob(f"{directory}/*")
        for f in files:
            os.remove(f)

    def save_covariance_matrix_plot(self, cov_matrix, observation_loop, landmark_loop):
        # Create plot
        plt.imshow(np.log1p(np.abs(cov_matrix)), cmap='coolwarm', interpolation='none')
        plt.colorbar()
        plt.title(f'Covariance Matrix (Obs: {observation_loop}, LM: {landmark_loop})')

        # Save the plot with observation and landmark loop in filename
        filename = f"../Spezialisierung-2/src/ekf_slam_pkg/plots/Covariance_Plots/cov_matrix_obs_{observation_loop}_lm_{landmark_loop}.png"
        plt.savefig(filename)
        plt.close()

    def save_jacobian_plot(self, jacobian, observation_loop, landmark_loop):
        # Create plot
        plt.matshow(jacobian, cmap='coolwarm')
        plt.colorbar()
        plt.title(f'H Jacobian (Obs: {observation_loop}, LM: {landmark_loop})')

        # Save the plot with observation and landmark loop in filename
        filename = f"../Spezialisierung-2/src/ekf_slam_pkg/plots/H_Jacobian_Plots/jacobian_obs_{observation_loop}_lm_{landmark_loop}.png"
        plt.savefig(filename)
        plt.close()
        
    def laser_scan_to_polar_tuples(self, scanMessage):

        # Extract the ranges from the scan message
        ranges = np.array(scanMessage.ranges)

        # Create an array of angles corresponding to each range value
        angles = scanMessage.angle_min + np.arange(len(ranges)) * scanMessage.angle_increment

        # Remove invalid readings (infinite or 0 values)
        valid_indices = np.isfinite(ranges) & (ranges > 0)

        # Filter the ranges and angles using valid indices
        valid_ranges = ranges[valid_indices]
        valid_angles = angles[valid_indices]

        # Create a list of tuples (r, phi)
        polar_coordinates = [(r, phi) for r, phi in zip(valid_ranges, valid_angles)]

        return polar_coordinates

    def visualize_expected_Observation(self, z_hat_list, correctionRun):
        # Convert range and angle to Cartesian coordinates
        x_coords = [z[0] * np.cos(z[1]) for z in z_hat_list]  # x = r * cos(theta)
        y_coords = [z[0] * np.sin(z[1]) for z in z_hat_list]  # y = r * sin(theta)

        # Create a scatter plot in Cartesian coordinates
        plt.figure()
        plt.scatter(x_coords, y_coords, c='b', marker='o')
        plt.title("Laser Scan in Cartesian Coordinates")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis('equal')  # Ensure aspect ratio is equal
        plt.grid(True)
        filename = f"../Spezialisierung-2/src/ekf_slam_pkg/plots/EstimatedObservations_Plots/Correction_{correctionRun}.png"
        plt.savefig(filename)
        plt.close()