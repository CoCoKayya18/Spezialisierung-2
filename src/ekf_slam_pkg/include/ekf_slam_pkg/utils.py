import numpy as np
import rospy
import csv
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_from_euler
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import json
import tf 
import threading

class Utils:
    
    def __init__(self):
        rospy.loginfo("Utils class initialized")

        # For the csv files
        self.ground_truth_csv_path = '/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/data/ground_truth_path.csv'
        self.ekf_path_csv_path = '/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/data/ekf_path.csv'
        self.odom_velocities_csv_path = '/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/data/odom_velocities.csv'
        self.correctionJsonPath = '/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/data/correctionData.json'

        self.listener = tf.TransformListener()

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
    
    def transform_scan_to_map(self, scan_msg):
        
        transformed_scan = LaserScan()
        transformed_scan.header.frame_id = "base_scan"  
        transformed_scan.header.stamp = scan_msg.header.stamp
        transformed_scan.angle_min = scan_msg.angle_min
        transformed_scan.angle_max = scan_msg.angle_max
        transformed_scan.angle_increment = scan_msg.angle_increment
        transformed_scan.time_increment = scan_msg.time_increment
        transformed_scan.scan_time = scan_msg.scan_time
        transformed_scan.range_min = scan_msg.range_min
        transformed_scan.range_max = scan_msg.range_max
        transformed_scan.ranges = []

        try:
            # Get the transformation from base_scan to map
            self.listener.waitForTransform("map", scan_msg.header.frame_id, rospy.Time(0), rospy.Duration(4.0))

            # Transform each laser scan point to the map frame
            angle = scan_msg.angle_min
            for r in scan_msg.ranges:
                if r < scan_msg.range_min or r > scan_msg.range_max:
                    transformed_scan.ranges.append(float('inf'))
                    angle += scan_msg.angle_increment
                    continue

                # Convert polar coordinates (range, angle) to Cartesian (x, y) in base_scan frame
                point = PointStamped()
                point.header.frame_id = scan_msg.header.frame_id
                point.point.x = r * np.cos(angle)
                point.point.y = r * np.sin(angle)
                point.point.z = 0

                # Transform the point to the map frame
                point_in_map = self.listener.transformPoint("map", point)

                # Convert back to polar coordinates (range) in the map frame
                transformed_range = np.sqrt(point_in_map.point.x ** 2 + point_in_map.point.y ** 2)
                transformed_scan.ranges.append(transformed_range)

                # Increment the angle for the next scan point
                angle += scan_msg.angle_increment

            return transformed_scan

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("Transform lookup failed. Returning untransformed scan data.")
            return scan_msg
    
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
        filename = f"/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/plots/Covariance_Plots/cov_matrix_obs_{observation_loop}_lm_{landmark_loop}.png"
        plt.savefig(filename)
        plt.close()

    def save_jacobian_plot(self, jacobian, observation_loop, landmark_loop):
        # Create plot
        plt.matshow(jacobian, cmap='coolwarm')
        plt.colorbar()
        plt.title(f'H Jacobian (Obs: {observation_loop}, LM: {landmark_loop})')

        # Save the plot with observation and landmark loop in filename
        filename = f"/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/plots/H_Jacobian_Plots/h_jacobian_obs_{observation_loop}_lm_{landmark_loop}.png"
        plt.savefig(filename)
        plt.close()
    
    def save_psi_plot(self, psi, observation_loop, landmark_loop):
        # Create plot
        plt.matshow(psi, cmap='coolwarm')
        plt.colorbar()
        plt.title(f'Psi (Obs: {observation_loop}, LM: {landmark_loop})')

        # Save the plot with observation and landmark loop in filename
        filename = f"/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/plots/Psi_Plots/psi_obs_{observation_loop}_lm_{landmark_loop}.png"
        plt.savefig(filename)
        plt.close()
    
    def save_kalman_plot(self, kalman_gain, observation_loop, landmark_loop):
        # Create plot
        plt.matshow(kalman_gain, cmap='coolwarm')
        plt.colorbar()
        plt.title(f'Kalman gain (Obs: {observation_loop}, LM: {landmark_loop})')

        # Save the plot with observation and landmark loop in filename
        filename = f"/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/plots/Kalman_Plots/kalman_gain_obs_{observation_loop}_lm_{landmark_loop}.png"
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
        filename = f"/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/plots/EstimatedObservations_Plots/Correction_{correctionRun}.png"
        plt.savefig(filename)
        plt.close()
    
    def save_correction_data_to_json(self, correction_data):
        
        # Check if file exists, if not, create it and initialize it with an empty list
        if not os.path.exists(self.correctionJsonPath) or os.stat(self.correctionJsonPath).st_size == 0:
            with open(self.correctionJsonPath, 'w') as json_file:
                json.dump([], json_file)  # Initialize with an empty list

        # Read the current contents of the file
        with open(self.correctionJsonPath, 'r') as json_file:
            existing_data = json.load(json_file)

        # Append the new correction data
        existing_data.append(correction_data)

        # Write the updated data back to the file
        with open(self.correctionJsonPath, 'w') as json_file:
            json.dump(existing_data, json_file, indent=4)
            
    def visualize_and_save_laserscan(self, scan_msg, counter):

        save_directory = "/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/plots/laserScan_Plots"
        
        num_points = len(scan_msg.ranges)
        
        # Convert polar coordinates (range, angle) to Cartesian (x, y)
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, num_points)
        ranges = np.array(scan_msg.ranges)
        
        # Filter out invalid ranges (inf or NaN)
        valid_indices = np.isfinite(ranges)
        ranges = ranges[valid_indices]
        angles = angles[valid_indices]
        
        # Convert to Cartesian coordinates
        x_coords = ranges * np.cos(angles)
        y_coords = ranges * np.sin(angles)
        
        rotated_x_coords = -y_coords
        rotated_y_coords = x_coords
        
        # Plotting
        plt.figure()
        plt.scatter(rotated_x_coords, rotated_y_coords, c='b', s=5, label='LaserScan Points')
        plt.title('LaserScan Data (Rotated)')
        plt.xlabel('X [m] (Up)')
        plt.ylabel('Y [m] (Left)')
        plt.axis('equal')
        plt.legend()

        # Create the save directory if it doesn't exist
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the plot
        save_path = os.path.join(save_directory, f'laserscan_plot_{counter}.png')
        plt.savefig(save_path)
        plt.close()
    
    # def plot_async(self, target_func, *args, **kwargs):
    #     # Use threading to call the target plotting function asynchronously
    #     threading.Thread(target=target_func, args=args, kwargs=kwargs).start()

