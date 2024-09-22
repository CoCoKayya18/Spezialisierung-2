import rospy
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os

class Sensor:
    def __init__(self, config):
        rospy.loginfo("Sensor class initialized")
        self.plot_counter = 1
        # self.first_call = True
        pass

    def extract_features_from_scan(self, scan_data, angle_min, angle_max, angle_increment, eps=0.2, min_samples=5):
            # Extracts features from LiDAR scan data using DBSCAN clustering

            # Extract ranges from the LaserScan message
            ranges = np.array(scan_data.ranges)

            # Convert polar coordinates (range and angle) to Cartesian coordinates (x, y)
            angles = angles = angle_min + np.arange(len(ranges)) * angle_increment

            # # Ensure angles array has the same length as ranges
            # if len(angles) > len(ranges):
            #     angles = angles[:len(ranges)]

            x_coords = ranges * np.cos(angles)
            y_coords = ranges * np.sin(angles)

            # Stack x and y coordinates into a single array
            points = np.vstack((x_coords, y_coords)).T

            # Remove points with invalid ranges (e.g., 0 or inf values)
            valid_points = points[np.isfinite(points).all(axis=1)]

            # Apply DBSCAN clustering algorithm
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(valid_points)
            labels = db.labels_

            # Extract features from each cluster
            features = []
            unique_labels = set(labels)
            
            for label in unique_labels:
                if label == -1:
                    continue  # Skip noise points

                # Extract points belonging to the current cluster
                cluster_points = valid_points[labels == label]
                
                # Calculate the mean of the cluster points as the feature location
                cluster_center = cluster_points.mean(axis=0)
                features.append(tuple(cluster_center))
            
            # self.visualize_features(valid_points, labels, features)

            print(f"\n Following features detected: {features}")
            
            return features

    def visualize_features(self, valid_points, labels, features):
        # Extract features using DBSCAN
        # features, labels, valid_points = self.extract_features_from_scan(scan_data, angle_min, angle_max, angle_increment, eps, min_samples)

        save_dir = '../Spezialisierung-2/src/ekf_slam_pkg/plots/DBSCAN_Plots'

        # Check if this is the first call
        # if self.first_call:
        #     # Remove all existing plots in the directory
        #     if os.path.exists(save_dir):
        #         for file_name in os.listdir(save_dir):
        #             file_path = os.path.join(save_dir, file_name)
        #             try:
        #                 if os.path.isfile(file_path):
        #                     os.remove(file_path)
        #                     rospy.loginfo(f"Deleted existing plot: {file_path}")
        #             except Exception as e:
        #                 rospy.logerr(f"Error deleting file {file_path}: {e}")
        #     else:
        #         os.makedirs(save_dir)  # Create the directory if it doesn't exist
        #     self.first_call = False  # Set the flag to False after the first call

        # Plot the LiDAR scan points
        plt.figure(figsize=(10, 10))
        plt.scatter(valid_points[:, 0], valid_points[:, 1], c=labels, cmap='tab20', s=10, label='Scan Points')

        # Plot the extracted features (cluster centers)
        features = np.array(features)
        plt.scatter(features[:, 0], features[:, 1], c='red', s=100, marker='x', label='Extracted Features (Cluster Centers)')

        # Set plot details
        plt.title('LiDAR Scan Data and Extracted Features')
        plt.xlabel('X [meters]')
        plt.ylabel('Y [meters]')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')

        # Save plot with sequential numbering
        plot_filename = os.path.join(save_dir, f'plot_{self.plot_counter}.png')
        plt.savefig(plot_filename)
        plt.close()  # Close the plot to free memory
        rospy.loginfo(f"Saved plot as {plot_filename}")

        # Increment the plot counter
        self.plot_counter += 1
