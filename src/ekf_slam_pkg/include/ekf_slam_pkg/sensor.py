import rospy
import numpy as np
from math import atan2, sqrt
from sklearn.cluster import DBSCAN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from sklearn.linear_model import RANSACRegressor

class Sensor:
    def __init__(self, config, utils):
        rospy.loginfo("Sensor class initialized")
        self.plot_counter = 1
        self.first_call = True
        self.utils = utils
        
        self.lines_data = []
        self.corners = None
        self.circles = None
        self.points = None
        
        pass

    def extract_features_from_scan(self, scan_data, angle_min, angle_max, angle_increment, eps=0.15, min_samples=5):
            # Extracts features from LiDAR scan data using DBSCAN clustering

            # Extract ranges from the LaserScan message
            ranges = np.array(scan_data.ranges)

            # Convert polar coordinates (range and angle) to Cartesian coordinates (x, y)
            angles = angle_min + np.arange(len(ranges)) * angle_increment

            x_coords = ranges * np.cos(angles)

            y_coords = ranges * np.sin(angles)

            # Stack x and y coordinates into a single array
            points = np.vstack((x_coords, y_coords)).T

            # Remove points with invalid ranges (e.g., 0 or inf values)
            valid_points = points[np.isfinite(points).all(axis=1)]

            if valid_points.size == 0:
                rospy.loginfo("No valid points found in the scan data.")
                return []

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
                
                cluster_center_x, cluster_center_y = cluster_points.mean(axis=0)

                # Convert back to polar coordinates (r, phi)
                r = np.sqrt(cluster_center_x**2 + cluster_center_y**2)
                phi = np.arctan2(cluster_center_y, cluster_center_x)

                # Append the polar coordinates (r, phi) to the features list
                features.append((r, phi))
            
            self.visualize_features(valid_points, labels, features)

            # print(f"\n Following features detected: {features}")
            
            return features

    def visualize_features(self, valid_points, labels, features):
        # Extract features using DBSCAN
        # features, labels, valid_points = self.extract_features_from_scan(scan_data, angle_min, angle_max, angle_increment, eps, min_samples)

        save_dir = '/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/plots/DBSCAN_Plots'

        # Check if this is the first call
        if self.first_call:
            # Remove all existing plots in the directory
            if os.path.exists(save_dir):
                for file_name in os.listdir(save_dir):
                    file_path = os.path.join(save_dir, file_name)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            rospy.loginfo(f"Deleted existing plot: {file_path}")
                    except Exception as e:
                        rospy.logerr(f"Error deleting file {file_path}: {e}")
            else:
                os.makedirs(save_dir)  # Create the directory if it doesn't exist
            self.first_call = False  # Set the flag to False after the first call

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

        if len(corners) == 0:
            return corners
        
        filtered_corners = [corners[0]]
        
        for corner in corners[1:]:
            if all(np.linalg.norm(np.array(corner) - np.array(fc)) > min_distance for fc in filtered_corners):
                filtered_corners.append(corner)
        
        return filtered_corners

    def detect_corners_and_circles_ransac(self, lidar_data, angle_min, angle_max, angle_increment, counter, distance_threshold=0.75, angle_threshold=np.pi / 3):
        
        num_points = len(lidar_data.ranges)
        
        # Convert polar coordinates (range, angle) to Cartesian (x, y)
        angles = angle_min + np.arange(num_points) * angle_increment
        ranges = np.array(lidar_data.ranges)
        
        # Filter out invalid ranges (inf or NaN)
        valid_indices = np.isfinite(ranges)
        ranges = ranges[valid_indices]
        angles = angles[valid_indices]
        
        # Convert to Cartesian coordinates
        x_coords = ranges * np.cos(angles)
        y_coords = ranges * np.sin(angles)

        points = np.vstack((x_coords, y_coords)).T
        
        self.points = points

        # Detect lines using RANSAC
        lines = self.detect_lines_ransac(points, counter)
        
        # Detect line intersections (corners)
        corners = self.detect_line_intersections(lines)
        self.corners = corners

        # Detect circles using RANSAC
        # best_circle, circle_inliers = self.detect_circles_ransac(points)
        # self.circles = best_circle
        
        best_circle = None

        # Convert corner points to polar coordinates
        corner_polar_coords = [self.cartesian_to_polar(x, y) for (x, y) in corners]

        # Convert circle midpoint to polar coordinates (if a circle was detected)
        if best_circle is not None:
            circle_center = (best_circle[0], best_circle[1])
            circle_polar_coords = [list(self.cartesian_to_polar(circle_center[0], circle_center[1]))]
        else:
            circle_polar_coords = []

        # Visualize the results
        # self.utils.plot_async(self.visualize_lidar_data, points, corners, [best_circle], lines, counter, base_name="lidar_extraction")
        # self.visualize_lidar_data(points, corners, [best_circle], lines, counter, base_name="lidar_extraction")
        
        features = corner_polar_coords + circle_polar_coords
        
        # rospy.loginfo(f"Features: {features}")
        
        # self.save_features(lines, [best_circle] if best_circle else [])

        return features

    def detect_lines_ransac(self, points, loopCounter, residual_threshold=0.0275, min_samples=3, max_trials=1000, stop_probability=0.99, min_inliers=8):
        lines = []
        remaining_points = points.copy()
        iteration = 0

        while len(remaining_points) > min_inliers:
            
            iteration += 1
            
            # Fit a line using RANSAC
            x_coords = remaining_points[:, 0].reshape(-1, 1)  # reshape for sklearn
            y_coords = remaining_points[:, 1]

            ransac = RANSACRegressor(min_samples=min_samples, residual_threshold=residual_threshold, max_trials=max_trials, stop_probability=stop_probability)
            ransac.fit(x_coords, y_coords)

            # Get inliers
            inlier_mask = ransac.inlier_mask_
            outlier_mask = np.logical_not(inlier_mask)

            # Check if we have enough inliers
            if np.sum(inlier_mask) < min_inliers:
                break

            # Get slope and intercept of the detected line
            slope = ransac.estimator_.coef_[0]
            intercept = ransac.estimator_.intercept_

            # Store the detected line
            lines.append((slope, intercept))
            
            #  # Visualize current iteration
            # self.utils.plot_async(self.visualize_ransac_iteration, remaining_points, inlier_mask, slope, intercept, iteration, loopCounter)
            # self.visualize_ransac_iteration(remaining_points, inlier_mask, slope, intercept, iteration, loopCounter)
            
            self.lines_data.append({
                "iteration": iteration,
                "loopCounter": loopCounter,
                "slope": slope,
                "intercept": intercept,
                "inliers": remaining_points[inlier_mask].tolist(),  # Save inlier points for visualization
                "outliers": remaining_points[outlier_mask].tolist()  # Save outlier points for visualization
            })

            # Remove inliers from the point set to detect more lines
            remaining_points = remaining_points[outlier_mask]

        return lines
    
    def visualize_ransac_iteration(self, points, inlier_mask, slope, intercept, iteration, loopCounter):

        save_dir = "/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/plots/RansacLinesInIteration"

        # Separate inliers and outliers for plotting
        inliers = points[inlier_mask]
        outliers = points[np.logical_not(inlier_mask)]

        # Create a new figure for each iteration
        plt.figure(figsize=(10, 10))

        # Plot inliers in blue
        plt.scatter(inliers[:, 0], inliers[:, 1], c='blue', label='Inliers')

        # Plot outliers in red
        plt.scatter(outliers[:, 0], outliers[:, 1], c='red', label='Outliers')

        # Plot the detected line
        x_vals = np.array([points[:, 0].min(), points[:, 0].max()])  # x-range for the line
        y_vals = slope * x_vals + intercept  # y = mx + b for the line
        plt.plot(x_vals, y_vals, 'g-', linewidth=2, label=f'Line: y={slope:.2f}x+{intercept:.2f}')

        # Set plot details
        plt.title(f'RANSAC Line Detection - Loop {loopCounter} - Iteration {iteration}')
        plt.xlabel('X [meters]')
        plt.ylabel('Y [meters]')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')

        # Save the plot for this iteration
        filename = f'ransac_loop_{loopCounter}_iteration_{iteration}.png'
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.close()


    def detect_line_intersections(self, lines, parallel_tolerance=1e-1):
        intersections = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                slope1, intercept1 = lines[i]
                slope2, intercept2 = lines[j]

                # Check if lines are parallel
                if abs(slope1 - slope2) < parallel_tolerance:
                    continue
                
                # # Check if lines are parallel
                # if slope1 == slope2:
                #     continue  # Parallel lines do not intersect

                # Calculate the intersection point (x, y)
                x_intersection = (intercept2 - intercept1) / (slope1 - slope2)
                y_intersection = slope1 * x_intersection + intercept1
                intersections.append((x_intersection, y_intersection))

        return intersections

    def detect_circles_ransac(self, points, max_error=0.05, min_samples=3):
        
        best_circle = None
        best_inliers = None
        best_error = float('inf')

        # Try fitting circles with RANSAC
        for _ in range(1000):  # RANSAC iterations
            
            if len(points) < min_samples:
                return None, None
            
            sample_indices = np.random.choice(len(points), min_samples, replace=False)
            sample_points = points[sample_indices]

            xs = sample_points[:, 0]
            ys = sample_points[:, 1]

            # Fit a circle to the sample points
            xc, yc, radius, error = self.fit_circle(xs, ys)

            if error < best_error and error < max_error:
                inliers = np.where(np.abs(np.sqrt((points[:, 0] - xc) ** 2 + (points[:, 1] - yc) ** 2) - radius) < max_error)[0]
                if len(inliers) > min_samples:
                    best_circle = (xc, yc, radius)
                    best_inliers = inliers
                    best_error = error

        return best_circle, best_inliers if best_circle is not None else (None, None)
    
    def fit_circle(self, xs, ys):
        if len(xs) < 3 or len(ys) < 3:
            rospy.logwarn("Not enough points for circle fitting.")
            return None, None, None, np.inf

        # Stack the xs and ys arrays into a 2D array where each row is [x, y]
        X = np.vstack([xs, ys]).T  # Combine xs and ys into 2D array
        
        A = np.vstack([X[:, 0], X[:, 1], np.ones(len(X))]).T
        B = X[:, 0]**2 + X[:, 1]**2
        sol = np.linalg.lstsq(A, B, rcond=None)[0]
        xc = sol[0] / 2
        yc = sol[1] / 2
        radius = np.sqrt((sol[2] + xc**2 + yc**2))

        # Compute the fitting error (distance of points from the circle)
        error = np.mean(np.abs(np.sqrt((X[:, 0] - xc)**2 + (X[:, 1] - yc)**2) - radius))
        
        return xc, yc, radius, error
    
    def cartesian_to_polar(self, x, y):
        r = sqrt(x**2 + y**2)
        phi = atan2(y, x)
        return (r, phi)

    def filter_close_corners(self, corners, min_distance=0.05):
        if len(corners) == 0:
            return corners

        filtered_corners = [corners[0]]

        for corner in corners[1:]:
            if all(np.linalg.norm(np.array(corner) - np.array(fc)) > min_distance for fc in filtered_corners):
                filtered_corners.append(corner)

        return filtered_corners

    # def save_features(self, lines, circles, filename="features.json"):
        
    #     filename = "/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/data/features.json"
        
    #     # Prepare data to be saved
    #     data = {
    #         "lines": [{"slope": slope, "intercept": intercept} for slope, intercept in lines],
    #         "circles": [{"x_center": xc, "y_center": yc, "radius": radius} for xc, yc, radius in circles] if circles else []
    #     }

    #     # Save the data to a JSON file
    #     with open(filename, "w") as f:
    #         json.dump(data, f)

    #     rospy.loginfo(f"Features saved to {filename}")

    def visualize_lidar_data(self, points, corners, circles, lines, counter, save_folder = '/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/plots/FeatureExtraction', base_name="extraction_step"):
        # Check if this is the first call and clear the folder
        if self.first_call:
            # Remove all existing plots in the directory
            if os.path.exists(save_folder):
                for file_name in os.listdir(save_folder):
                    file_path = os.path.join(save_folder, file_name)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            rospy.loginfo(f"Deleted existing plot: {file_path}")
                    except Exception as e:
                        rospy.logerr(f"Error deleting file {file_path}: {e}")
            else:
                os.makedirs(save_folder)  # Create the directory if it doesn't exist
            self.first_call = False  # Set the flag to False after the first call

        plt.figure(figsize=(10, 10))
        
        # Plot the LiDAR scan points
        plt.scatter(points[:, 0], points[:, 1], c='gray', s=10, label="LiDAR Points")

        # Highlight detected corners
        if corners:
            corners = np.array(corners)
            plt.scatter(corners[:, 0], corners[:, 1], c='red', s=100, marker='x', label="Corners")
        
        # Highlight detected circle centers
        if circles and circles[0] is not None:
            circles = np.array(circles)
            plt.scatter(circles[:, 0], circles[:, 1], c='blue', s=100, marker='o', label="Circle Centers")

        # # Plot the detected RANSAC lines
        # for slope, intercept in lines:
        #     x_vals = np.array([min(points[:, 0]), max(points[:, 0])])
        #     y_vals = slope * x_vals + intercept
        #     plt.plot(x_vals, y_vals, label=f"Line: y={slope:.2f}x+{intercept:.2f}")
        
        for slope, intercept in lines:
            # Select the range of x values from the points in the cluster
            x_values = points[:, 0]
            x_min, x_max = np.min(x_values), np.max(x_values)

            # Calculate the corresponding y values for the line
            y_min = slope * x_min + intercept
            y_max = slope * x_max + intercept

            # Plot the line segment within the range of the LiDAR points
            plt.plot([x_min, x_max], [y_min, y_max], label=f"Line: y={slope:.2f}x+{intercept:.2f}", linewidth=3)
        
        # Set plot details
        plt.title("LiDAR Data with Detected Corners, Circles, and Lines")
        plt.xlabel("X [meters]")
        plt.ylabel("Y [meters]")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")

        # Generate filename and save plot
        filename = f"{base_name}_{counter}.png"
        filepath = os.path.join(save_folder, filename)
        plt.savefig(filepath)
        plt.close()

    def get_lines(self):
        return self.lines_data

    # Getter for corners
    def get_corners(self):
        return self.corners

    # Getter for circles
    def get_circles(self):
        return self.circles

    # Getter for points
    def get_points(self):
        return self.points