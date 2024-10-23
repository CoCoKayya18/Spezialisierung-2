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
        self.circles_data = []
        self.corners = None
        self.circles = None
        self.points = None
        
        pass

    def extract_features_from_scan(self, scan_data, angle_min, angle_max, angle_increment, counter, eps=0.2, min_samples=5):
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
            # features = []
            cluster_dict = {}
            unique_labels = set(labels)
            
            for label in unique_labels:
                if label == -1:
                    continue  # Skip noise points

                # Extract points belonging to the current cluster
                cluster_points = valid_points[labels == label]
                
                # Save cluster points in a dictionary
                cluster_dict[label] = cluster_points
            
            features = []
            line_features = []
            circle_feature = []
            corner_features = []
            
            for cluster_label, cluster_points in cluster_dict.items():
                # Here you can decide whether to apply RANSAC for circles or lines
                if self.is_circle(cluster_points):
                    circle_feature = circle_feature + self.detect_circles_ransac(cluster_points, counter)
                else:
                    line_features = line_features + self.detect_lines_ransac(cluster_points, counter)
                    corner_features = corner_features + self.detect_line_intersections(line_features)

            if corner_features:
                # Convert corner features to polar coordinates and append to features list
                for corner in corner_features:
                    polar_corner = self.cartesian_to_polar(corner[0], corner[1])
                    features.append(polar_corner)

            if circle_feature:
                # Convert circle features to polar coordinates and append to features list
                for circle in circle_feature:
                    circle_center_x, circle_center_y, radius = circle
                    polar_circle_center = self.cartesian_to_polar(circle_center_x, circle_center_y)
                    # Append as (r, phi, radius)
                    features.append(polar_circle_center)

            self.visualize_features(valid_points, labels, line_features, corner_features, circle_feature, counter)

            rospy.loginfo(f"\n Following features detected: {features}")
            
            return features
        
    def is_circle(self, cluster_points, variance_threshold=0.0075, min_inliers=10, angular_threshold=np.pi/2):
        # Check if the points in the cluster form a circular pattern using the variance of radii
        x_coords, y_coords = cluster_points[:, 0], cluster_points[:, 1]
        mean_x, mean_y = np.mean(x_coords), np.mean(y_coords)
        
        # Compute distances from the center (mean_x, mean_y)
        distances = np.sqrt((x_coords - mean_x)**2 + (y_coords - mean_y)**2)
        radius_variance = np.var(distances)

        # Calculate angular spread
        angles = np.arctan2(y_coords - mean_y, x_coords - mean_x)
        angular_spread = np.max(angles) - np.min(angles)
        
        # Normalize angles to handle angle wrap-around (i.e., between -pi and pi)
        if angular_spread > np.pi:
            angular_spread = 2 * np.pi - angular_spread
        
        # Log radius variance and angular spread for debugging
        rospy.loginfo(f"Radius Variance: {radius_variance}, Angular Spread: {angular_spread}")
        
        # Classify as a circle based on the variance of radii and angular spread
        return radius_variance < variance_threshold and len(cluster_points) >= min_inliers and angular_spread > angular_threshold

    def visualize_features(self, valid_points, labels, line_features, corner_features, circle_features, loopCounter):
        save_dir = "/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/plots/FeatureExtraction"
        
        rospy.loginfo(f"Line features: {line_features}")
        rospy.loginfo(f"Corner features: {corner_features}")
        rospy.loginfo(f"Circle features: {circle_features}")
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Create a new figure
        plt.figure(figsize=(10, 10))

        # Plot all valid points (LiDAR points) with a light color
        plt.scatter(valid_points[:, 0], valid_points[:, 1], c='gray', label='LiDAR Points', alpha=0.5)

        # Plot detected lines (in blue)
        if line_features:
            for slope, intercept in line_features:
                x_vals = np.array([valid_points[:, 0].min(), valid_points[:, 0].max()])  # x-range for the line
                y_vals = slope * x_vals + intercept  # y = mx + b for the line
                plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='Detected Line')

        # Plot detected circles (in green)
        if circle_features:
            for circle in circle_features:
                # Unpack the tuple from the list inside 'circle'
                xc, yc, radius = circle
                circle_patch = plt.Circle((xc, yc), radius, color='g', fill=False, linewidth=2, label='Detected Circle')
                plt.gca().add_patch(circle_patch)
        
        # Plot detected corners/intersections (in red)
        if corner_features:
            for x, y in corner_features:
                plt.scatter(x, y, c='r', label='Corner/Intersection', s=100)

        # Set plot details
        plt.title(f'Detected Features - Loop {loopCounter}')
        plt.xlabel('X [meters]')
        plt.ylabel('Y [meters]')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')

        # Save the plot
        filename = f'detected_features_loop_{loopCounter}.png'
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.close()

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
        corners = self.filter_corners_by_distance(corners)
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
            
            rospy.loginfo(f"Current lines array: {lines}")
            
            #  # Visualize current iteration
            # self.utils.plot_async(self.visualize_ransac_iteration, remaining_points, inlier_mask, slope, intercept, iteration, loopCounter)
            self.visualize_ransac_iteration(remaining_points, inlier_mask, slope, intercept, iteration, loopCounter)
            
            rospy.loginfo(f"Current 2 step lines array: {lines}")
            
            # self.lines_data.append({
            #     "iteration": iteration,
            #     "loopCounter": loopCounter,
            #     "slope": slope,
            #     "intercept": intercept,
            #     "inliers": remaining_points[inlier_mask].tolist(),  # Save inlier points for visualization
            #     "outliers": remaining_points[outlier_mask].tolist()  # Save outlier points for visualization
            # })

            # Remove inliers from the point set to detect more lines
            remaining_points = remaining_points[outlier_mask]
            
            rospy.loginfo(f"Current 3 step lines array: {lines}")
        
        rospy.loginfo(f"Total lines detected: {len(lines)} at loop {loopCounter}")

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

    def detect_circles_ransac(self, points, loopCounter, max_error=0.05, min_samples=5, max_trials=1000, stop_inliers=0.9):

        circles = []
        remaining_points = points.copy()
        iteration = 0

        while len(remaining_points) > min_samples:
            iteration += 1

            best_circle = None
            best_inliers = None
            best_error = float('inf')

            n_points = len(remaining_points)
            if n_points < min_samples:
                rospy.logwarn(f"Not enough points to detect a circle (min_samples={min_samples}, got {n_points}).")
                break

            for _ in range(max_trials):
                # Randomly sample points for circle fitting
                sample_indices = np.random.choice(n_points, min_samples, replace=False)
                sample_points = remaining_points[sample_indices]

                xs = sample_points[:, 0]
                ys = sample_points[:, 1]

                # Fit a circle to the sample points
                xc, yc, radius, error = self.fit_circle(xs, ys)

                if error is None or radius <= 0:
                    continue  # Skip invalid circle fits

                # Find inliers: points within max_error distance from the circle
                distances = np.sqrt((remaining_points[:, 0] - xc) ** 2 + (remaining_points[:, 1] - yc) ** 2)
                inliers = np.where(np.abs(distances - radius) < max_error)[0]

                if len(inliers) > min_samples and error < best_error:
                    best_circle = (xc, yc, radius)
                    best_inliers = inliers
                    best_error = error

                    # Early stop if enough inliers found
                    if len(inliers) / n_points > stop_inliers:
                        rospy.loginfo(f"Early stopping RANSAC with {len(inliers)} inliers.")
                        break

            if best_circle is None:
                break  # No valid circle found, stop the process

            # Store the detected circle in a similar format to lines
            circles.append(best_circle)

            self.circles_data.append({
                "iteration": iteration,
                "loopCounter": loopCounter,
                "xc": best_circle[0],
                "yc": best_circle[1],
                "radius": best_circle[2],
                "inliers": remaining_points[best_inliers].tolist(),  # Save inlier points for visualization
                "outliers": remaining_points[np.logical_not(np.isin(np.arange(len(remaining_points)), best_inliers))].tolist()  # Save outlier points for visualization
            })

            # Remove inliers from the point set to detect more circles
            remaining_points = remaining_points[np.logical_not(np.isin(np.arange(len(remaining_points)), best_inliers))]

        return circles
    
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

    def filter_corners_by_distance(self, corners, max_distance=3.5):
        # Filter corners based on the distance from the origin (LiDAR position)
        filtered_corners = [corner for corner in corners if np.linalg.norm(corner) <= max_distance]
        return filtered_corners

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