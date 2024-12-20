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
from scipy.spatial import ConvexHull

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
    
    def extract_features_only_DBSCAN(self, scan_data, angle_min, angle_max, angle_increment, counter, eps=0.25, min_samples=5):
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

            # Lists to store cluster centroids and corner points
            polar_centroid_list = []
            cartesian_centroid_list = []
            corner_points_list = []
            polar_corner_list = []
            
            # Get the set of unique labels (each unique cluster has a unique label, -1 is noise)
            unique_labels = set(labels)
            
            # # Plotting setup
            # plt.figure(figsize=(8, 8))
            # colormap = plt.cm.get_cmap('tab20', len(unique_labels))
            
            iteration = 0
            
            for label in unique_labels:
                if label == -1:
                    continue  # Skip noise points
                
                iteration += 1

                # Extract points belonging to the current cluster
                cluster_points = valid_points[labels == label]
                
                # Calculate the centroid of the cluster (mean of the x and y coordinates)
                centroid_x, centroid_y = np.mean(cluster_points, axis=0)
                cartesian_centroid_list.append((centroid_x, centroid_y))

                # Convert the centroid from Cartesian to polar coordinates
                range_centroid = np.sqrt(centroid_x**2 + centroid_y**2)
                angle_centroid = np.arctan2(centroid_y, centroid_x)
                polar_centroid_list.append((range_centroid, angle_centroid))
                
                # corners = self.detect_corners_from_DBScan(cluster_points)
                corners = self.detect_corners_by_triplet(cluster_points)
                corners = self.filter_close_corners(corners)
                for corner in corners:
                    # Convert corner to polar coordinates
                    range_corner = np.sqrt(corner[0]**2 + corner[1]**2)
                    angle_corner = np.arctan2(corner[1], corner[0])
                    polar_corner_list.append((range_corner, angle_corner))
                
                # # Plot the cluster points
                # plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colormap(label), marker='o', s=10, label=f'Cluster {label}')
                
                # # Plot the centroid in the same color but with a larger, distinct marker
                # plt.scatter(centroid_x, centroid_y, marker='x', color=colormap(label), s=100, edgecolor='black', label=f'Centroid {label}')

                # # Plot corner points
                # for corner in corners:
                #     plt.scatter(corner[0], corner[1], color='red', marker='x', s=50, label=f'Corner {label}')

            
            # # Display corner points in the log
            # rospy.loginfo(f"Corner points detected: {corner_points_list}")

            # # Plot settings
            # plt.xlabel('X Coordinates [m]')
            # plt.ylabel('Y Coordinates [m]')
            # plt.title(f'DBSCAN Clustering and Corner Detection - Correction {counter}')
            # plt.legend(loc='upper right', fontsize='small')
            # plt.grid(True)

            # # Save or show plot
            # save_dir = '/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/plots/DBSCAN_Plots'
            # filename = f'detected_features_loop_{counter}.png'
            # filepath = os.path.join(save_dir, filename)
            # plt.savefig(filepath)
            # plt.close()

            
            # Return the list of centroids in polar coordinates
            return polar_corner_list
        
    def detect_corners_from_DBScan(self, cluster_points, angle_threshold=70, min_distance=0.2, exclude_edges=2, neighbor_offset=2):
        corners = []
        for i in range(exclude_edges, len(cluster_points) - exclude_edges - neighbor_offset):
            v1 = cluster_points[i] - cluster_points[i - neighbor_offset]
            v2 = cluster_points[i + neighbor_offset] - cluster_points[i]
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            angle_deg = np.degrees(angle)
            
            # Apply angle and distance filtering
            if angle_deg < angle_threshold:
                if len(corners) == 0 or np.linalg.norm(cluster_points[i] - corners[-1]) > min_distance:
                    corners.append(cluster_points[i])
        return corners

    def detect_corners_by_triplet(self, cluster_points, distance_threshold=0.04):
        corners = []
        for i in range(1, len(cluster_points) - 1):
            p1, p2, p3 = cluster_points[i - 1], cluster_points[i], cluster_points[i + 1]

            # Vector from p1 to p3
            line_vec = p3 - p1
            line_len = np.linalg.norm(line_vec)
            if line_len == 0:
                continue  # Skip if points are too close

            # Projection of p2 onto the line p1-p3
            projection = ((p2 - p1).dot(line_vec) / line_len**2) * line_vec
            proj_point = p1 + projection
            
            # Calculate distance from p2 to the line
            deviation = np.linalg.norm(p2 - proj_point)
            
            # Check if the deviation exceeds the threshold
            if deviation > distance_threshold:
                corners.append(p2)
                
        return corners

    def extract_features_from_scan(self, scan_data, angle_min, angle_max, angle_increment, counter, eps=0.25, min_samples=5):
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
            
            # self.visualize_dbscan_result(valid_points, labels, counter)

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
            all_lines = []
            circle_feature = []
            corner_features = []
            iteration = 0
            
            for cluster_label, cluster_points in cluster_dict.items():
               
                iteration += 1
                
                # Here you can decide whether to apply RANSAC for circles or lines
                # if self.is_circle(cluster_points, counter, iteration):
                #     circle_feature = circle_feature + self.detect_circles_ransac(cluster_points, counter, iteration)
                # else:
                line_features = self.detect_lines_ransac(cluster_points, counter, iteration)
                all_lines = all_lines + line_features
                corner_features = corner_features + self.detect_line_intersections(line_features, cluster_points)
                # self.visualize_features(valid_points, labels, line_features, corner_features, circle_feature, counter)
                
            max_valid_range = 3.5 # Max laser range reach

            if corner_features:
                
                # Filter invalid corner features
                corner_features[:] = [corner for corner in corner_features 
                          if self.cartesian_to_polar(corner[0], corner[1])[0] <= max_valid_range]
                
                corner_features = self.filter_close_corners(corner_features)
                
                # Convert corner features to polar coordinates and append to features list
                for corner in corner_features:
                    polar_corner = self.cartesian_to_polar(corner[0], corner[1])
                    features.append(polar_corner)

            # if circle_feature:
                
            #     # Filter invalid circle middle points out
            #     circle_feature[:] = [circle for circle in circle_feature 
            #              if self.cartesian_to_polar(circle[0], circle[1])[0] <= max_valid_range]
                
            #     circle_feature = self.filter_close_circle_centers(circle_feature)
                
            #     # Convert circle features to polar coordinates and append to features list
            #     for circle in circle_feature:
            #         circle_center_x, circle_center_y, radius = circle
            #         polar_circle_center = self.cartesian_to_polar(circle_center_x, circle_center_y)
            #         # Append as (r, phi, radius)
            #         features.append(polar_circle_center)

            # self.visualize_features(valid_points, labels, all_lines, corner_features, circle_feature, counter)

            # rospy.loginfo(f"\n Following features detected: {features}")
            
            return features
    
    def extract_features_from_scan_SplitAndMerge(self, scan_data, angle_min, angle_max, angle_increment, counter, eps=0.25, min_samples=5):
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
            
            # self.visualize_dbscan_result(valid_points, labels, counter)

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
            max_valid_range = 3.5  # Max laser range reach
            all_segments = []
            all_corners = []
            circle = []
            iteration = 0
            
            for cluster_label, cluster_points in cluster_dict.items():
               
                iteration += 1
                
                segments, corners = self.split_and_merge(cluster_points)
                valid_corners = [corner for corner in corners if np.linalg.norm(corner) <= max_valid_range]
                
                all_segments += segments
                all_corners += valid_corners

                # Convert corner features to polar coordinates and append to features list
                for corner in valid_corners:
                    polar_corner = self.cartesian_to_polar(corner[0], corner[1])
                    features.append(polar_corner)
                
            # self.visualize_features(valid_points, labels, all_segments, all_corners, circle, counter)
            
            return features
        
    def is_circle(self, cluster_points, loopCounter, iteration, variance_threshold=0.05, min_inliers=12, angular_threshold=np.pi/4):
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
        # rospy.loginfo(f"Radius Variance: {radius_variance}, Angular Spread: {angular_spread}")
        
        if radius_variance < variance_threshold and len(cluster_points) >= min_inliers and angular_spread > angular_threshold:
            isCircle = True
        else:
            isCircle = False
        
        # self.visualize_circle_classification(cluster_points, radius_variance, angular_spread, mean_x, mean_y, isCircle, loopCounter, iteration, variance_threshold, min_inliers, angular_threshold)
        
        # Classify as a circle based on the variance of radii and angular spread
        return radius_variance < variance_threshold and len(cluster_points) >= min_inliers and angular_spread > angular_threshold

    def visualize_features(self, valid_points, labels, line_features, corner_features, circle_features, loopCounter):
        save_dir = "/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/plots/FeatureExtraction"
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Create a new figure
        plt.figure(figsize=(10, 10))

        # Plot all valid points (LiDAR points) with a light color
        plt.scatter(valid_points[:, 0], valid_points[:, 1], c='gray', label='LiDAR Points', alpha=0.5)

        # Plot detected lines (in blue)
        if line_features:
            for i, (slope, intercept) in enumerate(line_features):
                x_vals = np.array([valid_points[:, 0].min(), valid_points[:, 0].max()])  # x-range for the line
                y_vals = slope * x_vals + intercept  # y = mx + b for the line
                plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='Detected Line')
                
                mid_x = (x_vals[0] + x_vals[1]) / 2
                mid_y = (y_vals[0] + y_vals[1]) / 2
                plt.text(mid_x, mid_y, f'{i+1}', fontsize=12, color='blue', ha='center')

        # Plot detected circles (in green)
        # if circle_features:
        #     for i, circle in enumerate(circle_features):
        #         # Unpack the tuple from the list inside 'circle'
        #         xc, yc, radius = circle
        #         circle_patch = plt.Circle((xc, yc), radius, color='g', fill=False, linewidth=2, label='Detected Circle')
        #         plt.gca().add_patch(circle_patch)
        #         plt.text(xc, yc, f'{i+1}', fontsize=12, color='green', ha='center')
        
        # Plot detected corners/intersections (in red)
        if corner_features:
            for i, (x, y) in enumerate(corner_features):
                plt.scatter(x, y, c='r', label='Corner/Intersection', s=100)
                plt.text(x, y, f'{i+1}', fontsize=12, color='red', ha='center')

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
        # plt.show()
        plt.savefig(filepath)
        plt.close()

    def detect_lines_ransac(self, points, loopCounter, iteration, residual_threshold=0.03, min_samples=3, max_trials=50, stop_probability=0.95, min_inliers=5):
        lines = []
        remaining_points = points.copy()
        iteration = iteration

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
        
        # rospy.loginfo(f"Total lines detected: {len(lines)} at loop {loopCounter}")

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

    def detect_line_intersections(self, lines, points, parallel_tolerance=0.2, distance_threshold=0.2):
        intersections = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                slope1, intercept1 = lines[i]
                slope2, intercept2 = lines[j]

                # Check if lines are parallel
                if abs(slope1 - slope2) < parallel_tolerance:
                    continue

                # Calculate the intersection point (x, y)
                x_intersection = (intercept2 - intercept1) / (slope1 - slope2)
                y_intersection = slope1 * x_intersection + intercept1
                
                intersections.append((x_intersection, y_intersection))
                
        # Filter intersections based on proximity to points in clusters
        filtered_intersections = []
        for intersection in intersections:
            x_inter, y_inter = intersection
            distances = np.linalg.norm(points - np.array([x_inter, y_inter]), axis=1)
            
            # Keep the intersection if it is within the distance threshold of any point in the cluster
            if np.any(distances < distance_threshold):
                filtered_intersections.append(intersection)

        return intersections

    def detect_circles_ransac(self, points, loopCounter, iteration, max_error=0.05, min_samples=5, max_trials=1000, stop_inliers=0.9):

        circles = []
        remaining_points = points.copy()
        iteration = 0

        while len(remaining_points) > min_samples:
            iteration += 1

            best_circle = None
            best_inliers = None
            best_error = float('inf')
            
            all_circles = []

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
                
                # all_circles.append((xc, yc, radius, inliers, error, sample_points))

                if len(inliers) > min_samples and error < best_error:
                    best_circle = (xc, yc, radius)
                    best_inliers = inliers
                    best_error = error

                    # Early stop if enough inliers found
                    if len(inliers) / n_points > stop_inliers:
                        # rospy.loginfo(f"Early stopping RANSAC with {len(inliers)} inliers.")
                        break

            if best_circle is None:
                break  # No valid circle found, stop the process

            # Store the detected circle in a similar format to lines
            circles.append(best_circle)

            # self.circles_data.append({
            #     "iteration": iteration,
            #     "loopCounter": loopCounter,
            #     "xc": best_circle[0],
            #     "yc": best_circle[1],
            #     "radius": best_circle[2],
            #     "inliers": remaining_points[best_inliers].tolist(),  # Save inlier points for visualization
            #     "outliers": remaining_points[np.logical_not(np.isin(np.arange(len(remaining_points)), best_inliers))].tolist()  # Save outlier points for visualization
            # })

            # Remove inliers from the point set to detect more circles
            remaining_points = remaining_points[np.logical_not(np.isin(np.arange(len(remaining_points)), best_inliers))]

        # self.visualize_circles(all_circles, loopCounter, iteration)

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
    
    def visualize_circles(self, circles_data, loopCounter, iteration, title="Detected Circles"):
        
        save_dir = "/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/plots/RansacCircleIteration"
        
        fig, ax = plt.subplots()
        
        # Set axis limits based on the expected range of your data
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        
        # Plot each circle and its corresponding inliers and outliers
        for circle_info in circles_data:
            xc, yc, radius, inliers, error, sample_points = circle_info
            
            # Plot circle
            circle = plt.Circle((xc, yc), radius, color='g', fill=False, linewidth=2, label="Detected Circle")
            ax.add_patch(circle)
            
            inlier_points = sample_points
            
            # Plot inliers
            if len(inlier_points) > 0:
                ax.scatter(inlier_points[:, 0], inlier_points[:, 1], color='b', label="Inliers", s=15)
            
            # Plot outliers
            if len(sample_points) > 0:
                ax.scatter(sample_points[:, 0], sample_points[:, 1], color='r', label="Sample points", s=15, marker='x')
        
        # Add labels and title
        ax.set_xlabel("X [meters]")
        ax.set_ylabel("Y [meters]")
        ax.set_title(title)
        
        # Show legend
        ax.legend(loc="best")
        
        # Show the plot
        plt.grid(True)
        filename = f'detected_circles_loop_{loopCounter}_iteration_{iteration}.png'
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.close()
    
    def cartesian_to_polar(self, x, y):
        r = sqrt(x**2 + y**2)
        phi = atan2(y, x)
        return (r, phi)

    def filter_close_corners(self, corners, min_distance=0.5):
        if len(corners) == 0:
            return corners

        filtered_corners = [corners[0]]

        for corner in corners[1:]:
            if all(np.linalg.norm(np.array(corner) - np.array(fc)) > min_distance for fc in filtered_corners):
                filtered_corners.append(corner)

        return filtered_corners
    
    def filter_close_circle_centers(self, circle_feature, min_distance = 0.7):
        filtered_circles = []
        
        for i, circle in enumerate(circle_feature):
            keep_circle = True
            circle_x, circle_y, radius = circle
            
            for filtered_circle in filtered_circles:
                filtered_x, filtered_y, _ = filtered_circle
                distance = self.euclidean_distance(circle_x, circle_y, filtered_x, filtered_y)
                if distance < min_distance:
                    keep_circle = False
                    break
            
            if keep_circle:
                filtered_circles.append(circle)
        
        return filtered_circles
    
    def euclidean_distance(self, x1, y1, x2, y2):
        return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

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
                            # rospy.loginfo(f"Deleted existing plot: {file_path}")
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

    def visualize_dbscan_result(self, points, labels, loop_counter):
        
        save_dir = "/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/plots/DBSCAN_Plots"
        
        # Create a figure for the DBSCAN result
        plt.figure(figsize=(10, 10))

        # Get unique labels (clusters and noise points labeled as -1)
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

        # Plot each cluster with a different color
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise points (label -1)
                col = [0, 0, 0, 1]

            # Filter points belonging to the current cluster (k)
            class_member_mask = (labels == k)
            xy = points[class_member_mask]

            # Plot the points for this cluster
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=8, label=f'Cluster {k}' if k != -1 else 'Noise')

        # Set plot title and labels
        plt.title(f'DBSCAN Clustering - Loop {loop_counter}')
        plt.xlabel('X [meters]')
        plt.ylabel('Y [meters]')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')

        # Save the plot
        filename = f'dbscan_result_loop_{loop_counter}.png'
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.close()
    
    def visualize_circle_classification(self, cluster_points, radius_variance, angular_spread, mean_x, mean_y, isCircle, loop_counter, iteration, variance_threshold, min_inliers, angular_threshold):
        
        save_dir = "/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/plots/Circle_Classification"
        
        x_coords, y_coords = cluster_points[:, 0], cluster_points[:, 1]

        # Visualization
        fig, ax = plt.subplots()
        
        # Plot cluster points
        ax.scatter(x_coords, y_coords, color='blue', label='Cluster Points', s=30)
        
        # Plot the center point
        ax.scatter(mean_x, mean_y, color='red', marker='x', label='Center', s=100)

        # Plot lines from center to each point
        for i in range(len(x_coords)):
            ax.plot([mean_x, x_coords[i]], [mean_y, y_coords[i]], 'g--', alpha=0.5)  # Dashed green lines

        # Set axis limits to ensure all points are visible
        ax.set_xlim([min(x_coords) - 1, max(x_coords) + 1])
        ax.set_ylim([min(y_coords) - 1, max(y_coords) + 1])
        
        # Add title and labels
        ax.set_title(f"Circle Classification\nVariance: {radius_variance:.4f}, Angular Spread: {angular_spread:.4f}, Is Circle: {isCircle}")
        ax.set_xlabel("X Coordinates")
        ax.set_ylabel("Y Coordinates")
        
        # Show legend
        ax.legend()
        
        # Add thresholds and other info as a text box on the side
        threshold_text = (f"Thresholds:\n"
                        f"Variance Threshold: {variance_threshold}\n"
                        f"Min Inliers: {min_inliers}\n"
                        f"Angular Threshold: {angular_threshold}\n"
                        f"Is Circle: {isCircle}")

        # Place the text box on the side of the plot (upper left corner)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.8, 0.2, threshold_text, transform=ax.transAxes, fontsize=6.5,
                verticalalignment='top', bbox=props)
        
        # Display the plot
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        # Save the plot
        filename = f'circleClassification_result_loop_{loop_counter}_iteration_{iteration}.png'
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        # plt.show()
        plt.close()
    
    def get_lines(self):
        return self.lines_data

    def get_corners(self):
        return self.corners

    def get_circles(self):
        return self.circles

    def get_points(self):
        return self.points
    
    def split_and_merge(self, points, split_threshold=0.005, min_points=3):
        def fit_line_segment(segment_points):
            x = segment_points[:, 0]
            y = segment_points[:, 1]
            A = np.vstack([x, np.ones(len(x))]).T
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
            return slope, intercept

        def point_line_distance(point, line):
            slope, intercept = line
            x, y = point
            return abs(slope * x - y + intercept) / np.sqrt(slope**2 + 1)

        segments = []
        stack = [(0, len(points) - 1)]
        print(f"Initial stack: {stack}")

        while stack:
            start_idx, end_idx = stack.pop()
            segment_points = points[start_idx:end_idx + 1]

            # Avoid splitting very small segments
            if len(segment_points) < min_points:
                print(f"Segment too small, skipped: start_idx={start_idx}, end_idx={end_idx}")
                continue

            line = fit_line_segment(segment_points)

            distances = [point_line_distance(p, line) for p in segment_points]
            max_distance = max(distances)
            max_index = start_idx + distances.index(max_distance)

            # Prevent infinite loop by checking if max_index is at start or end of the segment
            if max_distance > split_threshold and max_index != start_idx and max_index != end_idx:
                print(f"Splitting segment at max_index={max_index}")
                stack.append((start_idx, max_index))
                stack.append((max_index, end_idx))
            else:
                print(f"Adding segment: start_idx={start_idx}, end_idx={end_idx}")
                segments.append((line, (points[start_idx], points[end_idx])))

            print(f"Current stack: {stack}")

        # Extract corners as intersections between segments
        corners = []
        for i in range(1, len(segments)):
            line1, end_points1 = segments[i - 1]
            line2, end_points2 = segments[i]
            if abs(line1[0] - line2[0]) > 1e-6:
                x_intersect = (line2[1] - line1[1]) / (line1[0] - line2[0])
                y_intersect = line1[0] * x_intersect + line1[1]
                if min(end_points1[1][0], end_points2[0][0]) <= x_intersect <= max(end_points1[1][0], end_points2[0][0]):
                    corners.append((x_intersect, y_intersect))

        print(f"Total segments found: {len(segments)}")
        return segments, corners
