#!/usr/bin/env python

import rospy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from std_msgs.msg import Float64MultiArray
import os
import numpy as np
import psutil
import time
import csv

class SLAMPlotter:
    
    def __init__(self):
        rospy.init_node('slam_plotter', anonymous=True)

        # Subscribers
        self.ground_truth_sub = rospy.Subscriber('/ground_truth/state', Odometry, self.ground_truth_callback)
        self.ekf_path_sub = rospy.Subscriber('/EKF_Path', Marker, self.ekf_path_callback)
        self.landmark_sub = rospy.Subscriber('/slam_map', Marker, self.landmark_callback)
        self.covariance_sub = rospy.Subscriber('/slam_covariance', Float64MultiArray, self.covariance_callback)

        # Storage for paths, landmarks, and covariance
        self.ground_truth_positions = []
        self.ekf_positions = []
        self.landmarks = []
        self.covariances_x = []
        self.covariances_y = []
        self.covariances_theta = []

        # Resource usage tracking
        self.memory_usage = []
        self.cpu_usage = []

        # Tracking for metrics
        self.start_time = rospy.Time.now().to_sec()
        self.cumulative_distance = 0.0

        # Root path for all runs
        self.base_path = "/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/plots/slamPlotterEvaluationPlots"
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

        # Initialize run ID
        self.run_id = self.get_next_run_id()
        self.run_path = os.path.join(self.base_path, f"Run_{self.run_id}")
        os.makedirs(self.run_path)

        # Register shutdown hook
        rospy.on_shutdown(self.save_plots_and_metrics)
        
    def track_resources(self):

        self.memory_usage.append(psutil.virtual_memory().percent)
        self.cpu_usage.append(psutil.cpu_percent(interval=None))

    def get_next_run_id(self):
        """Read the current run ID from file and increment it for the next run."""
        counter_file = os.path.join(self.base_path, "run_counter.txt")
        if os.path.exists(counter_file):
            with open(counter_file, "r") as file:
                run_id = int(file.read().strip()) + 1
        else:
            run_id = 1
        # Save the incremented run_id back to file
        with open(counter_file, "w") as file:
            file.write(str(run_id))
        return run_id

    def ground_truth_callback(self, msg):
        position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.ground_truth_positions.append(position)

    def ekf_path_callback(self, msg):
        position = (msg.pose.position.x, msg.pose.position.y)
        self.ekf_positions.append(position)
        if len(self.ekf_positions) > 1:
            # Update cumulative distance
            prev_position = np.array(self.ekf_positions[-2])
            current_position = np.array(self.ekf_positions[-1])
            self.cumulative_distance += np.linalg.norm(current_position - prev_position)

    def landmark_callback(self, msg):
        for point in msg.points:
            position = (point.x, point.y)
            self.landmarks.append(position)

    def covariance_callback(self, msg):
        if len(msg.data) == 9:
            self.covariances_x.append(msg.data[0])      # Variance of x
            self.covariances_y.append(msg.data[4])      # Variance of y
            self.covariances_theta.append(msg.data[8])  # Variance of theta

    def calculate_ate(self):
        errors = [np.linalg.norm(np.array(gt) - np.array(ekf)) for gt, ekf in zip(self.ground_truth_positions, self.ekf_positions)]
        ate = np.sqrt(np.mean(np.square(errors)))
        return ate, errors

    def calculate_rpe(self):
        rpe_errors = []
        for i in range(1, len(self.ground_truth_positions)):
            gt_delta = np.array(self.ground_truth_positions[i]) - np.array(self.ground_truth_positions[i - 1])
            ekf_delta = np.array(self.ekf_positions[i]) - np.array(self.ekf_positions[i - 1])
            rpe_errors.append(np.linalg.norm(gt_delta - ekf_delta))
        rpe = np.sqrt(np.mean(np.square(rpe_errors)))
        return rpe, rpe_errors

    def save_metrics_to_csv(self, ate, rpe):
        """Save metrics to a CSV file in the run-specific directory."""
        csv_file = os.path.join(self.run_path, f"Run_{self.run_id}_Metrics.csv")
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["ATE", ate])
            writer.writerow(["RPE", rpe])
            writer.writerow(["Cumulative Distance", self.cumulative_distance])
            writer.writerow(["Average CPU Usage", np.mean(self.cpu_usage)])
            writer.writerow(["Average Memory Usage", np.mean(self.memory_usage)])

    def save_plots_and_metrics(self):
        """Generates and saves performance metrics and plots upon shutdown"""

        # Track final CPU and memory usage
        self.track_resources()
        
        # Calculate metrics
        ate, ate_errors = self.calculate_ate()
        rpe, rpe_errors = self.calculate_rpe()

        # Save metrics to CSV
        self.save_metrics_to_csv(ate, rpe)
        
        # Plot Ground Truth and EKF paths
        plt.figure(figsize=(10, 6))
        if self.ground_truth_positions:
            gt_x, gt_y = zip(*self.ground_truth_positions)
            plt.plot(gt_x, gt_y, 'g-', label="Ground Truth Path")
        if self.ekf_positions:
            ekf_x, ekf_y = zip(*self.ekf_positions)
            plt.plot(ekf_x, ekf_y, 'r--', label="EKF Path")
        if self.landmarks:
            lm_x, lm_y = zip(*self.landmarks)
            plt.scatter(lm_x, lm_y, c='b', marker='o', label="Landmarks")
        plt.xlabel("X Position (meters)")
        plt.ylabel("Y Position (meters)")
        plt.title("Comparison of Ground Truth Path, EKF Path, and Landmarks")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.run_path, f"Run_{self.run_id}_GT_vs_EKF_path.png"))
        plt.close()

        # Plot ATE over time
        plt.figure()
        plt.plot(ate_errors, label="ATE over time")
        plt.xlabel("Time step")
        plt.ylabel("Error (meters)")
        plt.legend()
        plt.title("Absolute Trajectory Error (ATE) over time")
        plt.savefig(os.path.join(self.run_path, f"Run_{self.run_id}_ATE_Over_Time.png"))
        plt.close()

        # Plot RPE over time
        plt.figure()
        plt.plot(rpe_errors, label="RPE over time")
        plt.xlabel("Time step")
        plt.ylabel("Error (meters)")
        plt.legend()
        plt.title("Relative Pose Error (RPE) over time")
        plt.savefig(os.path.join(self.run_path, f"Run_{self.run_id}_RPE_Over_Time.png"))
        plt.close()

        # Plot state covariance for x, y, and theta separately
        if self.covariances_x and self.covariances_y and self.covariances_theta:
            plt.figure()
            plt.plot(self.covariances_x, label="Covariance X (Variance of x)")
            plt.plot(self.covariances_y, label="Covariance Y (Variance of y)")
            plt.plot(self.covariances_theta, label="Covariance Theta (Variance of theta)")
            plt.xlabel("Time step")
            plt.ylabel("Variance")
            plt.legend()
            plt.title("State Covariance (Variance of x, y, theta) over time")
            plt.grid(True)
            plt.savefig(os.path.join(self.run_path, f"Run_{self.run_id}_State_Covariance.png"))
            plt.close()

        # Resource usage plots
        plt.figure()
        plt.plot(self.memory_usage, label="Memory Usage (%)")
        plt.plot(self.cpu_usage, label="CPU Usage (%)")
        plt.xlabel("Time step")
        plt.ylabel("Usage (%)")
        plt.legend()
        plt.title("Memory and CPU Usage over time")
        plt.savefig(os.path.join(self.run_path, f"Run_{self.run_id}_Resource_Usage.png"))
        plt.close()

        # Log results
        rospy.loginfo(f"ATE: {ate}")
        rospy.loginfo(f"RPE: {rpe}")
        rospy.loginfo(f"Cumulative Distance Travelled: {self.cumulative_distance}")
        rospy.loginfo("All plots and metrics saved in the run-specific folder.")

    def run(self):
        rate = rospy.Rate(1)  # 1 Hz
        while not rospy.is_shutdown():
            self.track_resources()
            rate.sleep()

if __name__ == '__main__':
    try:
        slam_plotter = SLAMPlotter()
        slam_plotter.run()
    except rospy.ROSInterruptException:
        pass
