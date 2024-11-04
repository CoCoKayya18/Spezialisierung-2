#!/usr/bin/env python

import rospy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from std_msgs.msg import Float64MultiArray
import os
import numpy as np
import psutil
import time
import csv
from scipy.stats import kstest, ttest_rel, wilcoxon, t
import scipy.stats as st
import pandas as pd 

class SLAMPlotter:
    
    def __init__(self):
        rospy.init_node('slam_plotter', anonymous=True)

        # Subscribers
        self.ground_truth_sub = rospy.Subscriber('/ground_truth/state', Odometry, self.ground_truth_callback)
        self.ekf_state_sub = rospy.Subscriber('/EKF_State', Odometry, self.ekf_state_callback)
        self.landmark_sub = rospy.Subscriber('/slam_map', Marker, self.landmark_callback)
        # self.covariance_sub = rospy.Subscriber('/slam_covariance', Float64MultiArray, self.covariance_callback)

        # Storage for paths, landmarks, and covariance with timestamps
        self.ground_truth_positions = []
        self.ground_truth_times = []
        self.ekf_positions = []
        self.aligned_ground_truth = []
        self.aligned_ekf_positions = []
        self.ekf_times = []
        self.landmarks = []
        self.covariances_x = []
        self.covariances_y = []
        self.covariances_theta = []

        # Load ground truth landmark positions from CSV
        self.ground_truth_landmarks = self.load_ground_truth_landmarks()
        
        # Resource usage tracking
        self.memory_usage = []
        self.cpu_usage = []

        # Root path for all runs
        self.base_path = "/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/slamPlotterEvaluationPlots"
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

        # Initialize run ID
        self.run_id = self.get_next_run_id()
        
        # CSV for metrics for all runs
        self.evaluation_csv = os.path.join(self.base_path, "SlamPlotterEvaluation.csv")
        self.nees_csv = os.path.join(self.base_path, f"Run_{self.run_id}/Run_{self.run_id}__NEES_values.csv")
        
        self.run_path = os.path.join(self.base_path, f"Run_{self.run_id}")
        os.makedirs(self.run_path)

        # Register shutdown hook
        rospy.on_shutdown(self.save_plots_and_metrics)
        
    def track_resources(self):
        self.memory_usage.append(psutil.virtual_memory().percent)
        self.cpu_usage.append(psutil.cpu_percent(interval=None))

    def get_next_run_id(self):

        counter_file = os.path.join(self.base_path, "run_counter.txt")
        
        with open(counter_file, "r") as file:
            run_id = int(file.read().strip()) + 1
        
        # Save the incremented run_id back to file
        with open(counter_file, "w") as file:
            file.write(str(run_id))
        return run_id

    def ground_truth_callback(self, msg):
        position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.ground_truth_positions.append(position)
        self.ground_truth_times.append(msg.header.stamp.to_sec())

    def ekf_state_callback(self, msg):
        position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        timestamp = msg.header.stamp.to_sec()
        self.ekf_positions.append(position)
        self.ekf_times.append(timestamp)

        # Find the closest ground truth position based on timestamp
        closest_gt_idx = min(range(len(self.ground_truth_times)), key=lambda i: abs(self.ground_truth_times[i] - timestamp))
        closest_gt_position = self.ground_truth_positions[closest_gt_idx]

        # Align the ground truth and EKF positions
        self.aligned_ground_truth.append(closest_gt_position)
        self.aligned_ekf_positions.append(position)

        if len(msg.pose.covariance) >= 9:  # Ensure that covariance data is available
            self.covariances_x.append(msg.pose.covariance[0])      # Variance of x
            self.covariances_y.append(msg.pose.covariance[4])      # Variance of y
            self.covariances_theta.append(msg.pose.covariance[8])  # Variance of theta

    def landmark_callback(self, msg):
        for point in msg.points:
            position = (point.x, point.y)
            self.landmarks.append(position)

    def calculate_ate(self):
        # Use the aligned ground truth and EKF positions for ATE calculation
        errors = [np.linalg.norm(np.array(gt) - np.array(ekf)) for gt, ekf in zip(self.aligned_ground_truth, self.aligned_ekf_positions)]
        ate = np.sqrt(np.mean(np.square(errors)))
        return ate, errors

    def calculate_rpe(self):
        rpe_errors = []
        # Ensure both lists have enough elements to calculate RPE
        min_length = min(len(self.aligned_ground_truth), len(self.aligned_ekf_positions))
        if min_length < 2:
            rospy.logwarn("Not enough data points to calculate RPE.")
            return float('nan'), rpe_errors  # Return NaN and an empty list if not enough points

        for i in range(1, min_length):
            gt_delta = np.array(self.aligned_ground_truth[i]) - np.array(self.aligned_ground_truth[i - 1])
            ekf_delta = np.array(self.aligned_ekf_positions[i]) - np.array(self.aligned_ekf_positions[i - 1])
            rpe_errors.append(np.linalg.norm(gt_delta - ekf_delta))
        
        rpe = np.sqrt(np.mean(np.square(rpe_errors))) if rpe_errors else float('nan')
        return rpe, rpe_errors
    
    def calculate_rmse(self):
        squared_errors = [(np.array(gt) - np.array(ekf)) ** 2 for gt, ekf in zip(self.aligned_ground_truth, self.aligned_ekf_positions)]
        rmse = np.sqrt(np.mean([np.sum(error) for error in squared_errors]))
        return rmse
    
    def save_nees_values(self, nees_values):
        # Write NEES values to a CSV for this run
        with open(self.nees_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["NEES Value"])
            for value in nees_values:
                writer.writerow([value])
    
    def calculate_nees(self):
        nees_values = []
        
        for gt, ekf, cov_x, cov_y in zip(self.aligned_ground_truth, self.aligned_ekf_positions, self.covariances_x, self.covariances_y):
            error = np.array(gt) - np.array(ekf)
            
            # Check for zero covariance values to prevent division by zero
            if cov_x > 0 and cov_y > 0:
                # Only calculate NEES if both cov_x and cov_y are positive
                nees = (error[0]**2 / cov_x) + (error[1]**2 / cov_y)
            else:
                # If cov_x or cov_y is zero, set NEES to NaN to handle it later
                nees = float('nan')
                
            nees_values.append(nees)
        
        self.save_nees_values(nees_values)
        
        # Filter out NaN or infinite values before calculating the mean and standard deviation
        valid_nees_values = [value for value in nees_values if np.isfinite(value)]
        
        # Calculate mean and standard deviation of valid NEES values
        mean_nees = np.mean(valid_nees_values) if valid_nees_values else float('nan')
        std_nees = np.std(valid_nees_values) if valid_nees_values else float('nan')
        
        return mean_nees, std_nees
    
    def error_distribution_analysis(self):
        x_errors = [gt[0] - ekf[0] for gt, ekf in zip(self.ground_truth_positions, self.ekf_positions)]
        y_errors = [gt[1] - ekf[1] for gt, ekf in zip(self.ground_truth_positions, self.ekf_positions)]
        
        plt.hist(x_errors, bins=30, alpha=0.5, label='X Errors')
        plt.hist(y_errors, bins=30, alpha=0.5, label='Y Errors')
        plt.legend()
        plt.title("Error Distribution")
        plt.savefig(os.path.join(self.run_path, f"Run_{self.run_id}_Error_Distribution.png"))
        plt.close()
        
        ks_x = kstest(x_errors, 'norm')
        ks_y = kstest(y_errors, 'norm')
        return ks_x, ks_y
    
    def confidence_interval_analysis(self, metric_values):
        mean_metric = np.mean(metric_values)
        confidence_interval = st.t.interval(0.95, len(metric_values)-1, loc=mean_metric, scale=st.sem(metric_values))
        return mean_metric, confidence_interval
    
    def significance_test(self):
        errors = [np.linalg.norm(np.array(gt) - np.array(ekf)) for gt, ekf in zip(self.ground_truth_positions, self.ekf_positions)]
        t_stat, p_value = ttest_rel(errors, np.zeros(len(errors)))
        w_stat, p_value_wilcoxon = wilcoxon(errors)
        return (t_stat, p_value), (w_stat, p_value_wilcoxon)
        
    def load_ground_truth_landmarks(self):

        gt_landmark_file = "/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/features/cornerGroundTruthPositions.csv"
        try:
            return pd.read_csv(gt_landmark_file)[['X', 'Y']].values
        except Exception as e:
            rospy.logerr(f"Error loading ground truth landmarks: {e}")
            return []

    def save_metrics_to_csv(self, ate, mean_ate, ci_ate, rpe, mean_rpe, ci_rpe, rmse, mean_nees, std_nees, ks_x, ks_y, significance_t, significance_w):
        
        # Check if the evaluation CSV exists, and add headers if it doesn't
        file_exists = os.path.isfile(self.evaluation_csv)
        file_is_empty = os.path.getsize(self.evaluation_csv) == 0 if file_exists else True

        with open(self.evaluation_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            if file_is_empty:
                # Write the header if the file is new
                writer.writerow(["Run_ID", "ATE", "Mean_ATE", "CI_ATE", "RPE", "Mean_RPE", "Mean_CI", "RMSE", "Mean_NEES", "Std_Nees", "Kolmogorov-Smirnov test (X errors)", "Kolmogorov-Smirnov test (Y errors)", "T-test", "Wilcoxon signed-rank test", "Average_CPU_Usage", "Average_Memory_Usage"])
            # Write the metrics for the current run
            writer.writerow([self.run_id, ate, mean_ate, ci_ate, rpe, mean_rpe, ci_rpe, rmse, mean_nees, std_nees, ks_x, ks_y, significance_t, significance_w, np.mean(self.cpu_usage), np.mean(self.memory_usage)])

    def save_plots_and_metrics(self):

        # Track final CPU and memory usage
        self.track_resources()
        
        # Calculate metrics
        ate, ate_errors = self.calculate_ate()
        rpe, rpe_errors = self.calculate_rpe()
        rmse = self.calculate_rmse()
        mean_nees, std_nees = self.calculate_nees()
        ks_x, ks_y = self.error_distribution_analysis()
        significance_t, significance_w = self.significance_test()
        mean_ate, ci_ate = self.confidence_interval_analysis(ate_errors)
        mean_rpe, ci_rpe = self.confidence_interval_analysis(rpe_errors)

        # Append metrics to the combined evaluation CSV
        self.save_metrics_to_csv(ate, mean_ate, ci_ate, rpe, mean_rpe, ci_rpe, rmse, mean_nees, std_nees, ks_x, ks_y, significance_t, significance_w)
        
        # Plot Ground Truth and EKF paths iwith landmarks
        plt.figure(figsize=(10, 6))
        if self.ground_truth_positions:
            gt_x, gt_y = zip(*self.ground_truth_positions)
            plt.plot(gt_x, gt_y, 'g-', label="Ground Truth Path")
            
        if self.ekf_positions:
            ekf_x, ekf_y = zip(*self.ekf_positions)
            plt.plot(ekf_x, ekf_y, 'r-', label="EKF Path")
            
        if self.landmarks:
            lm_x, lm_y = zip(*self.landmarks)
            plt.scatter(lm_x, lm_y, c='b', marker='o', label="Landmarks")
            
        plt.xlabel("X Position [m]")
        plt.ylabel("Y Position [m]")
        plt.title("Comparison of Ground Truth Path and Estimated Path with Landmarks")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.run_path, f"Run_{self.run_id}_GT_vs_EKF_path_with_LM.png"))
        plt.close()
        
        # Plot Ground Truth and EKF paths without landmarks
        plt.figure(figsize=(10, 6))
        if self.ground_truth_positions:
            gt_x, gt_y = zip(*self.ground_truth_positions)
            plt.plot(gt_x, gt_y, 'g-', label="Ground Truth Path")
            
        if self.ekf_positions:
            ekf_x, ekf_y = zip(*self.ekf_positions)
            plt.plot(ekf_x, ekf_y, 'r-', label="EKF Path")
            
        plt.xlabel("X Position [m]")
        plt.ylabel("Y Position [m]")
        plt.title("Comparison of Ground Truth Path and Estimated Path")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.run_path, f"Run_{self.run_id}_GT_vs_EKF_path.png"))
        plt.close()

        # Plot ATE over time
        plt.figure()
        plt.plot(ate_errors, label="ATE over time")
        plt.xlabel("Time step [s]")
        plt.ylabel("Error [m]")
        plt.legend()
        plt.title("Absolute Trajectory Error (ATE) over time")
        plt.grid(True)
        plt.savefig(os.path.join(self.run_path, f"Run_{self.run_id}_ATE_Over_Time.png"))
        plt.close()

        # Plot RPE over time
        plt.figure()
        plt.plot(rpe_errors, label="RPE over time")
        plt.xlabel("Time step [s]")
        plt.ylabel("Error [m]")
        plt.legend()
        plt.title("Relative Pose Error (RPE) over time")
        plt.grid(True)
        plt.savefig(os.path.join(self.run_path, f"Run_{self.run_id}_RPE_Over_Time.png"))
        plt.close()

        # Plot state covariance for x, y, and theta separately
        if self.covariances_x and self.covariances_y and self.covariances_theta:
            plt.figure()
            plt.plot(self.covariances_x, label="Variance of x")
            plt.plot(self.covariances_y, label="Variance of y")
            plt.plot(self.covariances_theta, label="Variance of theta")
            plt.xlabel("Time step [s]")
            plt.ylabel("Variance")
            plt.legend()
            plt.title("Variance of x, y, theta over time")
            plt.grid(True)
            plt.savefig(os.path.join(self.run_path, f"Run_{self.run_id}_State_Covariance.png"))
            plt.close()

        # Resource usage plots
        plt.figure()
        plt.plot(self.memory_usage, label="Memory Usage (%)")
        plt.plot(self.cpu_usage, label="CPU Usage (%)")
        plt.xlabel("Time step [s]")
        plt.ylabel("Usage [%]")
        plt.legend()
        plt.title("Memory and CPU Usage over time")
        plt.grid(True)
        plt.savefig(os.path.join(self.run_path, f"Run_{self.run_id}_Resource_Usage.png"))
        plt.close()

        # Log results
        rospy.loginfo(f"ATE: {ate}")
        rospy.loginfo(f"RPE: {rpe}")
        rospy.loginfo(f"RMSE: {rmse}")
        rospy.loginfo(f"NEES: {mean_nees}")
        rospy.loginfo(f"Kolmogorov-Smirnov test (X errors): {ks_x}")
        rospy.loginfo(f"Kolmogorov-Smirnov test (Y errors): {ks_y}")
        rospy.loginfo(f"T-test: {significance_t}")
        rospy.loginfo(f"Wilcoxon signed-rank test: {significance_w}")


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
