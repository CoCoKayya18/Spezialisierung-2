import os
import pickle

scalerX_path = "/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/Scaler/sparse_scaler_X_random3_odometry.pkl"

with open(scalerX_path, 'rb') as file:
    scaler_X = pickle.load(file)