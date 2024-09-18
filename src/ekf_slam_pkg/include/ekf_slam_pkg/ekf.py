import numpy as np
import pickle
import rospy
from geometry_msgs.msg import Pose
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_from_euler


class EKFSLAM:
    def __init__(self, robot, sensor, map, config, utils):
        self.robot = robot
        self.sensor = sensor
        self.map = map
        self.config = config
        self.utils = utils

        self.covariance = np.eye(3)
        self.num_landmarks = 0 
        self.state = []
        
        self.process_noise = config['process_noise']
        self.measurement_noise = np.array([[config['measurement_noise'],0],[0, config['measurement_noise']]])

        model_path = "../Spezialisierung-2/src/ekf_slam_pkg/myMLModel/sparse_gpy_model_random3_odometry.pkl"
        scalerX_path = "../Spezialisierung-2/src/ekf_slam_pkg/Scaler/sparse_scaler_X_random3_odometry.pkl"
        scalerY_path = "../Spezialisierung-2/src/ekf_slam_pkg/Scaler/sparse_scaler_Y_random3_odometry.pkl"
        
        # Load the scalers
        with open(scalerX_path, 'rb') as file:
            self.scaler_X = pickle.load(file)
        with open(scalerY_path, 'rb') as file:
            self.scaler_Y = pickle.load(file)

        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

        rospy.loginfo("EKF Class initialized")
        

    # EKF prediction step
    def predict(self, currentVel, currentPosition):

        currentVel = np.array(currentVel).reshape(1, -1)  # Ensure it has the shape (1, 3)

        # Scale the input data correctly
        odomVelScaled = self.scaler_X.transform(currentVel)

        # Predict using the loaded model and the standardized data
        y_predict_mean, y_predict_variance = self.model.predict(odomVelScaled)

        predictedDelta = self.scaler_Y.inverse_transform(y_predict_mean)
        predicted_covariance = predicted_covariance = np.diag(np.full(3, y_predict_variance[0]))

        currentPosition[0] += predictedDelta[0][0]  # Add the first delta to x
        currentPosition[1] += predictedDelta[0][1]  # Add the second delta to y
        currentPosition[2] += predictedDelta[0][2]  # Add the third delta to theta

        # Update state
        self.state = currentPosition
        self.covariance = predicted_covariance

        # print("\n=== Predicted Covariance ===")
        # print(f"predicted_covariance shape: {y_predict_variance.shape}\n{y_predict_variance}")

        # print("\n=== Converted Covariance ===")
        # print(f"predicted_covariance shape: {predicted_covariance.shape}\n{predicted_covariance}")
        
        # print("\n=== Self Covariance (after assignment) ===")
        # print(f"self.covariance shape: {self.covariance.shape}\n{self.covariance}\n")        

        return self.state, self.covariance

    def correct(self, scanMessage, currentStateVector, currentCovarianceMatrix):

        # Extract current pose information
        x = currentStateVector[0]
        y = currentStateVector[1]
        theta = currentStateVector[2]
        
        # EKF update step
        z_t = self.sensor.extract_features_from_scan(scanMessage, scanMessage.angle_min, scanMessage.angle_max, scanMessage.angle_increment)  # Extract features from LaserScan

        for z_i in z_t:
            
            z_i = np.array(z_i)

            # Add new Landmark
            newLandmark_x, newLandmark_y = self.map.add_landmark_estimates(x, y, theta, z_i)

            currentStateVector.append(newLandmark_x)
            currentStateVector.append(newLandmark_y)

            # rospy.loginfo(f"Correction State Vector: {currentStateVector}")

            # Iterate through observed landmarks
            for k in range(0, self.num_landmarks + 1):
                
                delta_k = np.array([newLandmark_x - x, newLandmark_y - y])
                
                q_k = np.dot(delta_k.T, delta_k)
                
                z_hat_k = np.array([np.sqrt(q_k), np.arctan2(delta_k[1], delta_k[0]) - theta])

                # Compute F_x,k matrix
                F_x_k = self.map.compute_F_x_k(self.num_landmarks, k)

                # Compute H^k_t matrix
                H_k_t = self.map.compute_H_k_t(delta_k, q_k, F_x_k)

                # Compute Mahalanobis distance
                pi_k, Psi_k = self.map.compute_mahalanobis_distance(z_i, z_hat_k, H_k_t, currentCovarianceMatrix, self.measurement_noise)


            # Data association step
            correctLandmarkIndex = self.map.data_association(pi_k)

            if correctLandmarkIndex is not None:
                # Update landmark index
                self.num_landmarks = max(self.num_landmarks, correctLandmarkIndex)

                # Kalman gain
                K_i_t = self.covariance @ H_k_t.T @ np.linalg.inv(Psi_k)

                # Update state mean and covariance using MapHandler
                x, y, theta = self.map.update_state(x, y, theta, z_i, z_hat_k, K_i_t)
                self.covariance = (np.eye(len(self.covariance)) - K_i_t @ H_k_t) @ self.covariance


        # # Update the robot's pose based on the corrected state estimate
        # self.state = self.utils.update_pose_from_state(currentStateVector, x, y, theta)

        # rospy.loginfo("EKF correction step completed.")

        return self.state, self.covariance

