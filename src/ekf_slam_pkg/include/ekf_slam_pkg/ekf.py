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
        self.alpha = 5.991 # 95% confidence based on Chi-squared distribution
        
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

        return self.state, self.covariance

    def correct(self, scanMessage, currentStateVector, currentCovarianceMatrix):

        x = currentStateVector[0]
        y = currentStateVector[1]
        theta = currentStateVector[2]

        initial_LM_variance = 1000  # Large initial uncertainty for new landmarks

        # EKF update step
        z_t = self.sensor.extract_features_from_scan(scanMessage, scanMessage.angle_min, scanMessage.angle_max, scanMessage.angle_increment)

        # Lists to store Kalman gains and H_k matrices
        K_list = []
        H_list = []

        # Initialize state and covariance updates
        updateStateSum = np.zeros_like(currentStateVector)
        updateCovarianceSum = np.zeros_like(currentCovarianceMatrix)

        for z_i in z_t:
            
            z_i = np.array(z_i)
            
            # Add new Landmark
            newLandmark_x, newLandmark_y = self.map.add_landmark_estimates(x, y, theta, z_i)

            if self.num_landmarks == 0:
                # If no landmarks exist, treat this as the first new landmark
                print("Initializing first landmark...")
                
                # Add the new landmark to the state vector
                currentStateVector.extend([newLandmark_x, newLandmark_y])

                # Expand the covariance matrix to include the new landmark
                n = currentCovarianceMatrix.shape[0]
                expanded_covariance = np.zeros((n + 2, n + 2))
                expanded_covariance[:n, :n] = currentCovarianceMatrix
                landmark_covariance = np.array([[initial_LM_variance, 0],
                                                [0, initial_LM_variance]])
                expanded_covariance[n:, n:] = landmark_covariance
                currentCovarianceMatrix = expanded_covariance

                updateStateSum = np.zeros_like(currentStateVector)
                updateCovarianceSum = np.zeros_like(currentCovarianceMatrix)

                self.num_landmarks += 1  # Increment the number of landmarks

                # No need for data association, directly update the state and covariance
                delta_k = np.array([newLandmark_x - x, newLandmark_y - y])
                q_k = np.dot(delta_k.T, delta_k)
                z_hat_k = np.array([np.sqrt(q_k), np.arctan2(delta_k[1], delta_k[0]) - theta])

                F_x_k = self.map.compute_F_x_k(self.num_landmarks, self.num_landmarks)
                H_k_t = self.map.compute_H_k_t(delta_k, q_k, F_x_k)
                Psi_k = H_k_t @ currentCovarianceMatrix @ H_k_t.T + self.measurement_noise
                K_i_t = currentCovarianceMatrix @ H_k_t.T @ np.linalg.inv(Psi_k)

                # Store Kalman gain and H_k_t for final summation later
                K_list.append(K_i_t)
                H_list.append(H_k_t)

            else:
                # Lists to store variables for all landmarks
                F_x_k_list = []
                H_k_list = []
                Psi_k_list = []
                pi_k_list = []

                # Iterate through observed landmarks
                for k in range(0, self.num_landmarks):

                    # Get the position of the k-th landmark
                    newLandmark_x = currentStateVector[3 + 2 * k]
                    newLandmark_y = currentStateVector[4 + 2 * k]
                    
                    delta_k = np.array([newLandmark_x - x, newLandmark_y - y])
                    
                    q_k = np.dot(delta_k.T, delta_k)
                    
                    z_hat_k = np.array([np.sqrt(q_k), np.arctan2(delta_k[1], delta_k[0]) - theta])

                    # Compute F_x,k matrix
                    F_x_k = self.map.compute_F_x_k(self.num_landmarks, k)

                    # Compute H^k_t matrix
                    H_k_t = self.map.compute_H_k_t(delta_k, q_k, F_x_k)

                    # Compute Mahalanobis distance
                    pi_k, Psi_k = self.map.compute_mahalanobis_distance(z_i, z_hat_k, H_k_t, currentCovarianceMatrix, self.measurement_noise)

                    # Store the computed values in lists
                    F_x_k_list.append(F_x_k)
                    H_k_list.append(H_k_t)
                    Psi_k_list.append(Psi_k)
                    pi_k_list.append(pi_k)

                # Data association decision: Find the best match by finding the smallest pi_k
                best_pi_k = min(pi_k_list)
                best_landmark_index = pi_k_list.index(best_pi_k)

                if best_pi_k < self.alpha:
                    # Use the best-matching landmark index to get the corresponding matrices
                    best_F_x_k = F_x_k_list[best_landmark_index]
                    best_H_k = H_k_list[best_landmark_index]
                    best_Psi_k = Psi_k_list[best_landmark_index]

                    K_i_t = currentCovarianceMatrix @ best_H_k.T @ np.linalg.inv(best_Psi_k)
                    
                else:

                    # Add the new landmark to the state vector and covariance matrix
                    currentStateVector.extend([newLandmark_x, newLandmark_y])

                    # Expand the covariance matrix to include the new landmark
                    n = currentCovarianceMatrix.shape[0]
                    expanded_covariance = np.zeros((n + 2, n + 2))
                    expanded_covariance[:n, :n] = currentCovarianceMatrix
                    landmark_covariance = np.array([[initial_LM_variance, 0],
                                                    [0, initial_LM_variance]])
                    expanded_covariance[n:, n:] = landmark_covariance
                    currentCovarianceMatrix = expanded_covariance

                    self.num_landmarks += 1

                    # Perform update for new landmark
                    delta_k = np.array([newLandmark_x - x, newLandmark_y - y])
                    q_k = np.dot(delta_k.T, delta_k)
                    z_hat_k = np.array([np.sqrt(q_k), np.arctan2(delta_k[1], delta_k[0]) - theta])

                    # Compute F_x,k and H_k for the Kalman update
                    F_x_k = self.map.compute_F_x_k(self.num_landmarks, self.num_landmarks)
                    H_k_t = self.map.compute_H_k_t(delta_k, q_k, F_x_k)
                    Psi_k = H_k_t @ currentCovarianceMatrix @ H_k_t.T + self.measurement_noise
                    K_i_t = currentCovarianceMatrix @ H_k_t.T @ np.linalg.inv(Psi_k)

                # Store Kalman gain and H_k_t for final summation later
                K_list.append(K_i_t)
                H_list.append(H_k_t)

        # Final Summation: Apply the cumulative update
        for i, K_i_t in enumerate(K_list):
            # Recompute state vector size before applying updates
            updateStateSum += K_i_t @ (z_t[i] - (H_list[i] @ currentStateVector[:len(H_list[i][0])]))  # Match dimensions
            updateCovarianceSum += K_i_t @ H_list[i]

        # Update state mean and covariance
        self.state = self.state + updateStateSum
        self.covariance = currentCovarianceMatrix - updateCovarianceSum @ currentCovarianceMatrix

        print(f"Landmark Numbers: {self.num_landmarks}")

        return self.state, self.covariance



