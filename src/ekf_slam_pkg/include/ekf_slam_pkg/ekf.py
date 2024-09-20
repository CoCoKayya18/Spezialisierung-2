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

        self.F_x = np.eye(3)
        
        self.process_noise = np.array([[config['process_noise'],0,0],[0, config['process_noise'],0], [0, 0, config['process_noise']]]) 
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
    def predict(self, currentVel, currentPosition, currentCovariance):

        currentVel = np.array(currentVel).reshape(1, -1)  # Ensure it has the shape (1, 3)

        # Scale the input data correctly
        odomVelScaled = self.scaler_X.transform(currentVel)

        # Predict using the loaded model and the standardized data
        y_predict_mean, y_predict_variance = self.model.predict(odomVelScaled)

        predictedDelta = self.scaler_Y.inverse_transform(y_predict_mean)
        predicted_covariance = np.diag(np.full(3, y_predict_variance[0]))

        ' Resize the F_x matrix to adjust for the landmarks too'
        state_vector_size = 3 + 2 * self.num_landmarks
        F_x = np.zeros((state_vector_size, state_vector_size))

        # The top-left 3x3 block is the identity matrix that updates the robot's pose
        F_x[0, 0] = 1  # x -> x
        F_x[1, 1] = 1  # y -> y
        F_x[2, 2] = 1

        predictedDelta = np.array(currentVel).reshape(-1, 1)
        currentPosition[:3] += self.F_x.T @ predictedDelta
        currentCovariance[:3, :3] +=  predicted_covariance + self.F_x.T @ self.process_noise @ self.F_x

        # Update state
        self.state = currentPosition
        self.covariance = currentCovariance       

        # Structured print statement for self.state (currentPosition)
        print("\n=== Updated State Vector (self.state) ===")
        print(f"Shape: {self.state.shape}")
        print(f"State Vector:\n{self.state}")

        # Structured print statement for self.covariance (currentCovariance)
        print("\n=== Updated Covariance Matrix (self.covariance) ===")
        print(f"Shape: {self.covariance.shape}")
        print(f"Covariance Matrix:\n{self.covariance}")

        return self.state, self.covariance

    def correct(self, scanMessage, currentStateVector, currentCovarianceMatrix):

        x = currentStateVector[0].item()
        y = currentStateVector[1].item()
        theta = currentStateVector[2].item()

        # Structured print statement for self.state (currentPosition)
        print("\n=== currentStateVector (currentStateVectorcurrentStateVector) ===")
        print(f"Shape: {currentStateVector.shape}")
        print(f"currentStateVector:\n{currentStateVector}")

        # Structured print statement for self.covariance (currentCovariance)
        print("\n=== currentCovarianceMatrix (currentCovarianceMatrix) ===")
        print(f"Shape: {currentCovarianceMatrix.shape}")
        print(f"currentCovarianceMatrix:\n{currentCovarianceMatrix}")

        # EKF update step
        z_t = self.sensor.extract_features_from_scan(scanMessage, scanMessage.angle_min, scanMessage.angle_max, scanMessage.angle_increment)

        # Lists to store Kalman gains and H_k matrices
        K_list = []
        H_list = []

        for z_i in z_t:
            
            z_i = np.array(z_i)
            
            # Add new Landmark
            newLandmark_x, newLandmark_y = self.map.calculate_landmark_estimates(x, y, theta, z_i)
            new_landmark = np.array([newLandmark_x, newLandmark_y])

            # Create temporary state and covariance matrix with the landmark in it

            tempState = np.pad(self.state, ((0, 2),(0,0)), mode='constant', constant_values=new_landmark)

            # print("\n=== Temp State (with new landmark added) ===")
            # print(f"Shape: {tempState.shape}")
            # print(f"Temp State:\n{tempState}")

            n = currentCovarianceMatrix.shape[0]
            tempCovariance = np.zeros((n + 2, n + 2))
            tempCovariance[:n, :n] = self.covariance
            initial_landmark_uncertainty = 1000
            tempCovariance[n:, n:] = np.array([[initial_landmark_uncertainty, 0],
                                            [0, initial_landmark_uncertainty]])
            
            # print("\n=== Temp Covariance (expanded for new landmark) ===")
            # print(f"Shape: {tempCovariance.shape}")
            # print(f"Temp Covariance:\n{tempCovariance}")

            # Lists to store variables for all landmarks
            F_x_k_list = []
            H_k_list = []
            Psi_k_list = []
            pi_k_list = []

            if self.num_landmarks == 0:
                # Initialize the state and covariance for the first landmark
                # print("Initializing the first landmark")
                self.state = tempState  # Assuming tempState includes the first landmark
                self.covariance = tempCovariance
                self.num_landmarks += 1
                print("\n=== Updated State Vector (self.state) ===")
                print(f"Shape: {self.state.shape}")
                print(f"State Vector:\n{self.state}")

                # Structured print statement for self.covariance (currentCovariance)
                print("\n=== Updated Covariance Matrix (self.covariance) ===")
                print(f"Shape: {self.covariance.shape}")
                print(f"Covariance Matrix:\n{self.covariance}")

            else:
            # Iterate through observed landmarks
                for k in range(1, self.num_landmarks + 1):
                    
                    delta_k = np.array([newLandmark_x - x, newLandmark_y - y])

                    # print("\n=== delta_k (difference between landmark and robot position) ===")
                    # print("delta_k (shape: {}):\n{}".format(delta_k.shape, delta_k))
                    
                    q_k = np.dot(delta_k.T, delta_k).item()

                    # print("\n=== q_k (squared distance between robot and landmark) ===")
                    # print("q_k (value):\n{}".format(q_k))
                    
                    z_hat_k = np.array([np.sqrt(q_k), np.arctan2(delta_k[1].item(), delta_k[0].item()) - theta])

                    # print("\n=== z_hat_k (predicted measurement in range and bearing) ===")
                    # print("z_hat_k (shape: {}):\n{}".format(z_hat_k.shape, z_hat_k))

                    # Compute F_x,k matrix
                    F_x_k = self.map.compute_F_x_k(self.num_landmarks, k)
                    # print("\n=== F_x_k (state transition matrix for this landmark) ===")
                    # print("F_x_k (shape: {}):\n{}".format(F_x_k.shape, F_x_k))

                    # Compute H^k_t matrix
                    H_k_t = self.map.compute_H_k_t(delta_k, q_k, F_x_k)
                    # print("\n=== H_k_t (measurement Jacobian matrix) ===")
                    # print("H_k_t (shape: {}):\n{}".format(H_k_t.shape, H_k_t))

                    # Compute Mahalanobis distance
                    pi_k, Psi_k = self.map.compute_mahalanobis_distance(z_i, z_hat_k, H_k_t, tempCovariance, self.measurement_noise)

                    # print("\n=== pi_k (Mahalanobis distance) ===")
                    # print("pi_k (value):\n{}".format(pi_k))

                    # print("\n=== Psi_k (innovation covariance matrix) ===")
                    # print("Psi_k (shape: {}):\n{}".format(Psi_k.shape, Psi_k))

                    # Store the computed values in lists
                    F_x_k_list.append(F_x_k)
                    H_k_list.append(H_k_t)
                    Psi_k_list.append(Psi_k)
                    pi_k_list.append(pi_k)

            pi_k_list.append(self.alpha)
            j_i = min(pi_k_list)
            best_landmark_index = pi_k_list.index(j_i)

            # print(f"H_k_list size: {len(H_k_list)}")
            # print(f"best_landmark_index: {best_landmark_index}")

            # Check if a new landmark is being added
            if best_landmark_index >= self.num_landmarks:

                # Update state vector to include the new landmark
                self.state = tempState
                self.covariance = tempCovariance

                # Update the number of landmarks
                self.num_landmarks += 1

                H_k_t = H_k_list[best_landmark_index]
                # print(f"H_k_t for best_landmark_index {best_landmark_index}: {H_k_t}")
                # print(f"H_k_t before best_landmark_index {best_landmark_index - 1}: {H_k_t}")

            else:
                # Measurement is associated with an existing landmark
                # Truncate the H matrix to prevent dimension mismatch
                H_k_t = H_k_list[best_landmark_index]
                H_k_t = H_k_t[:, :self.num_landmarks * 2 + 3]  # Truncate H matrix

            # print(f"CovarianceMatrix shape: {self.covariance.shape}")
            # print(f"H_k_t shape: {H_k_t.shape}")
            # print(f"H_k_t.T shape: {H_k_t.T.shape}")
            # print(f"Psi_k_list[best_landmark_index] shape: {Psi_k_list[best_landmark_index].shape}")
            
            # Compute the Kalman gain
            K_t_i = self.covariance @ H_k_t.T @ np.linalg.inv(Psi_k_list[best_landmark_index])

            # Store Kalman gain and H_k_t for final summation later
            K_list.append(K_t_i)
            H_list.append(H_k_t)

            # Final Summation: Apply the cumulative update for both state and covariance
            updateStateSum = np.zeros_like(self.state)
            updateCovarianceSum = np.zeros_like(self.covariance)

            for i, K_t_i in enumerate(K_list):
                # Recompute state vector size before applying updates
                measurement_residual = z_t[i] - (H_list[i] @ self.state[:len(H_list[i][0])])
                updateStateSum += K_t_i @ measurement_residual
                updateCovarianceSum += K_t_i @ H_list[i]

            # Update state mean and covariance
            self.state = self.state + updateStateSum
            self.state[2] = self.utils.wrap_angle(self.state[2])  # Normalize the orientation angle
            self.covariance = self.covariance - updateCovarianceSum @ self.covariance

            # Print updated number of landmarks
            print(f"Landmark Numbers: {self.num_landmarks}")

        return self.state, self.covariance


