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

        # print("\n=== State Vector after prediction(self.state) ===")
        # print(f"Shape: {self.state.shape}")
        # print(f"State Vector:\n{self.state}")
        
        # print("\n=== Covariance Matrix after prediction(self.covariance) ===")
        # print(f"Shape: {self.covariance.shape}")
        # print(f"Covariance Matrix:\n{self.covariance}")

        return self.state, self.covariance


    def correct(self, scanMessage, currentStateVector, currentCovarianceMatrix):

        self.state = currentStateVector
        self.covariance = currentCovarianceMatrix
        
        x = self.state[0].item()
        y = self.state[1].item()
        theta = self.state[2].item()

        # Feature Extraction Step
        z_t = self.sensor.extract_features_from_scan(scanMessage, scanMessage.angle_min, scanMessage.angle_max, scanMessage.angle_increment)
            
        # Start observation loop

        observation_counter = 0

        kalman_gain_list = []
        best_z_hat_list = []
        best_H_Matrix_list = []

        for z_i in z_t:

            observation_counter += 1

            # intialize new landmark and create tempoprary state and covariance matrices
            newLandmark_x, newLandmark_y = self.map.calculate_landmark_estimates(x, y, theta, z_i)
            new_landmark = np.array([newLandmark_x, newLandmark_y])

            print(f"new_landmark coordinate X: {newLandmark_x}")
            print(f"new_landmark coordinate Y: {newLandmark_y}")
            print(f"new_landmark coordinates: {new_landmark}")

            # Create temporary state and covariance matrix with the landmark in it

            tempState = np.pad(self.state, ((0, 2),(0,0)), mode='constant', constant_values=new_landmark)

            n = self.covariance.shape[0]

            tempCovariance = np.zeros((n + 2, n + 2))
            tempCovariance[:n, :n] = self.covariance
            initial_landmark_uncertainty = 1000
            tempCovariance[n:, n:] = np.array([[initial_landmark_uncertainty, 0],
                                            [0, initial_landmark_uncertainty]])

            # Temporarily adjust landmark number too
            temp_num_landmarks = self.num_landmarks + 1

            # print(f"\n Current obs loop: {observation_counter}")

            # print("\n--- Current Loop Variables ---")
            # print(f"Current number of landmarks: {self.num_landmarks}")
            # print(f"Temporary number of landmarks: {temp_num_landmarks}")

            # # Print the covariance matrix before expanding
            # print("\n=== Current Covariance Matrix (Before Expanding for New Landmark) ===")
            # print(f"Shape: {currentCovarianceMatrix.shape}")
            # print(f"Covariance Matrix:\n{currentCovarianceMatrix}")

            # Print the expanded covariance matrix
            # print("\n=== Temporary Covariance Matrix (Expanded for New Landmark) ===")
            # print(f"Shape: {tempCovariance.shape}")
            # print(f"Temp Covariance Matrix:\n{tempCovariance}")

            # # Print the uncertainty initialization for the new landmark
            # print("\n=== Initial Uncertainty for New Landmark ===")
            # print(f"Initial uncertainty value: {initial_landmark_uncertainty}")
            # print(f"New landmark uncertainty block:\n{tempCovariance[n:, n:]}")

            # Start landmark loop

            landmark_counter = 0

            H_matrix_list = []
            psi_list = []
            pi_list = []
            z_hat_list = []

            for k in range(1, temp_num_landmarks + 1):

                landmark_counter += 1
                
                delta_k = np.array([newLandmark_x - x, newLandmark_y - y])
                
                q_k = np.dot(delta_k.T, delta_k).item()
                
                z_hat_k = np.array([np.sqrt(q_k), np.arctan2(delta_k[1].item(), delta_k[0].item()) - theta])

                # Compute F_x,k matrix
                F_x_k = self.map.compute_F_x_k(temp_num_landmarks, k)
                # Print F_x_k matrix for the current iteration
                # print(f"\n=== F_x_k Matrix (for Observation {obs_counter}, Landmark {lm_counter}) ===")
                # print(f"Shape: {F_x_k.shape}")
                # print(f"F_x_k:\n{F_x_k}")

                # Print current covariance (self.covariance)
                # print(f"\n=== Covariance Matrix (currentCovarianceMatrix) Before Kalman Gain (for Observation {observation_counter}, Landmark {landmark_counter}) ===")
                # print(f"Shape: {currentCovarianceMatrix.shape}")
                # print(f"Covariance Matrix:\n{currentCovarianceMatrix}")

                # Compute H^k_t matrix
                H_k_t = self.map.compute_H_k_t(delta_k, q_k, F_x_k)
                
                # Print H_k_t matrix for the current iteration
                # print(f"\n=== H_k_t Matrix (for Observation {obs_counter}, Landmark {lm_counter}) ===")
                # print(f"Shape: {H_k_t.shape}")
                # print(f"H_k_t:\n{H_k_t}")

                # Compute Mahalanobis distance
                pi_k, Psi_k = self.map.compute_mahalanobis_distance(z_i, z_hat_k, H_k_t, tempCovariance, self.measurement_noise)

                # Add the calculated values to the list
                H_matrix_list.append(H_k_t)
                psi_list.append(Psi_k)
                pi_list.append(pi_k)
                z_hat_list.append(z_hat_k)
            
            # End Landmark Loop

            # Set the added landmarks pi to the alpha value by hand
            pi_list[-1] = self.alpha

            j_i = min(pi_list)
            best_landmark_index = pi_list.index(j_i)

            # print(f"best landmark index: {best_landmark_index}")

            # Check if a new landmark is being added
            if best_landmark_index >= self.num_landmarks:

                print("\n ADDING NEW LANDMARK")

                # Update state vector to include the new landmark
                self.state = tempState
                self.covariance = tempCovariance

                # Update the number of landmarks
                self.num_landmarks = temp_num_landmarks

                best_H_matrix = H_matrix_list[best_landmark_index]
                best_z_hat = z_hat_list[best_landmark_index]

            else:
                # Measurement is associated with an existing landmark
                # Truncate the H matrix to prevent dimension mismatch
                best_H_matrix = H_matrix_list[best_landmark_index]
                best_H_matrix = best_H_matrix[:, :self.num_landmarks * 2 + 3]  # Truncate H matrix
                best_z_hat = z_hat_list[best_landmark_index]

                temp_num_landmarks -= 1
            
            # Add both variables to the list for later
            best_H_Matrix_list.append(best_H_matrix)
            best_z_hat_list.append(best_z_hat)
                
            # Calculte Kalman gain and att it to the list
            Kalman_gain = self.covariance @ best_H_matrix.T @ np.linalg.inv(psi_list[best_landmark_index])
            kalman_gain_list.append(Kalman_gain)
        
        # end of observation loop    
        
        print("\n=== State Vector after correction(self.state) ===")
        print(f"Shape: {self.state.shape}")
        print(f"State Vector:\n{self.state}")
        
        print("\n=== Covariance Matrix after correction(self.covariance) ===")
        print(f"Shape: {self.covariance.shape}")
        print(f"Covariance Matrix:\n{self.covariance}")
        
        updateStateSum = np.zeros_like(self.state)
        updateCovarianceSum = np.zeros_like(self.covariance)

        for i, K_t_i in enumerate(kalman_gain_list):

            measurement_residual = z_t[i] - best_z_hat_list[i]

            measurement_residual = measurement_residual.reshape(-1, 1)

            # print(f"\nIteration in summation{i}:")

            # print(f"K_t_i shape: {K_t_i.shape}")
            # print(f"K_t_i dim: {K_t_i.ndim}")
            # print(f"K_t_i:\n{K_t_i}")
            
            # print(f"Measurement Residual shape: {measurement_residual.shape}")
            # print(f"Measurement Dimension: {measurement_residual.ndim}")
            # print(f"Measurement Residual:\n{measurement_residual}")

            # print(f"UpdateStateSum shape before update: {updateStateSum.shape}")
            # print(f"UpdateStateSum shape before update: {updateStateSum.ndim}")
            # print(f"UpdateStateSum before update:\n{updateStateSum}")

            updateStateSum += K_t_i @ measurement_residual
            updateCovarianceSum += K_t_i @ best_H_Matrix_list[i]

        self.state += updateStateSum
        self.state[2] = self.utils.wrap_angle(self.state[2])  # Normalize the orientation angle
        self.covariance = (np.eye(updateCovarianceSum.shape[0]) - updateCovarianceSum) @ self.covariance

        return self.state, self.covariance, self.num_landmarks
    

