import numpy as np
import rospy
import pickle
import rospy
import time
import sys
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
        self.state = np.eye(3)
        self.alpha = 5

        self.F_x = np.eye(3)
        
        self.process_noise = np.array([[np.power(config['process_noise'],2),0,0],[0, np.power(config['process_noise'],2),0], [0, 0, np.power(config['process_noise'],2)]]) 
        self.measurement_noise = np.array([[np.power(config['measurement_noise'],2),0],[0, np.power(config['measurement_noise'],2)]])

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
    def predict(self, currentVel, currentPosition, currentCovariance, num_landmarks):

        # with self.lock:

        # rospy.loginfo("\n === currentPosition before prediction(self.state) ===")
        # rospy.loginfo(f"\n Shape: {currentPosition.shape}")
        # rospy.loginfo(f"\n currentPosition:\n{currentPosition}")
        
        # rospy.loginfo("\n=== currentCovariance Matrix before prediction(self.covariance) ===")
        # rospy.loginfo(f"\n Shape: {currentCovariance.shape}")
        # rospy.loginfo(f"\n currentCovariance Matrix:\n{currentCovariance}")

        self.num_landmarks = num_landmarks

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

        # Extend the process noise and the prediction to the actuall state and variance

        R_t_ext = np.zeros((state_vector_size, state_vector_size))
        R_t_ext[:3, :3] = self.process_noise

        predictedDelta_ext = np.zeros((state_vector_size, 1))
        predictedDelta_ext[:3] = predictedDelta.reshape(3, 1)

        # rospy.loginfo(f"\n Predicted_Delta_ext Shape: {predictedDelta_ext.shape}")
        # rospy.loginfo(f"Predicted_Delta_ext: {predictedDelta_ext}")

        predicted_covariance_ext = np.zeros((state_vector_size, state_vector_size))
        predicted_covariance_ext[:3, :3] = predicted_covariance

        # rospy.loginfo(f"\n F_x Shape: {F_x.shape}")
        # rospy.loginfo(f"\n F_x Matrix: {F_x}")

        # rospy.loginfo(f"\n R_t_ext Shape: {R_t_ext.shape}")
        # rospy.loginfo(f"\n R_t_ext Matrix: {R_t_ext}")

        # rospy.loginfo(f"\n Predicted Covariance ext Shape: {predicted_covariance_ext.shape}")
        # rospy.loginfo(f"\n predicted_covariance ext Matrix: {predicted_covariance_ext}")

        currentPosition += F_x.T @ predictedDelta_ext
        currentCovariance +=  predicted_covariance_ext + F_x.T @ R_t_ext @ F_x

        # Update state
        self.state = currentPosition
        self.covariance = currentCovariance    

        # if np.any(self.covariance < 0):
        #     rospy.logerr("Negative value detected in covariance matrix in prediction step!")
        #     rospy.logerr(f"Covariance Matrix:\n{self.covariance}")
        #     # Shut down the program with an error message
        #     sys.exit("Shutting down program due to negative covariance value.")   

        # rospy.loginfo("\n === State Vector after prediction(self.state) ===")
        # rospy.loginfo(f"\n Shape: {self.state.shape}")
        # rospy.loginfo(f"\n State Vector:\n{self.state}")
        
        # rospy.loginfo("\n=== Covariance Matrix after prediction(self.covariance) ===")
        # rospy.loginfo(f"\n Shape: {self.covariance.shape}")
        # rospy.loginfo(f"\n Covariance Matrix:\n{self.covariance}")

        # rospy.loginfo("\n === PREDICTION FINISHED ====== PREDICTION FINISHED ======")

        return self.state, self.covariance


    def correct(self, scanMessage, currentStateVector, currentCovarianceMatrix):

        rospy.loginfo("\n === CORRECTION BEGINNING ====== CORRECTION BEGINNING ======")

        start_time = time.time()

        # with self.lock:

        self.state = currentStateVector
        self.covariance = currentCovarianceMatrix

        # if np.any(self.covariance < 0):
        #     rospy.logerr("Negative value detected in covariance matrix in correction step!")
        #     rospy.logerr(f"Covariance Matrix:\n{self.covariance}")
        #     # Shut down the program with an error message
        #     sys.exit("Shutting down program due to negative covariance value.")

        # rospy.loginfo("\n === State Vector before correction(self.state) ===")
        # rospy.loginfo(f"\n Shape: {self.state.shape}")
        # rospy.loginfo(f"\n State Vector:\n{self.state}")
        
        # rospy.loginfo("\n === Covariance Matrix before correction(self.covariance) ===")
        # rospy.loginfo(f"\n Shape: {self.covariance.shape}")
        # rospy.loginfo(f"\n Covariance Matrix:\n{self.covariance}")
        
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

        # rospy.loginfo(f"Observations: {z_t}")

        for z_i in z_t:

            observation_counter += 1

            # rospy.loginfo("\n === State Vector in obs loop (self.state) ===")
            # rospy.loginfo(f"\n Shape: {self.state.shape}")
            # rospy.loginfo(f"\n State Vector:\n{self.state}")
            
            # rospy.loginfo("\n === Covariance Matrix in obs loop (self.covariance) ===")
            # rospy.loginfo(f"\n Shape: {self.covariance.shape}")
            # rospy.loginfo(f"\n Covariance Matrix:\n{self.covariance}")

            # intialize new landmark and create tempoprary state and covariance matrices
            newLandmark_x, newLandmark_y = self.map.calculate_landmark_estimates(x, y, theta, z_i)
            new_landmark = np.array([newLandmark_x, newLandmark_y])

            # Create temporary state and covariance matrix with the landmark in it

            # tempState = np.pad(self.state, ((0, 2),(0,0)), mode='constant', constant_values=new_landmark)
            tempState = np.vstack((self.state, new_landmark.reshape(2, 1)))

            n = self.covariance.shape[0]

            tempCovariance = np.zeros((n + 2, n + 2))
            tempCovariance[:n, :n] = self.covariance
            initial_landmark_uncertainty = 1000
            tempCovariance[n:, n:] = np.array([[initial_landmark_uncertainty, 0],
                                            [0, initial_landmark_uncertainty]])

            # Temporarily adjust landmark number too
            temp_num_landmarks = self.num_landmarks + 1

            # rospy.loginfo(f"\n Current obs loop: {observation_counter}")

            # rospy.loginfo("\n === Current Loop Variables ===")
            # rospy.loginfo(f"\n Current number of landmarks: {self.num_landmarks}")
            # rospy.loginfo(f"\n Temporary number of landmarks: {temp_num_landmarks}")

            # # rospy.loginfo the covariance matrix before expanding
            # rospy.loginfo("\nv=== Current Covariance Matrix (Before Expanding for New Landmark) ===")
            # rospy.loginfo(f"\n Shape: {currentCovarianceMatrix.shape}")
            # rospy.loginfo(f"\n Covariance Matrix:\n{currentCovarianceMatrix}")

            # # rospy.loginfo the expanded covariance matrix and state
            # rospy.loginfo("\n === Temporary State Vector (Expanded for New Landmark) ===")
            # rospy.loginfo(f"\n Shape: {tempState.shape}")
            # rospy.loginfo(f"\n Temp State :\n{tempState}")

            # rospy.loginfo("\n === Temporary Covariance Matrix (Expanded for New Landmark) ===")
            # rospy.loginfo(f"\n Shape: {tempCovariance.shape}")
            # rospy.loginfo(f"\n Temp Covariance Matrix:\n{tempCovariance}")


            # # rospy.loginfo the uncertainty initialization for the new landmark
            # rospy.loginfo("\n === Initial Uncertainty for New Landmark ===")
            # rospy.loginfo(f"\n Initial uncertainty value: {initial_landmark_uncertainty}")
            # rospy.loginfo(f"\n New landmark uncertainty block:\n{tempCovariance[n:, n:]}")


            landmark_counter = 0

            H_matrix_list = []
            psi_list = []
            pi_list = []
            z_hat_list = []

            # Start landmark loop
            for k in range(1, temp_num_landmarks + 1):

                landmark_counter += 1

                # rospy.loginfo(f"\n Current LM loop: {landmark_counter}")

                # rospy.loginfo("\n=== State Vector in LM loop (self.state) ===")
                # rospy.loginfo(f"\n Shape: {self.state.shape}")
                # rospy.loginfo(f"\n State Vector:\n{self.state}")
                
                # rospy.loginfo("\n=== Covariance Matrix in LM loop (self.covariance) ===")
                # rospy.loginfo(f"\n Shape: {self.covariance.shape}")
                # rospy.loginfo(f"\n Covariance Matrix:\n{self.covariance}")
                
                delta_k = np.array([tempState[2+k] - x, tempState[2+k+1] - y])
                
                q_k = np.dot(delta_k.T, delta_k).item()
                
                z_hat_k = np.array([np.sqrt(q_k), np.arctan2(delta_k[1].item(), delta_k[0].item()) - theta])

                # rospy.loginfo("\n=== State Vector in LM loop after z_hat_k (self.state) ===")
                # rospy.loginfo(f"\n Shape: {self.state.shape}")
                # rospy.loginfo(f"\n State Vector:\n{self.state}")
                
                # rospy.loginfo("\n=== Covariance Matrix in LM loop after z_hat_k (self.covariance) ===")
                # rospy.loginfo(f"\n Shape: {self.covariance.shape}")
                # rospy.loginfo(f"\n Covariance Matrix:\n{self.covariance}")

                # Compute F_x,k matrix
                F_x_k = self.map.compute_F_x_k(temp_num_landmarks, k)
                
                # rospy.loginfo F_x_k matrix for the current iteration
                # rospy.loginfo(f"\n=== F_x_k Matrix (for Observation {observation_counter}, Landmark {landmark_counter}) ===")
                # rospy.loginfo(f"\n Shape: {F_x_k.shape}")
                # rospy.loginfo(f"\n F_x_k:\n{F_x_k}")

                # # rospy.loginfo current covariance (self.covariance)
                # rospy.loginfo(f"\n=== Covariance Matrix (currentCovarianceMatrix) Before Kalman Gain (for Observation {observation_counter}, Landmark {landmark_counter}) ===")
                # rospy.loginfo(f"\n Shape: {currentCovarianceMatrix.shape}")
                # rospy.loginfo(f"\n Covariance Matrix:\n{currentCovarianceMatrix}")

                # Compute H^k_t matrix
                H_k_t = self.map.compute_H_k_t(delta_k, q_k, F_x_k)

                # rospy.loginfo("\n=== State Vector in LM loop after H_k_t (self.state) ===")
                # rospy.loginfo(f"\n Shape: {self.state.shape}")
                # rospy.loginfo(f"\n State Vector:\n{self.state}")
                
                # rospy.loginfo("\n=== Covariance Matrix in LM loop after H_k_t (self.covariance) ===")
                # rospy.loginfo(f"\n Shape: {self.covariance.shape}")
                # rospy.loginfo(f"\n Covariance Matrix:\n{self.covariance}")
                
                # # rospy.loginfo H_k_t matrix for the current iteration
                # rospy.loginfo(f"\n=== H_k_t Matrix (for Observation {observation_counter}, Landmark {landmark_counter}) ===")
                # rospy.loginfo(f"\n Shape: {H_k_t.shape}")
                # rospy.loginfo(f"\n H_k_t:\n{H_k_t}")

                # Compute Mahalanobis distance
                pi_k, Psi_k = self.map.compute_mahalanobis_distance(z_i, z_hat_k, H_k_t, tempCovariance, self.measurement_noise)

                # Create plots for H_Matrix jacobian and Covariance matrix
                # self.utils.save_covariance_matrix_plot(self.covariance, observation_counter, landmark_counter)
                # self.utils.save_jacobian_plot(H_k_t, observation_counter, landmark_counter)

                # Add the calculated values to the list
                H_matrix_list.append(H_k_t)
                psi_list.append(Psi_k)
                pi_list.append(pi_k)
                z_hat_list.append(z_hat_k)

                # rospy.loginfo("\n === State Vector after LM loop (self.state) ===")
                # rospy.loginfo(f"\n Shape: {self.state.shape}")
                # rospy.loginfo(f"\n State Vector:\n{self.state}")
                
                # rospy.loginfo("\n === Covariance Matrix after LM loop (self.covariance) ===")
                # rospy.loginfo(f"\n Shape: {self.covariance.shape}")
                # rospy.loginfo(f"\n Covariance Matrix:\n{self.covariance}")
            
            # End Landmark Loop

            # Set the added landmarks pi to the alpha value by hand
            pi_list[-1] = self.alpha

            j_i = min(pi_list)
            best_landmark_index = pi_list.index(j_i)

            # rospy.loginfo(f"\n best landmark index: {best_landmark_index}")

            # Check if a new landmark is being added
            if best_landmark_index >= self.num_landmarks:

                rospy.loginfo(f"\n ADDING NEW LANDMARK at obs {observation_counter}, landmark {landmark_counter}")

                # Update state vector to include the new landmark
                self.state = tempState
                self.covariance = tempCovariance

                # Update the number of landmarks
                self.num_landmarks = temp_num_landmarks

                best_H_matrix = H_matrix_list[best_landmark_index]
                best_z_hat = z_hat_list[best_landmark_index]
                
                # rospy.loginfo("\n=== State Vector after landmark addition(self.state) ===")
                # rospy.loginfo(f"\n Shape: {self.state.shape}")
                # rospy.loginfo(f"\n State Vector:\n{self.state}")
                
                # rospy.loginfo("\n=== Covariance Matrix after landmark addition(self.covariance) ===")
                # rospy.loginfo(f"\n Shape: {self.covariance.shape}")
                # rospy.loginfo(f"\n Covariance Matrix:\n{self.covariance}")

                # # rospy.loginfo H_k_t matrix for the current calculation
                # rospy.loginfo(f"\n=== best_H_matrix Matrix after landmark addition ===")
                # rospy.loginfo(f"\n Shape: {best_H_matrix.shape}")
                # rospy.loginfo(f"\n best_H_matrix:\n{best_H_matrix}")

            else:
                # Measurement is associated with an existing landmark
                # Truncate the H matrix to prevent dimension mismatch

                best_H_matrix = H_matrix_list[best_landmark_index]
                best_H_matrix = best_H_matrix[:, :self.num_landmarks * 2 + 3]  # Truncate H matrix
                best_z_hat = z_hat_list[best_landmark_index]

                temp_num_landmarks -= 1
                
                # rospy.loginfo("\n === Running the else and reducing the states ===")

                # rospy.loginfo("\n=== State Vector after landmark subtraction(self.state) ===")
                # rospy.loginfo(f"\n Shape: {self.state.shape}")
                # rospy.loginfo(f"\n State Vector:\n{self.state}")
                
                # rospy.loginfo("\n=== Covariance Matrix after landmark subtraction(self.covariance) ===")
                # rospy.loginfo(f"\n Shape: {self.covariance.shape}")
                # rospy.loginfo(f"\n Covariance Matrix:\n{self.covariance}")

                # # rospy.loginfo H_k_t matrix for the current calculation
                # rospy.loginfo(f"\n=== best_H_matrix Matrix after landmark subtraction ===")
                # rospy.loginfo(f"\n Shape: {best_H_matrix.shape}")
                # rospy.loginfo(f"\n best_H_matrix:\n{best_H_matrix}")
            
            # Add both variables to the list for later
            best_H_Matrix_list.append(best_H_matrix)
            best_z_hat_list.append(best_z_hat)

            # rospy.loginfo("\n=== State Vector before Kalman Gain Calculation(self.state) ===")
            # rospy.loginfo(f"\n Shape: {self.state.shape}")
            # rospy.loginfo(f"\n State Vector:\n{self.state}")
            
            rospy.loginfo("\n=== Covariance Matrix before Kalman Gain Calculation(self.covariance) ===")
            # rospy.loginfo(f"\n Shape: {self.covariance.shape}")
            rospy.loginfo(f"\n Covariance Matrix:\n{self.covariance}")

            # rospy.loginfo H_k_t matrix for the current calculation
            rospy.loginfo(f"\n=== best_H_matrix Matrix before Kalman Gain Calculation ===")
            # rospy.loginfo(f"\n Shape: {best_H_matrix.shape}")
            rospy.loginfo(f"\n best_H_matrix:\n{best_H_matrix}")

            rospy.loginfo(f"\n=== inverse psi_list[best_landmark_index] Matrix before Kalman Gain Calculation ===")
            # rospy.loginfo(f"\n Shape: {psi_list[best_landmark_index].shape}")
            rospy.loginfo(f"\n psi_list[best_landmark_index]:\n{np.linalg.inv(psi_list[best_landmark_index])}")
                
            # Calculte Kalman gain and att it to the list
            Kalman_gain = self.covariance @ best_H_matrix.T @ np.linalg.inv(psi_list[best_landmark_index])

            rospy.loginfo(f"Resulting Kalman Gain: {Kalman_gain}")

            kalman_gain_list.append(Kalman_gain)
        
        # end of observation loop   

        # rospy.loginfo(f"Kalman_gain List: {kalman_gain_list}")
        
        updateStateSum = np.zeros_like(self.state)
        updateCovarianceSum = np.zeros_like(self.covariance)

        # rospy.loginfo(f"\n UpdateStateSum shape before update: {updateStateSum.shape}")
        # rospy.loginfo(f"\n UpdateStateSum dim before update: {updateStateSum.ndim}")
        # rospy.loginfo(f"\n UpdateStateSum before update:\n{updateStateSum}")

        for i, K_t_i in enumerate(kalman_gain_list):

            measurement_residual = z_t[i] - best_z_hat_list[i]

            measurement_residual = measurement_residual.reshape(-1, 1)

            # Expand Kalman gain to match the full state size (9 in your case)
            # K_t_i_ext = np.zeros((self.state.shape[0], measurement_residual.shape[0]))
            # K_t_i_ext[:K_t_i.shape[0], :] = K_t_i  # Copy original Kalman gain into the top-left

            # H_t_i_ext = np.zeros((best_H_Matrix_list[i].shape[0], K_t_i_ext.shape[0]))

            # rospy.loginfo(f"\nIteration in summation{i}:")

            # rospy.loginfo(f"\n K_t_i_ext shape: {K_t_i_ext.shape}")
            # rospy.loginfo(f"\n K_t_i_ext dim: {K_t_i_ext.ndim}")
            # rospy.loginfo(f"\n K_t_i_ext:\n{K_t_i_ext}")

            # rospy.loginfo(f"\n K_t_i shape: {K_t_i.shape}")
            # rospy.loginfo(f"\n K_t_i dim: {K_t_i.ndim}")
            # rospy.loginfo(f"\n K_t_i:\n{K_t_i}")
            
            # rospy.loginfo(f"\n Measurement Residual shape: {measurement_residual.shape}")
            # rospy.loginfo(f"\n Measurement Dimension: {measurement_residual.ndim}")
            # rospy.loginfo(f"\n Measurement Residual:\n{measurement_residual}")

            # rospy.loginfo(f"\n H_t_i_ext shape before update: {H_t_i_ext.shape}")
            # rospy.loginfo(f"\n H_t_i_ext dim before update: {H_t_i_ext.ndim}")
            # rospy.loginfo(f"\n H_t_i_ext before update:\n{H_t_i_ext}")

            # rospy.loginfo(f"\n updateCovarianceSum shape before update: {updateCovarianceSum.shape}")
            # rospy.loginfo(f"\n updateCovarianceSum dim before update: {updateCovarianceSum.ndim}")
            # rospy.loginfo(f"\n updateCovarianceSum before update:\n{updateCovarianceSum}")

            state_update = K_t_i @ measurement_residual

            covariance_update = K_t_i @ best_H_Matrix_list[i]

            state_indices = slice(0, K_t_i.shape[0])

            updateStateSum[state_indices] += state_update

            updateCovarianceSum[state_indices, state_indices] += covariance_update

            # updateStateSum += K_t_i_ext @ measurement_residual
            # updateCovarianceSum += K_t_i_ext @ H_t_i_ext

        rospy.loginfo(f"\n UpdateStateSum shape before update: {updateStateSum.shape}")
        # rospy.loginfo(f"\n UpdateStateSum dim before update: {updateStateSum.ndim}")
        rospy.loginfo(f"\n UpdateStateSum before update:\n{updateStateSum}")

        rospy.loginfo(f"\n updateCovarianceSum shape before update: {updateCovarianceSum.shape}")
        # rospy.loginfo(f"\n updateCovarianceSum dim before update: {updateCovarianceSum.ndim}")
        rospy.loginfo(f"\n updateCovarianceSum before update:\n{updateCovarianceSum}")

        self.state += updateStateSum
        self.state[2] = self.utils.wrap_angle(self.state[2])  # Normalize the orientation angle
        self.covariance = (np.eye(updateCovarianceSum.shape[0]) - updateCovarianceSum) @ self.covariance

        if np.any(self.covariance < 0):
            rospy.logerr("Negative value detected in covariance matrix in correction step at the end!")
            rospy.logerr(f"Covariance Matrix:\n{self.covariance}")
            # Shut down the program with an error message
            sys.exit("Shutting down program due to negative covariance value.")

        # rospy.loginfo("\n=== State Vector after correction(self.state) ===")
        # rospy.loginfo(f"\n Shape: {self.state.shape}")
        # rospy.loginfo(f"\n State Vector:\n{self.state}")
        
        # rospy.loginfo("\n=== Covariance Matrix after correction(self.covariance) ===")
        # rospy.loginfo(f"\n Shape: {self.covariance.shape}")
        # rospy.loginfo(f"\n Covariance Matrix:\n{self.covariance}")

        rospy.loginfo("\n === CORRECTION FINISHED ====== CORRECTION FINISHED ======")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n Correction function took {elapsed_time:.6f} seconds.")

        return self.state, self.covariance, self.num_landmarks
        

