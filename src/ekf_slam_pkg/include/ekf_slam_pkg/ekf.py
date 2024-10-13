import numpy as np
import rospy
import pickle
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
        self.alpha = 0.5

        self.F_x = np.eye(3)
        
        self.process_noise = np.array([[np.power(config['process_noise'],2),0,0],[0, np.power(config['process_noise'],2),0], [0, 0, np.power(config['process_noise'],2)]]) 
        self.measurement_noise = np.array([[np.power(config['measurement_noise'],2),0],[0, np.power(config['measurement_noise'],2)]])

        self.correctionCounter = 1

        model_path = "/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/myMLModel/sparse_gpy_model_random3_odometry.pkl"
        scalerX_path = "/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/Scaler/sparse_scaler_X_random3_odometry.pkl"
        scalerY_path = "/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/Scaler/sparse_scaler_Y_random3_odometry.pkl"
        
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

        self.state = currentPosition
        self.covariance = currentCovariance

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
        F_x = np.zeros((3, state_vector_size))

        # The top-left 3x3 block is the identity matrix that updates the robot's pose
        F_x[0, 0] = 1  # x -> x
        F_x[1, 1] = 1  # y -> y
        F_x[2, 2] = 1

        predictedDelta = np.array(predictedDelta).reshape(-1, 1)

        self.state += F_x.T @ predictedDelta

        self.state[2] = self.utils.normalize_angle(self.state[2])

        self.covariance += F_x.T @ predicted_covariance @ F_x + F_x.T @ self.process_noise @ F_x

        return self.state, self.covariance


    def correct(self, scanMessage, currentStateVector, currentCovarianceMatrix):

        rospy.loginfo("\n === CORRECTION BEGINNING ====== CORRECTION BEGINNING ======")

        start_time = time.time()

        self.state = currentStateVector
        self.covariance = currentCovarianceMatrix
        
        x = self.state[0].item()
        y = self.state[1].item()
        theta = self.state[2].item()

        # Feature Extraction Step
        z_t = self.sensor.extract_features_from_scan(scanMessage, scanMessage.angle_min, scanMessage.angle_max, scanMessage.angle_increment)
        
        # z_t = self.utils.laser_scan_to_polar_tuples(scanMessage)

        # Start observation loop

        observation_counter = 0

        kalman_gain_list = []
        best_z_hat_list = []
        best_H_Matrix_list = []
        all_pi_list = []
        best_pi_list = []

        for z_i in z_t:

            observation_counter += 1

            # rospy.loginfo(f"Observation {observation_counter}")

            # initialize new landmark and create tempoprary state and covariance matrices
            newLandmark_x, newLandmark_y = self.map.calculate_landmark_estimates(x, y, theta, z_i)
            new_landmark = np.array([newLandmark_x, newLandmark_y])

            # Create temporary state and covariance matrix with the landmark in it

            tempState = np.vstack((self.state, new_landmark.reshape(2, 1)))

            n = self.covariance.shape[0]

            tempCovariance = np.zeros((n + 2, n + 2))
            tempCovariance[:n, :n] = self.covariance
            # Initialize landmark uncertainty proportional to the range measurement
            initial_landmark_uncertainty = (z_i[0] ** 2) / 130

            # initial_landmark_uncertainty = 10

            tempCovariance[n:, n:] = np.array([[initial_landmark_uncertainty, 0],
                                            [0, initial_landmark_uncertainty]])

            # Temporarily adjust landmark number too
            temp_num_landmarks = self.num_landmarks + 1

            landmark_counter = 0

            H_matrix_list = []
            psi_list = []
            pi_list = []
            z_hat_list = []

            # Start landmark loop
            for k in range(1, temp_num_landmarks + 1):

                landmark_counter += 1
                
                delta_k = np.array([tempState[1 + 2 * k] - x, tempState[1 + 2 * k + 1] - y])
                
                q_k = np.dot(delta_k.T, delta_k).item()
                
                z_hat_k = np.array([np.sqrt(q_k), np.arctan2(delta_k[1].item(), delta_k[0].item()) - theta])
                
                'The calculation of delta and q_k seem fine, as the z_hat_k is very close to the correct z_i'
                # rospy.loginfo(f"Observed measurement at obs {observation_counter}: {z_i}")
                # rospy.loginfo(f"Predicted Measurement at LM {landmark_counter}: {z_hat_k}")

                # Compute F_x,k matrix
                F_x_k = self.map.compute_F_x_k(temp_num_landmarks, k)
                
                ' F matrix structure seems to be correct'
                # rospy.loginfo(f"F matrix at obs {observation_counter} and lm {landmark_counter}: {F_x_k}")

                # Compute H^k_t matrix
                H_k_t = self.map.compute_H_k_t(delta_k, q_k, F_x_k)
            
                # Compute Mahalanobis distance
                
                Psi_k = H_k_t @ tempCovariance @ H_k_t.T + self.measurement_noise
                
                # rospy.loginfo(f"Psi matrix: {Psi_k}")
                
                eigenvalues, _ = np.linalg.eig(Psi_k)

                if np.any(eigenvalues <= 0):
                    rospy.logwarn(f"Warning: Negative or zero eigenvalues detected in Psi_k: {Psi_k}")

                measurement_residual_k = z_i - z_hat_k

                measurement_residual_k[1] = self.utils.normalize_angle(measurement_residual_k[1])

                pi_k = (measurement_residual_k).T @ np.linalg.inv(Psi_k) @ (measurement_residual_k)

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

            # rospy.loginfo(f"best_landmark_index: {best_landmark_index}")

            # Check if a new landmark is being added
            if best_landmark_index >= self.num_landmarks:

                rospy.loginfo(f"\n ADDING NEW LANDMARK at obs {observation_counter}, landmark {landmark_counter}")

                # Update state vector to include the new landmark
                self.state = tempState
                self.covariance = tempCovariance

                # Update the number of landmarks
                self.num_landmarks = temp_num_landmarks
                
                #  If a new Landmark is beeing added, return without correcting the state or covariance
                # return self.state, self.covariance, self.num_landmarks

                # best_H_matrix = H_matrix_list[best_landmark_index]
                # best_z_hat = z_hat_list[best_landmark_index]

            else:

                # Measurement is associated with an existing landmark

                best_H_matrix = H_matrix_list[best_landmark_index]
                best_H_matrix = best_H_matrix[:, :self.num_landmarks * 2 + 3]  # Truncate H matrix
                best_z_hat = z_hat_list[best_landmark_index]
                
                temp_num_landmarks -= 1
                
                rospy.loginfo(f"\n MATCHED OBSERVATION {observation_counter} WITH LANDMARK {best_landmark_index + 1} "
                              f"at position {new_landmark}, "
                              f"Observation Data: {z_i}, "
                              f"Predicted Measurement: {best_z_hat}, "
                            #   f"Covariance: {self.covariance}, "
                              f"Match Quality: {pi_list.index(j_i)}, "
                              f"Total Landmarks: {self.num_landmarks}")
            
                # Add both variables to the list for later
                best_H_Matrix_list.append(best_H_matrix)
                best_z_hat_list.append(best_z_hat)
                
                # rospy.loginfo(f"current z_i: {z_i}")
                # rospy.loginfo(f"best_z_hat: {best_z_hat}")
                
                # best_pi_list.append(pi_list[best_landmark_index])
                # all_pi_list.append(pi_list)
                
                # rospy.loginfo(f"Inverse of psi: {np.linalg.inv(psi_list[best_landmark_index])}")
                    
                # Calculte Kalman gain and att it to the list
                Kalman_gain = self.covariance @ best_H_matrix.T @ np.linalg.inv(psi_list[best_landmark_index])

                # rospy.loginfo(f"Kalman gain at obs {observation_counter}: {Kalman_gain}")
                # rospy.loginfo(f"H Matrix at obs {observation_counter}: {best_H_matrix}")
                # rospy.loginfo(f"Covariance at obs {observation_counter}: {self.covariance}")
                # rospy.loginfo(f"Psi inverse at obs {observation_counter}: {np.linalg.inv(psi_list[best_landmark_index])}")

                kalman_gain_list.append(Kalman_gain)
                
                
                ' Incremental updating '
                
                measurement_residual = z_i - best_z_hat
                
                state_update = Kalman_gain @ measurement_residual
                state_update = state_update.reshape((state_update.shape[0], 1))
                
                # rospy.loginfo(f"State update: {state_update.shape}")

                covariance_update = Kalman_gain @ best_H_matrix
                
                self.state += state_update
                self.state[2] = self.utils.normalize_angle(self.state[2])  # Normalize the orientation angle
                self.covariance = (np.eye(covariance_update.shape[0]) - covariance_update) @ self.covariance
                
                eigenvalues, _ = np.linalg.eig(self.covariance)

                if np.any(eigenvalues <= 0):
                    rospy.logwarn(f"Warning: Negative or zero eigenvalues detected in self.covariance before applying final covariance update: {self.covariance}")
                
                ' Incremental updating '
                
                
                # rospy.loginfo(f"Best h_matrix: {best_H_matrix}")
                # rospy.loginfo(f"Kalman Gain: {Kalman_gain}")
                
                # Create plots for H_Matrix jacobian and Covariance matrix
                # self.utils.save_covariance_matrix_plot(self.covariance, observation_counter, landmark_counter)
                # self.utils.save_jacobian_plot(best_H_matrix, observation_counter, landmark_counter)
                # self.utils.save_psi_plot(psi_list[best_landmark_index], observation_counter, landmark_counter)
                # self.utils.save_kalman_plot(Kalman_gain, observation_counter, landmark_counter)
            
        # end of observation loop  
        
        ' Multiple Kalman Gain Update '
        
        # updateStateSum = np.zeros_like(self.state)
        # updateCovarianceSum = np.zeros_like(self.covariance)

        # # rospy.loginfo(f"Best pi list: {best_pi_list}")
        # # rospy.loginfo(f"Pi list: {all_pi_list}")
        # # rospy.loginfo(f"\nBest z hat list: {best_z_hat_list}")
        # # rospy.loginfo(f"Z_t list: {z_t}")
        # # rospy.loginfo(f"\nKalman Gain list: {kalman_gain_list}")
        # # rospy.loginfo(f"Best H matrix list: {best_H_Matrix_list}")

        # for i, K_t_i in enumerate(kalman_gain_list):

        #     measurement_residual = z_t[i] - best_z_hat_list[i]

        #     measurement_residual[1] = self.utils.normalize_angle(measurement_residual[1])

        #     measurement_residual = measurement_residual.reshape(-1, 1)
            
        #     # rospy.loginfo(f"z_t: {z_t[i]}")
        #     # rospy.loginfo(f"z_hat: {best_z_hat_list[i]}")

        #     # rospy.loginfo(f"Kalman gain at iteration {i}: {K_t_i}")
        #     # rospy.loginfo(f"H Matrix at iteration {i}: {best_H_Matrix_list[i]}")

        #     state_update = K_t_i @ measurement_residual

        #     covariance_update = K_t_i @ best_H_Matrix_list[i]

        #     # rospy.loginfo(f"Calculated covariance update at iteration {i}: {covariance_update}")

        #     state_indices = slice(0, K_t_i.shape[0])

        #     updateStateSum[state_indices] += state_update

        #     updateCovarianceSum[state_indices, state_indices] += covariance_update

        # # rospy.loginfo(f"UpdateCovarianceSum at iteration {i}: {updateCovarianceSum}")


        # # rospy.loginfo(f"\n UpdateStateSum before update:\n{updateStateSum}")

        # # rospy.loginfo(f"\n updateCovarianceSum before update:\n{updateCovarianceSum}")


        # # rospy.loginfo(f"State before update: {self.state}")
        # # rospy.loginfo(f"Covariance before update: {self.covariance}")

        # self.state += updateStateSum
        # self.state[2] = self.utils.normalize_angle(self.state[2])  # Normalize the orientation angle
        # self.covariance = (np.eye(updateCovarianceSum.shape[0]) - updateCovarianceSum) @ self.covariance
        
        ' Multiple Kalman Gain Update '
        
        eigenvalues, _ = np.linalg.eig(self.covariance)

        if np.any(eigenvalues <= 0):
            rospy.logwarn(f"Warning: Negative or zero eigenvalues detected in self.covariance before applying final covariance update: {self.covariance}")

        # rospy.loginfo(f"Identity minus sum: {np.eye(updateCovarianceSum.shape[0]) - updateCovarianceSum}")

        # rospy.loginfo(f"State after update: {self.state}")
        # rospy.loginfo(f"Covariance after update: {self.covariance}")
        
        # self.utils.save_covariance_matrix_plot(self.covariance, observation_counter - 100, landmark_counter - 100)

        rospy.loginfo("\n === CORRECTION FINISHED ====== CORRECTION FINISHED ======")

        # self.utils.visualize_expected_Observation(z_hat_list, self.correctionCounter)
        self.correctionCounter += 1
        

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n Correction function took {elapsed_time:.6f} seconds.")

        return self.state, self.covariance, self.num_landmarks