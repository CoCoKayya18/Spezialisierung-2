import numpy as np
import rospy
import pickle
import time
import sys
from geometry_msgs.msg import Pose
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_from_euler
import ujson as json


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
        self.alpha = 0.2

        self.F_x = np.eye(3)
        
        self.oldSignature = np.eye(2)
        
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
        z_t = self.sensor.extract_features_from_scan(scanMessage, scanMessage.angle_min, scanMessage.angle_max, scanMessage.angle_increment, self.correctionCounter)

        # z_t = self.sensor.detect_corners_and_circles_ransac(scanMessage, scanMessage.angle_min, scanMessage.angle_max, scanMessage.angle_increment, self.correctionCounter)

        # Start observation loop

        observation_counter = 0

        kalman_gain_list = []
        best_z_hat_list = []
        best_H_Matrix_list = []
        all_pi_list = []
        best_pi_list = []
        
        # Initialize the structure to store the correction data
        correction_data = {
            "correction": {
                "number": self.correctionCounter ,
                "initial_state": self.state.tolist(),
                "initial_covariance": self.covariance.tolist(),
                "All": {
                    "observations": []
                },
                "Matched": {
                    "observations": []
                },
                "newLandmarkData": {
                "landmarks": []
                },
                "final_state": None,
                "final_covariance": None
                # "features": {  # Add features section for lines, circles, corners, points
                #     "lines": [],
                #     "corners": [],
                #     "circles": [],
                #     "points": []
                # }
            }
        }
        
        # lines_data = self.sensor.get_lines()
        # corners = self.sensor.get_corners()
        # circles = self.sensor.get_circles()
        # points = self.sensor.get_points()
        
        # if lines_data:
        #     correction_data["correction"]["features"]["lines"] = [
        #         {
        #             "slope": line["slope"],
        #             "intercept": line["intercept"],
        #             "iteration": line["iteration"],
        #             "loopCounter": line["loopCounter"],
        #             "inliers": line["inliers"],  # Store inliers for visualization
        #             "outliers": line["outliers"]  # Store outliers for visualization
        #         }
        #         for line in lines_data
        #     ]

        # if corners:
        #     correction_data["correction"]["features"]["corners"] = [
        #         {"x": x, "y": y} for x, y in corners
        #     ]

        # if circles:
        #     correction_data["correction"]["features"]["circles"] = [
        #         {"x_center": xc, "y_center": yc, "radius": radius} for xc, yc, radius in circles
        #     ]

        # if points is not None:
        #     correction_data["correction"]["features"]["points"] = [
        #         {"x": x, "y": y} for x, y in points
        #     ]

        for z_i in z_t:

            observation_counter += 1
            
            observation = {
                "observation_id": observation_counter,
                "landmarks": []
            }
            
            # Normalize z_i
            z_i = list(z_i)  # Convert the tuple to a list
            z_i[1] = self.utils.normalize_angle(z_i[1])  # Modify the angle
            z_i = tuple(z_i)  # Convert it back to a tuple if necessary
                        
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
            
            # initial_landmark_uncertainty = 5

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
                
                z_hat_k[1] = self.utils.normalize_angle(z_hat_k[1])
                
                # Compute F_x,k matrix
                F_x_k = self.map.compute_F_x_k(temp_num_landmarks, k)
                
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
                
                landmark = {
                    "landmark_id": landmark_counter,
                    "z_i": z_i,
                    "z_hat": z_hat_k.tolist(),
                    "measurement_residual": measurement_residual_k.tolist(),
                    "H_matrix": H_k_t.tolist(),
                    "psi": Psi_k.tolist(),
                    "pi": pi_k
                }
                
                observation["landmarks"].append(landmark)
            
            # End Landmark Loop
            
            # Append the observation to the "All" section
            correction_data["correction"]["All"]["observations"].append(observation)

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
                
                measurement_residual_forJson = z_i - z_hat_list[best_landmark_index]
                
                # Add the new landmark to the "newLandmarkData" section
                new_landmark_data = {
                    "landmark_id": self.num_landmarks,
                    "new_landmark_position": new_landmark.tolist(),
                    "z_i": z_i,
                    "z_hat": z_hat_list[best_landmark_index].tolist(),
                    "measurement_residual": measurement_residual_forJson.tolist(),
                    "H_matrix": H_matrix_list[best_landmark_index].tolist(),
                    "psi": psi_list[best_landmark_index].tolist(),
                    "pi": pi_list[best_landmark_index]
                }
                
                correction_data["correction"]["newLandmarkData"]["landmarks"].append(new_landmark_data)

            else:

                # Measurement is associated with an existing landmark

                best_H_matrix = H_matrix_list[best_landmark_index]
                best_H_matrix = best_H_matrix[:, :self.num_landmarks * 2 + 3]  # Truncate H matrix
                best_z_hat = z_hat_list[best_landmark_index]
                
                temp_num_landmarks -= 1
            
                # Add both variables to the list for later
                best_H_Matrix_list.append(best_H_matrix)
                best_z_hat_list.append(best_z_hat)
                
                
                # best_pi_list.append(pi_list[best_landmark_index])
                # all_pi_list.append(pi_list)
                
                # rospy.loginfo(f"Inverse of psi: {np.linalg.inv(psi_list[best_landmark_index])}")
                    
                # Calculte Kalman gain and att it to the list
                Kalman_gain = self.covariance @ best_H_matrix.T @ np.linalg.inv(psi_list[best_landmark_index])
                kalman_gain_list.append(Kalman_gain)
                
                
                ' Incremental updating '
                
                measurement_residual = z_i - best_z_hat
                
                state_update = Kalman_gain @ measurement_residual
                state_update = state_update.reshape((state_update.shape[0], 1))

                covariance_update = Kalman_gain @ best_H_matrix
                
                self.state += state_update
                self.state[2] = self.utils.normalize_angle(self.state[2])  # Normalize the orientation angle
                self.covariance = (np.eye(covariance_update.shape[0]) - covariance_update) @ self.covariance
                
                eigenvalues, _ = np.linalg.eig(self.covariance)

                if np.any(eigenvalues <= 0):
                    rospy.logwarn(f"Warning: Negative or zero eigenvalues detected in self.covariance before applying final covariance update: {self.covariance}")
                
                ' Incremental updating '

                # Add match details to the "Matched" section
                matched_observation = {
                    "observation_id": observation_counter,
                    "matched_landmark_index": best_landmark_index + 1,
                    "landmarks": [
                        {
                            "landmark_id": best_landmark_index + 1,
                            "z_i": z_i,
                            "z_hat": best_z_hat.tolist(),
                            "measurement_residual": measurement_residual.tolist(),
                            "H_matrix": best_H_matrix.tolist(),
                            "psi": psi_list[best_landmark_index].tolist(),
                            "pi": pi_list[best_landmark_index],
                            "Kalman gain": Kalman_gain.tolist(),
                            "State update": state_update.tolist(),
                            "Covariance Update": covariance_update.tolist()
                        }
                    ]
                }

                correction_data["correction"]["Matched"]["observations"].append(matched_observation)
                
        # end of observation loop  
        
        eigenvalues, _ = np.linalg.eig(self.covariance)

        if np.any(eigenvalues <= 0):
            rospy.logwarn(f"Warning: Negative or zero eigenvalues detected in self.covariance before applying final covariance update: {self.covariance}")

        rospy.loginfo("\n === CORRECTION FINISHED ====== CORRECTION FINISHED ======")
        
        correction_data["correction"]["final_state"] = self.state.tolist()
        correction_data["correction"]["final_covariance"] = self.covariance.tolist()
        
        # Save everything to the JSON file
        self.utils.save_correction_data_to_json(correction_data)

        # self.utils.visualize_expected_Observation(z_hat_list, self.correctionCounter)
        self.correctionCounter += 1
        
        rospy.loginfo(f"New State: {self.state}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n Correction function took {elapsed_time:.6f} seconds.")

        return self.state, self.covariance, self.num_landmarks

    # JCBB Correction
    def correct_with_jcbb(self, scanMessage, currentStateVector, currentCovarianceMatrix):
        rospy.loginfo("\n === JCBB CORRECTION BEGINNING ====== JCBB CORRECTION BEGINNING ======")
        
        start_time = time.time()
        
        self.state = currentStateVector
        self.covariance = currentCovarianceMatrix
        
        x = self.state[0].item()
        y = self.state[1].item()
        theta = self.state[2].item()
        
        # Feature Extraction
        z_t = self.sensor.extract_features_from_scan(scanMessage, scanMessage.angle_min, scanMessage.angle_max, scanMessage.angle_increment, self.correctionCounter)
        
        # Observation Loop Initialization
        observation_counter = 0
        correction_data = {
            "correction": {
                "number": self.correctionCounter,
                "initial_state": self.state.tolist(),
                "initial_covariance": self.covariance.tolist(),
                "All": {"observations": []},
                "Matched": {"observations": []},
                "newLandmarkData": {"landmarks": []},
                "final_state": None,
                "final_covariance": None
            }
        }
        
        # Find matches using JCBB
        best_matches = self.jcbb_associate(z_t, self.map.get_landmarks(self.state), self.state, self.covariance)
        
        rospy.loginfo(f"Best matches: {best_matches}")
        
        for obs, landmark_idx in best_matches:
            observation_counter += 1
            observation = {
                "observation_id": observation_counter,
                "landmarks": []
            }
            
            # Normalize observation angle
            obs = list(obs)
            obs[1] = self.utils.normalize_angle(obs[1])
            obs = tuple(obs)
            
            if landmark_idx is None:
                # Handle New Landmark Case
                rospy.loginfo(f"\n ADDING NEW LANDMARK at observation {observation_counter}")
                newLandmark_x, newLandmark_y = self.map.calculate_landmark_estimates(x, y, theta, obs)
                new_landmark = np.array([newLandmark_x, newLandmark_y])
                
                # Update state and covariance for new landmark
                tempState = np.vstack((self.state, new_landmark.reshape(2, 1)))
                n = self.covariance.shape[0]
                tempCovariance = np.zeros((n + 2, n + 2))
                tempCovariance[:n, :n] = self.covariance
                initial_landmark_uncertainty = (obs[0] ** 2) / 130
                tempCovariance[n:, n:] = np.array([[initial_landmark_uncertainty, 0], [0, initial_landmark_uncertainty]])
                
                # Update state and covariance
                self.state = tempState
                self.covariance = tempCovariance
                self.num_landmarks += 1
                
                # Log new landmark
                new_landmark_data = {
                    "landmark_id": self.num_landmarks,
                    "new_landmark_position": new_landmark.tolist(),
                    "z_i": obs,
                    "measurement_residual": [0, 0]
                }
                correction_data["correction"]["newLandmarkData"]["landmarks"].append(new_landmark_data)
            
            else:
                # Handle Existing Landmark Update
                z_hat, H_extended = self.compute_expected_observation_and_jacobian(self.state, landmark_idx, self.num_landmarks)
                residual = obs - z_hat
                residual[1] = self.utils.normalize_angle(residual[1])
                
                Psi_k = H_extended @ self.covariance @ H_extended.T + self.measurement_noise
                Kalman_gain = self.covariance @ H_extended.T @ np.linalg.inv(Psi_k)
                
                # EKF State and Covariance Updates
                self.state += Kalman_gain @ residual
                self.state[2] = self.utils.normalize_angle(self.state[2])
                self.covariance = (np.eye(len(self.state)) - Kalman_gain @ H_extended) @ self.covariance
                
                # Log matched observation
                matched_observation = {
                    "observation_id": observation_counter,
                    "matched_landmark_index": landmark_idx + 1,
                    "landmarks": [
                        {
                            "landmark_id": landmark_idx + 1,
                            "z_i": obs,
                            "z_hat": z_hat.tolist(),
                            "measurement_residual": residual.tolist(),
                            "Kalman_gain": Kalman_gain.tolist()
                        }
                    ]
                }
                correction_data["correction"]["Matched"]["observations"].append(matched_observation)
        
        # Finalize Correction Data
        correction_data["correction"]["final_state"] = self.state.tolist()
        correction_data["correction"]["final_covariance"] = self.covariance.tolist()
        
        # Save correction data
        self.utils.save_correction_data_to_json(correction_data)
        
        rospy.loginfo("\n === JCBB CORRECTION FINISHED ====== JCBB CORRECTION FINISHED ======")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n JCBB Correction function took {elapsed_time:.6f} seconds.")
        
        return self.state, self.covariance, self.num_landmarks

    
    def jcbb_associate(self, observations, landmarks, currentStateVector, currentCovarianceMatrix):

        best_match = []
        best_compatibility = float('inf')

        def joint_compatibility_test(association):
            total_compatibility = 0
            for obs, idx in association:
                if idx is not None:
                    z_hat, H = self.compute_expected_observation_and_jacobian(currentStateVector, idx, len(landmarks))
                    residual = obs - z_hat
                    residual[1] = self.utils.normalize_angle(residual[1])

                    Psi_k = H @ currentCovarianceMatrix @ H.T + self.measurement_noise
                    Psi_k = Psi_k.astype(np.float64)
                    # Calculate mahalanobis_distance as a scalar
                    mahalanobis_distance = np.dot(residual.T, np.dot(np.linalg.inv(Psi_k), residual))


                    # Debugging output for mahalanobis_distance and threshold
                    rospy.loginfo(f"Mahalanobis distance: {mahalanobis_distance}, Threshold: {self.mahalanobis_threshold}")

                    if mahalanobis_distance > self.alpha:
                        return float('inf')  # Early rejection
                    total_compatibility += mahalanobis_distance

            return total_compatibility

        def recursive_jcbb(association, unmatched_observations, unmatched_landmarks):
            nonlocal best_match, best_compatibility

            if not unmatched_observations:
                # Evaluate the current association's total compatibility
                total_compatibility = joint_compatibility_test(association)
                rospy.loginfo(f"Total compatibility: {total_compatibility}, Best compatibility: {best_compatibility}")

                if total_compatibility < best_compatibility:
                    best_compatibility = total_compatibility
                    best_match = association[:]  # Update best match found so far
                return

            current_obs = unmatched_observations[0]
            remaining_observations = unmatched_observations[1:]

            for idx, landmark in enumerate(unmatched_landmarks):
                z_hat, H = self.compute_expected_observation_and_jacobian(currentStateVector, idx, len(landmarks))
                residual = current_obs - z_hat
                residual[1] = self.utils.normalize_angle(residual[1])

                Psi_k = H @ currentCovarianceMatrix @ H.T + self.measurement_noise
                Psi_k = Psi_k.astype(np.float64)
                # Calculate mahalanobis_distance as a scalar
                mahalanobis_distance = np.dot(residual.T, np.dot(np.linalg.inv(Psi_k), residual))


                # Debugging output to track each attempt at compatibility
                rospy.loginfo(f"Observation {current_obs}, Landmark {idx}, Mahalanobis distance: {mahalanobis_distance}")

                if mahalanobis_distance < self.alpha:  # Compatibility check
                    new_association = association + [(current_obs, idx)]
                    remaining_landmarks = unmatched_landmarks[:idx] + unmatched_landmarks[idx + 1:]
                    recursive_jcbb(new_association, remaining_observations, remaining_landmarks)

            # Consider the unmatched case for new landmark
            recursive_jcbb(association + [(current_obs, None)], remaining_observations, unmatched_landmarks)

        # Initial call to the recursive function
        recursive_jcbb([], observations, landmarks)

        # Debug output for final best match result
        rospy.loginfo(f"Best match found: {best_match} with compatibility: {best_compatibility}")
        return best_match

    def compute_expected_observation_and_jacobian(self, currentStateVector, landmark_index, num_landmarks):

        # Extract the robot's current state
        x = currentStateVector[0]
        y = currentStateVector[1]
        theta = currentStateVector[2]

        # Extract the landmark's position from the state vector
        x_l = currentStateVector[3 + 2 * landmark_index]
        y_l = currentStateVector[3 + 2 * landmark_index + 1]

        # Compute the differences in position between the robot and the landmark
        delta_x = x_l - x
        delta_y = y_l - y

        # Compute the squared distance (q) between the robot and the landmark
        q = delta_x**2 + delta_y**2

        # Compute the expected observation (z_hat), which consists of:
        # - The range (distance) to the landmark
        # - The bearing (angle) to the landmark, relative to the robot’s orientation
        range_to_landmark = np.sqrt(q)
        bearing_to_landmark = np.arctan2(delta_y, delta_x) - theta
        bearing_to_landmark = self.utils.normalize_angle(bearing_to_landmark)

        # z_hat: Expected observation [range, bearing]
        z_hat = np.array([range_to_landmark, bearing_to_landmark])

        # Compute the Jacobian matrix H (2x5) for the current observation
        sqrt_q = np.sqrt(q)

        H = np.array([
            [-delta_x / sqrt_q, -delta_y / sqrt_q, 0, delta_x / sqrt_q, delta_y / sqrt_q],     # Derivatives of range w.r.t [x, y, theta]
            [delta_y / q,      -delta_x / q,      -1, -delta_y / q,      delta_x / q]      # Derivatives of bearing w.r.t [x, y, theta]
        ])

        # Construct the F_xk matrix to map the small H matrix to the full state
        # The matrix F_xk is (5 x (3 + 2*num_landmarks)), where 3 is for the robot and 2*num_landmarks is for all landmarks
        F_xk = np.zeros((5, 3 + 2 * num_landmarks))
        F_xk[:3, :3] = np.eye(3)  # Mapping for robot state
        F_xk[3:, 3 + 2 * landmark_index - 1: 3 + 2 * landmark_index - 1 + 2] = np.eye(2)  # Mapping for landmark state

        # Extend the Jacobian matrix to the full state using F_xk
        H_extended = H @ F_xk

        return z_hat, H_extended



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