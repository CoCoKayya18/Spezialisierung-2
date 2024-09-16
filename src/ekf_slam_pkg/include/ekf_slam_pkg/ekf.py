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
        self.state = Pose()
        self.covariance = None  
        self.process_noise = config['process_noise']
        self.measurement_noise = config['measurement_noise']
        self.predictedPose = Pose()

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
        predicted_covariance = np.diag(y_predict_variance ** 2)

        orientation_q = currentPosition.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, theta) = euler_from_quaternion(orientation_list)

        currentPosition.position.x += predictedDelta[0][0]  # Add the first delta to x
        currentPosition.position.y += predictedDelta[0][1]  # Add the second delta to y
        currentPosition.position.z = 0
        theta += predictedDelta[0][2]  # Add the third delta to theta

        # Convert updated theta back to quaternion
        q = quaternion_from_euler(0, 0, theta)

        # Set the orientation using the updated quaternion
        currentPosition.orientation.x = q[0]
        currentPosition.orientation.y = q[1]
        currentPosition.orientation.z = q[2]
        currentPosition.orientation.w = q[3]

        # Update state
        self.state = currentPosition
        self.covariance = predicted_covariance

        rospy.loginfo(f"Updated Position: {self.state.position.x}, {self.state.position.y}, {theta}")

        return self.state, self.covariance

    def correct(self, scanMessage, currentPredictionPose):

        # Extract current pose information
        x = currentPredictionPose.position.x
        y = currentPredictionPose.position.y
        orientation_q = currentPredictionPose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, theta = euler_from_quaternion(orientation_list)
        
        # EKF update step
        z_t = self.sensor.extract_features_from_scan(scanMessage, scanMessage.angle_min, scanMessage.angle_max, scanMessage.angle_increment)  # Extract features from LaserScan

        for z_i in z_t:

            # Add new Landmark
            newLandmark_x, newLandmark_y = self.map.add_landmark_estimates(x, y, theta, z_i)

            # Iterate through observed landmarks
            for k in range(1, self.N_t + 1):
                
                delta_k = np.array([
                    newLandmark_x - x,
                    newLandmark_y - y
                ])
                
                q_k = np.dot(delta_k.T, delta_k)
                
                z_hat_k = np.array([
                    np.sqrt(q_k),
                    np.arctan2(delta_k[1], delta_k[0]) - theta,
                ])

                # Compute F_x,k matrix
                F_x_k = self.map.compute_F_x_k(self.N_t, k)

                # Compute H^k_t matrix
                H_k_t = self.map.compute_H_k_t(delta_k, q_k, F_x_k)

                # Compute Mahalanobis distance
                pi_k, Psi_k = self.map.compute_mahalanobis_distance(z_i, z_hat_k, H_k_t, self.covariance)


            # Data association step
            correctLandmarkIndex = self.map_handler.data_association(pi_k)

            if correctLandmarkIndex is not None:
                # Update landmark index
                self.N_t = max(self.N_t, correctLandmarkIndex)

                # Kalman gain
                K_i_t = self.covariance @ H_k_t.T @ np.linalg.inv(Psi_k)

                # Update state mean and covariance using MapHandler
                x, y, theta = self.map_handler.update_state(x, y, theta, z_i, z_hat_k, K_i_t)
                self.covariance = (np.eye(len(self.covariance)) - K_i_t @ H_k_t) @ self.Sigma_t


        # Update the robot's pose based on the corrected state estimate
        self.state = self.utils.update_pose_from_state(currentPredictionPose, x, y, theta)

        # rospy.loginfo("EKF correction step completed.")

        return self.state, self.covariance

