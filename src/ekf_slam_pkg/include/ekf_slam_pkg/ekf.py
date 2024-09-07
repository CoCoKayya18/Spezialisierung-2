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

        return self.state

    def correct(self, scanMessage):
        # print("It works too:" + str(scanMessage))
        # EKF update step
        correctedPose = None
        self.state = correctedPose
        return self.state
