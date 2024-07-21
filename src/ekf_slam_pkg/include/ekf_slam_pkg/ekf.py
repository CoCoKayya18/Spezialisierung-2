import numpy as np
import pickle
import rospy

class EKFSLAM:
    def __init__(self, robot, sensor, map, config, utils):
        self.robot = robot
        self.sensor = sensor
        self.map = map
        self.config = config
        self.utils = utils
        self.state = np.zeros((3, 1))  # Example: [x, y, theta]
        self.process_noise = config['process_noise']
        self.measurement_noise = config['measurement_noise']
        rospy.loginfo("EKF Class initialized")
        

    # EKF prediction step
    def predict(self, currentPose):

        print("It works:" + str(currentPose))

        # loaded_model = pickle.load(filepath)

        # # Predict using the loaded model and the standardized data
        # y_predict_mean, y_predict_variance = loaded_model.predict()

        # # Print the model parameters
        # print(loaded_model)
        self.predictedPose = currentPose
        return self.predictedPose

    def correct(self, scanMessage):
        print("It works too:" + str(scanMessage))
        # EKF update step
        self.correctedPose = self.predictedPose
        return self.correctedPose
