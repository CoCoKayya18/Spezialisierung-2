import numpy as np
import pickle

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

    def predict(self, control):

        print("It works:" + control)
        # loaded_model = pickle.load(filepath)

        # # Predict using the loaded model and the standardized data
        # y_predict_mean, y_predict_variance = loaded_model.predict()

        # # Print the model parameters
        # print(loaded_model)

        # EKF prediction step
        pass

    def update(self, measurements):
        # EKF update step
        pass

    def run(self):
        control = self.robot.get_control()
        prediction = self.predict(control)
        
        # measurements = self.sensor.get_measurements()
        # self.update(measurements)
        # self.map.update(self.state, measurements)
