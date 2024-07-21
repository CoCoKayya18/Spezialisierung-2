import numpy as np
import rospy
import pickle
import os

# Add more utility functions as needed

# def scaleInput(odometryData):
#     with open(os.path.join(scalerFilePath, scaler_filenameX), 'rb') as file:
#         scaler_X = pickle.load(file)

#     standardizedInput = scaler_X.transform(odometryData)

#     return standardizedInput

# def rescaleOutput(output):
#     with open(os.path.join(scalerFilePath, scaler_filenameY), 'rb') as file:
#         scaler_Y = pickle.load(file)
    
#     rescaledOuput = scaler_Y.inverse_transform(output)

#     return rescaledOuput

class Utils:
    
    def __init__(self):
        rospy.loginfo("Utils class initialized")

    def wrap_angle(angle):
        # Wrap angle between -pi and pi
        return (angle + np.pi) % (2 * np.pi) - np.pi