import numpy as np
import pickle
import os

def wrap_angle(angle):
    # Wrap angle between -pi and pi
    return (angle + np.pi) % (2 * np.pi) - np.pi

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