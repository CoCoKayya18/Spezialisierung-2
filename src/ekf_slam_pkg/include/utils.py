import numpy as np

def wrap_angle(angle):
    # Wrap angle between -pi and pi
    return (angle + np.pi) % (2 * np.pi) - np.pi

# Add more utility functions as needed
