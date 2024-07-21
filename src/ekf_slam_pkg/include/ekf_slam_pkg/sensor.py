import rospy

class Sensor:
    def __init__(self, config):
        rospy.loginfo("Sensor class initialized")
        pass

    def get_measurements(self):
        # Return sensor measurements
        return []
