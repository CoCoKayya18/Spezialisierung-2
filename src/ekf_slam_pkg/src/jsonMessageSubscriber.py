#!/usr/bin/env python
import rospy
import json
from std_msgs.msg import String

class CorrectionDataSubscriber:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('correction_data_subscriber', anonymous=True)
        
        # Path to save the JSON file
        self.json_file_path = '/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/data/correction_data.json'
        
        # Subscribe to the topic where correction data is published
        rospy.Subscriber('/correction_data', String, self.callback)
        
        rospy.loginfo("CorrectionDataSubscriber node initialized.")

    def callback(self, msg):
        # Parse the JSON data from the message
        correction_data = json.loads(msg.data)
        
        # Append data to JSON file in a human-readable format
        self.save_data_in_readable_format(correction_data)

    def save_data_in_readable_format(self, correction_data):
        # Open the JSON file in append mode and write each JSON object with indentation
        with open(self.json_file_path, 'a') as json_file:
            json_file.write(json.dumps(correction_data, indent=4))  # Pretty-print JSON with indentation
            json_file.write('\n\n')  # Add a newline for separation between entries

    def on_shutdown(self):
        rospy.loginfo("CorrectionDataSubscriber node is shutting down.")

if __name__ == '__main__':
    try:
        # Instantiate the subscriber and keep the node running
        node = CorrectionDataSubscriber()
        rospy.on_shutdown(node.on_shutdown)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
