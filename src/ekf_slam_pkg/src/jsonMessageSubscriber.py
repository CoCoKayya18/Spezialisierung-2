#!/usr/bin/env python
import rospy
import json
import os
from std_msgs.msg import String

class CorrectionDataSubscriber:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('correction_data_subscriber', anonymous=True)
        
        # Base path for the JSON file and run counter file
        self.base_path = "/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/data"
        self.run_id = self.get_next_run_id()
        
        # Path to save the JSON file with run_id in the filename
        self.json_file_path = os.path.join(self.base_path, f'correction_data_run_{self.run_id}.json')

        # Subscribe to the topic where correction data is published
        rospy.Subscriber('/correction_data', String, self.callback)
        
        rospy.loginfo("CorrectionDataSubscriber node initialized.")

    def get_next_run_id(self):
        """Read the current run ID from file and increment it for the next run."""
        counter_file = os.path.join(self.base_path, "run_counter.txt")
        
        # Check if the counter file exists; initialize if not
        if not os.path.exists(counter_file):
            with open(counter_file, "w") as file:
                file.write("1")
            return 1

        # Read the current run_id, increment, and save it
        with open(counter_file, "r") as file:
            run_id = int(file.read().strip()) + 1
        
        with open(counter_file, "w") as file:
            file.write(str(run_id))
        
        return run_id

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
