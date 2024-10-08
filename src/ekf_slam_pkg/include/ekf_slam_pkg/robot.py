import rospy
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler
from math import sqrt
import threading
import csv

class Robot:
    def __init__(self, config, ekf_slam, utils):

        self.covariance = np.eye(3)
        self.num_landmarks = 0 
        self.state = np.array([[config['initial_position']['x']], 
                       [config['initial_position']['y']], 
                       [config['initial_position']['theta']]])

        self.ekf_slam = ekf_slam
        self.utils = utils
        rospy.loginfo("Robot class initialized")

        self.lock = threading.Lock() # Thread lock so predict and correct dont collide

        # Publishers and subscribers
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=1)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)
        self.ground_truth_sub = rospy.Subscriber('/ground_truth/state', Odometry, self.ground_truth_callback, queue_size=1)

        self.GT_path_pub = rospy.Publisher('/Ground_Truth_Path', Marker, queue_size=10)
        self.EKF_path_pub = rospy.Publisher('/EKF_Path', Marker, queue_size=10)
        self.map_pub = rospy.Publisher('/slam_map', Marker, queue_size=10)
        
        self.current_pose = None    
        self.current_vel = np.zeros((3, 1))
        self.ground_truth_path = []
        self.ekf_path = []
        self.filtered_points = []

        ground_truth_csv_path = '/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/data/ground_truth_path.csv'
        ekf_path_csv_path = '/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/data/ekf_path.csv'
        odom_velocities_csv_path = '/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/data/odom_veloc ities.csv'

        # Clear the plot directories
        self.utils.clear_directory("/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/plots/Covariance_Plots")
        self.utils.clear_directory("/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/plots/H_Jacobian_Plots")
        self.utils.clear_directory("/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/plots/DBSCAN_Plots")



        # Initialize CSV files with headers
        self.utils.initialize_csv_files(ground_truth_csv_path, ekf_path_csv_path, odom_velocities_csv_path)
        
    def odom_callback(self, msg):

        # self.current_pose = self.state
        with self.lock:

            self.current_vel = self.utils.transform_odometry_to_world(msg)
        
            # rospy.loginfo(f"\n State Vector:\n{self.state}")

            # rospy.loginfo(f"\n Covariance Matrix:\n{self.covariance}")

            ekf_predicted_pose, ekf_predicted_covariance = self.ekf_slam.predict(self.current_vel, self.state, self.covariance, self.num_landmarks)  # Run the EKF prediction
            
            self.state = ekf_predicted_pose
            self.covariance = ekf_predicted_covariance
            
            # self.ekf_path.append(ekf_predicted_pose)

            # rospy.loginfo(self.ekf_path)

            self.publish_EKF_path(self.ekf_path, "ekf_path", [0.0, 0.0, 1.0])  # Blue path

            # Save odom velocities to CSV
            # self.utils.save_odom_velocities_to_csv(msg)

    def scan_callback(self, msg):

        self.scan_message = msg

        with self.lock:

            # rospy.loginfo(f"\n State Vector before correction:\n{self.state}")
        
            ekf_corrected_pose, ekf_corrected_covariance, num_landmarks = self.ekf_slam.correct(self.scan_message, self.state, self.covariance)

            self.state = ekf_corrected_pose
            self.covariance = ekf_corrected_covariance
            self.num_landmarks = num_landmarks

            # rospy.loginfo(f"\n State Vector:\n{self.state}")

            # rospy.loginfo(f"\n Covariance Matrix:\n{self.covariance}")
        
            self.publish_map(self.ekf_slam.map.get_landmarks(self.state))
            self.ekf_path.append(ekf_corrected_pose)

            rospy.loginfo(self.ekf_path)

            self.publish_EKF_path(self.ekf_path, "ekf_path", [0.0, 0.0, 1.0])  # Blue path
            


    def ground_truth_callback(self, msg):
        self.ground_truth_path.append(msg.pose.pose)
        self.publish_GT_path(self.ground_truth_path, "ground_truth_path", [0.0, 1.0, 0.0])  # Green path

        # Save ground truth path to CSV
        # self.utils.save_ground_truth_path_to_csv(msg.pose.pose)

    def publish_GT_path(self, path, namespace, color):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = namespace
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.03  # Line width
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 0.5
        marker.pose.orientation.w = 1.0
        marker.lifetime = rospy.Duration(0)

        marker.points = [Point(p.position.x, p.position.y, p.position.z) for p in path]

        if self.GT_path_pub.get_num_connections() > 0:
            self.GT_path_pub.publish(marker)
        else:
            rospy.logwarn("No subscribers to the GT path topic or the topic is closed.")
        
    def publish_EKF_path(self, path, namespace, color, min_distance=0.0005):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = namespace
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.03  # Line width
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 0.5
        # Set the default pose for the marker (identity pose)
        marker.pose.orientation.w = 1.0
        marker.lifetime = rospy.Duration(0)

        # Persistent filtered points list to store path markers
        if not hasattr(self, 'filtered_points'):
            self.filtered_points = []  # Initialize the list only once

        # Check and add new points to the filtered points list
        if len(path) > 0:
            # Get the last filtered point
            last_point = self.filtered_points[-1] if self.filtered_points else None
            
            for p in path:

                x_val = p[0].item() if isinstance(p[0], np.ndarray) else p[0]
                y_val = p[1].item() if isinstance(p[1], np.ndarray) else p[1]

                new_point = Point(x_val, y_val, 0)

                if last_point is None or sqrt((new_point.x - last_point.x) ** 2 + (new_point.y - last_point.y) ** 2) > min_distance:
                    self.filtered_points.append(new_point)
                    last_point = new_point
                # else:
                    # rospy.loginfo(f"Point x: {new_point.x}, y: {new_point.y}, z: {new_point.z} is too close to the last point x: {last_point.x}, y: {last_point.y}, z: {last_point.z}, skipping.")

        # Set the filtered points to the marker
        marker.points = self.filtered_points

        # rospy.loginfo(f"Marker length: {len(marker.points)}")

        if self.EKF_path_pub.get_num_connections() > 0:
            self.EKF_path_pub.publish(marker)
        else:
            rospy.logwarn("No subscribers to the EKF path topic or the topic is closed.")

        # Save EKF path to CSV
        # self.utils.save_ekf_path_to_csv(path[-1])  # Save the last path point

    def publish_map(self, landmarks, namespace="landmarks", color=[1.0, 1.0, 0.0]):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = namespace
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.1  # Set the size of the points
        marker.scale.y = 0.1
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0  # Fully opaque

        # Check if landmarks are available
        # if len(landmarks) > 0:
        #     # Add only the first landmark as a point
        #     first_landmark = landmarks[0]
        #     point = Point()
        #     point.x = first_landmark[0]  # First landmark X coordinate
        #     point.y = first_landmark[1]  # First landmark Y coordinate
        #     point.z = 0                  # Landmarks are in 2D, so Z is 0
        #     marker.points.append(point)

        # Add landmarks as points
        for lm in landmarks:
            point = Point()
            point.x = lm[0]  # Landmark X coordinate
            point.y = lm[1]  # Landmark Y coordinate
            point.z = 0      # Landmarks are in 2D, so Z is 0
            marker.points.append(point)

        # # Publish the marker to the /slam_map topic
        if self.map_pub.get_num_connections() > 0:
            self.map_pub.publish(marker)
        else:
            rospy.logwarn("No subscribers to the map topic or the topic is closed.")

