import rospy
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler
from math import sqrt
import csv

class Robot:
    def __init__(self, config, ekf_slam, utils):

        self.covariance = np.eye(3)
        self.num_landmarks = 0 
        self.state = [config['initial_position']['x'], config['initial_position']['y'], config['initial_position']['theta']]

        self.ekf_slam = ekf_slam
        self.utils = utils
        rospy.loginfo("Robot class initialized")

        # Publishers and subscribers
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.ground_truth_sub = rospy.Subscriber('/ground_truth/state', Odometry, self.ground_truth_callback)

        self.GT_path_pub = rospy.Publisher('/Ground_Truth_Path', Marker, queue_size=10)
        self.EKF_path_pub = rospy.Publisher('/EKF_Path', Marker, queue_size=10)
        self.map_pub = rospy.Publisher('/slam_map', Marker, queue_size=10)
        
        self.current_pose = None
        self.current_vel = np.zeros((3, 1))
        self.ground_truth_path = []
        self.ekf_path = []

        ground_truth_csv_path = '../Spezialisierung-2/src/ekf_slam_pkg/data/ground_truth_path.csv'
        ekf_path_csv_path = '../Spezialisierung-2/src/ekf_slam_pkg/data/ekf_path.csv'
        odom_velocities_csv_path = '../Spezialisierung-2/src/ekf_slam_pkg/data/odom_velocities.csv'

        # Initialize CSV files with headers
        # self.utils.initialize_csv_files(ground_truth_csv_path, ekf_path_csv_path, odom_velocities_csv_path)
        
    def odom_callback(self, msg):

        self.current_pose = self.state

        self.current_vel = self.utils.transform_odometry_to_world(msg)

        ekf_predicted_pose, ekf_predicted_covariance = self.ekf_slam.predict(self.current_vel, self.current_pose)  # Run the EKF prediction
        
        self.state = ekf_predicted_pose
        self.covariance = ekf_predicted_covariance
        
        self.ekf_path.append(ekf_predicted_pose)
        self.publish_EKF_path(self.ekf_path, "ekf_path", [0.0, 0.0, 1.0])  # Blue path

        # Save odom velocities to CSV
        # self.utils.save_odom_velocities_to_csv(msg)

    def scan_callback(self, msg):
        self.scan_message = msg
        
        ekf_corrected_pose, ekf_corrected_covariance = self.ekf_slam.correct(self.scan_message, self.state, self.covariance)

        self.position = ekf_corrected_pose
        
        self.publish_map(self.ekf_slam.map.get_landmarks(self.state))
        self.ekf_path.append(ekf_corrected_pose)
        self.publish_EKF_path(self.ekf_path, "ekf_path", [0.0, 0.0, 1.0])  # Red path

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
        
    def publish_EKF_path(self, path, namespace, color, min_distance=0.05):
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
                new_point = Point(p[0], p[1], 0)
                if last_point is None or sqrt((new_point.x - last_point.x) ** 2 + (new_point.y - last_point.y) ** 2) > min_distance:
                    self.filtered_points.append(new_point)
                    last_point = new_point

        # Set the filtered points to the marker
        marker.points = self.filtered_points

        if self.EKF_path_pub.get_num_connections() > 0:
            self.EKF_path_pub.publish(marker)
        else:
            rospy.logwarn("No subscribers to the EKF path topic or the topic is closed.")

        # Save EKF path to CSV
        self.utils.save_ekf_path_to_csv(path[-1])  # Save the last path point

    def publish_map(self, landmarks, namespace="landmarks", color=[1.0, 0.0, 0.0]):
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

        # Add landmarks as points
        for lm in landmarks:
            point = Point()
            point.x = lm[0]  # Landmark X coordinate
            point.y = lm[1]  # Landmark Y coordinate
            point.z = 0      # Landmarks are in 2D, so Z is 0
            marker.points.append(point)

        # Publish the marker to the /slam_map topic
        if self.map_pub.get_num_connections() > 0:
            self.map_pub.publish(marker)
        else:
            rospy.logwarn("No subscribers to the map topic or the topic is closed.")

