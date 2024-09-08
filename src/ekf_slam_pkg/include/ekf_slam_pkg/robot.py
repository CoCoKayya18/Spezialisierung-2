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

        #self.position = config['initial_position']
        self.position = Pose()
        self.position.position.x = config['initial_position']['x']
        self.position.position.y = config['initial_position']['y']
        self.position.position.z = 0  # Assuming the robot is in 2D space

        # Initialize orientation using Euler angles to quaternion (assuming yaw in config)
        quaternion = quaternion_from_euler(0, 0, config['initial_position']['theta'])
        self.position.orientation.x = quaternion[0]
        self.position.orientation.y = quaternion[1]
        self.position.orientation.z = quaternion[2]
        self.position.orientation.w = quaternion[3]

        self.ekf_slam = ekf_slam
        self.utils = utils
        rospy.loginfo("Robot class initialized")

        # Publishers and subscribers
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.ground_truth_sub = rospy.Subscriber('/ground_truth/state', Odometry, self.ground_truth_callback)

        self.GT_path_pub = rospy.Publisher('/Ground_Truth_Path', Marker, queue_size=10)
        self.EKF_path_pub = rospy.Publisher('/EKF_Path', Marker, queue_size=10)
        
        self.current_pose = None
        self.current_vel = np.zeros((3, 1))
        self.ground_truth_path = []
        self.ekf_path = []

        ground_truth_csv_path = '../Spezialisierung-2/src/ekf_slam_pkg/data/ground_truth_path.csv'
        ekf_path_csv_path = '../Spezialisierung-2/src/ekf_slam_pkg/data/ekf_path.csv'
        odom_velocities_csv_path = '../Spezialisierung-2/src/ekf_slam_pkg/data/odom_velocities.csv'

        # Initialize CSV files with headers
        self.utils.initialize_csv_files(ground_truth_csv_path, ekf_path_csv_path, odom_velocities_csv_path)
        
    def odom_callback(self, msg):

        self.current_pose = self.position
        # self.current_vel[0] = msg.twist.twist.linear.x
        # self.current_vel[1] = msg.twist.twist.linear.y
        # self.current_vel[2] = msg.twist.twist.angular.z

        self.current_vel = self.utils.transform_odometry_to_world(msg)

        ekf_predicted_pose = self.ekf_slam.predict(self.current_vel, self.current_pose)  # Run the EKF prediction
        
        self.position = ekf_predicted_pose
        
        self.ekf_path.append(ekf_predicted_pose)
        self.publish_EKF_path(self.ekf_path, "ekf_path", [0.0, 0.0, 1.0])  # Blue path

        # Save odom velocities to CSV
        self.utils.save_odom_velocities_to_csv(msg)

    def scan_callback(self, msg):
        self.scan_message = msg
        
        # ekf_corrected_pose = self.ekf_slam.correct(self.scan_message)

        # self.position = ekf_corrected_pose
        
        # self.ekf_path.append(ekf_corrected_pose)
        # self.publish_EKF_path(self.ekf_path, "ekf_path", [1.0, 0.0, 0.0])  # Red path

    def ground_truth_callback(self, msg):
        self.ground_truth_path.append(msg.pose.pose)
        self.publish_GT_path(self.ground_truth_path, "ground_truth_path", [0.0, 1.0, 0.0])  # Green path

        # Save ground truth path to CSV
        self.utils.save_ground_truth_path_to_csv(msg.pose.pose)

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
        marker.lifetime = rospy.Duration(0)

        marker.points = [Point(p.position.x, p.position.y, p.position.z) for p in path]

        # print(f"Number of points in marker: {len(marker.points)}")

        self.GT_path_pub.publish(marker)
        
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
                new_point = Point(p.position.x, p.position.y, p.position.z)
                if last_point is None or sqrt((new_point.x - last_point.x) ** 2 + (new_point.y - last_point.y) ** 2) > min_distance:
                    self.filtered_points.append(new_point)
                    last_point = new_point

        # Set the filtered points to the marker
        marker.points = self.filtered_points

        self.EKF_path_pub.publish(marker)

        # Save EKF path to CSV
        self.utils.save_ekf_path_to_csv(path[-1])  # Save the last path point
    

    # def get_pose(self):
    #     return self.current_pose

    # def get_control(self):
    #     rospy.loginfo("Control returned")
    #     return self.current_pose
