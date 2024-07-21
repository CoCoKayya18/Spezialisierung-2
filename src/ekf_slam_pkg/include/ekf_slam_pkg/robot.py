import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

class Robot:
    def __init__(self, config, ekf_slam):
        self.position = config['initial_position']
        self.ekf_slam = ekf_slam
        rospy.loginfo("Robot class initialized")

        # Publishers and subscribers
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.odom_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.ground_truth_sub = rospy.Subscriber('/ground_truth/state', Odometry, self.ground_truth_callback)

        self.GT_path_pub = rospy.Publisher('/Ground_Truth_Path', Marker, queue_size=10)
        self.EKF_path_pub = rospy.Publisher('/EKF_Path', Marker, queue_size=10)
        
        self.current_pose = None
        self.ground_truth_path = []
        self.ekf_path = []
        
    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose
        ekf_predicted_pose = self.ekf_slam.predict(self.current_pose)  # Run the EKF prediction
        self.ekf_path.append(ekf_predicted_pose)
        self.publish_EKF_path(self.ekf_path, "ekf_path", [0.0, 0.0, 1.0])  # Blue path

    def scan_callback(self, msg):
        self.scan_message = msg
        ekf_corrected_pose = self.ekf_slam.correct(self.scan_message)
        self.ekf_path.append(ekf_corrected_pose)
        self.publish_EKF_path(self.ekf_path, "ekf_path", [0.0, 0.0, 1.0])  # Blue path

    def ground_truth_callback(self, msg):
        self.ground_truth_path.append(msg.pose.pose)
        self.publish_GT_path(self.ground_truth_path, "ground_truth_path", [0.0, 1.0, 0.0])  # Green path

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

        marker.points = [Point(p.position.x, p.position.y, p.position.z) for p in path]

        self.GT_path_pub.publish(marker)
        
    def publish_EKF_path(self, path, namespace, color):
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

        marker.points = [Point(p.position.x, p.position.y, p.position.z) for p in path]

        self.EKF_path_pub.publish(marker)
    

    # def get_pose(self):
    #     return self.current_pose

    # def get_control(self):
    #     rospy.loginfo("Control returned")
    #     return self.current_pose
