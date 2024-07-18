import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

class Robot:
    def __init__(self, config, ekf_slam):
        self.state = config['initial_state']
        self.position = config['initial_position']
        self.ekf_slam = ekf_slam

        # Publishers and subscribers
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        self.current_pose = None
        
    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose
        self.ekf_slam.run()  # Run the EKF 
        
    def get_pose(self):
        return self.current_pose
