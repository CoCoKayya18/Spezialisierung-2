import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from math import atan2, sqrt, pow

class GoToGoal:
    def __init__(self):
        rospy.init_node('go_to_goal')

        # Publisher to send velocity commands
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # Subscriber to get current position from odometry
        self.odom_subscriber = rospy.Subscriber('/odom', Odometry, self.update_position)

        # Initialize robot position
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Control loop rate
        self.rate = rospy.Rate(10)

    def update_position(self, data):
        self.x = data.pose.pose.position.x
        self.y = data.pose.pose.position.y
        orientation_q = data.pose.pose.orientation
        self.theta = 2 * atan2(orientation_q.z, orientation_q.w)  # Only z and w are needed for 2D yaw

    def move_to_goal(self, goal_x, goal_y, linear_speed=0.2, angular_speed=1.0):

        goal_reached = False

        while not rospy.is_shutdown() and not goal_reached:
            # Calculate distance to goal
            distance = sqrt(pow((goal_x - self.x), 2) + pow((goal_y - self.y), 2))

            # Calculate the desired yaw angle to the goal
            goal_theta = atan2(goal_y - self.y, goal_x - self.x)
            angle_diff = goal_theta - self.theta

            # Twist message for velocity
            velocity_msg = Twist()

            # Proportional control for angular speed
            if abs(angle_diff) > 0.1:
                velocity_msg.angular.z = angular_speed * angle_diff / abs(angle_diff)
            else:
                velocity_msg.angular.z = 0.0

            # Proportional control for linear speed (move forward when facing the goal)
            if abs(angle_diff) < 0.1:
                velocity_msg.linear.x = min(linear_speed, linear_speed * distance)
            else:
                velocity_msg.linear.x = 0.0

            # Publish velocity command
            self.velocity_publisher.publish(velocity_msg)

            # Check if goal is reached
            if distance < 0.05:
                goal_reached = True
                velocity_msg.linear.x = 0
                velocity_msg.angular.z = 0
                self.velocity_publisher.publish(velocity_msg)  # Stop the robot
                rospy.loginfo("Goal reached!")

            self.rate.sleep()

if __name__ == '__main__':
    try:
        # Initialize the GoToGoal class
        navigator = GoToGoal()

        # Define goal locations
        goal_positions = [(5.0, 8.0), (10.0, -2.0), (-7.0, -4.0)]

        # Move the robot to each goal position
        for (goal_x, goal_y) in goal_positions:
            rospy.loginfo(f"Moving to goal: x={goal_x}, y={goal_y}")
            navigator.move_to_goal(goal_x, goal_y)
            rospy.sleep(1)  # Pause briefly before moving to the next goal

    except rospy.ROSInterruptException:
        pass
