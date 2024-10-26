import os
import sys
import rospy
import subprocess
import cProfile
import pstats
from ekf_slam_pkg import EKFSLAM, Sensor, Map, Robot, Utils
import time  # Import the time module for frequency checking

def run_analysis_script():
    """Run the analysis script when ROS shuts down."""
    script_path = '/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/Debugging Scripts/correctionAnalysis.py'
    
    # Run the script using subprocess (this avoids blocking)
    subprocess.Popen(['python3', script_path])

def main_loop():
    """This is the main loop for EKF SLAM that we want to profile."""
    rospy.init_node('ekf_slam_launcher', anonymous=True)
    
    rospy.on_shutdown(run_analysis_script)

    # Initialize components
    config = {
        'initial_position': rospy.get_param("robot/initial_position"),
        'sensor_noise': rospy.get_param("sensor/noise"),
        'process_noise': rospy.get_param("ekf/process_noise"),
        'measurement_noise': rospy.get_param("ekf/measurement_noise")
    }
    
    map = Map(config)
    utils = Utils()
    sensor = Sensor(config, utils)
    ekf_slam = EKFSLAM(None, sensor, map, config, utils)  # Initialize without robot first
    
    robot = Robot(config, ekf_slam, utils)
    ekf_slam.robot = robot  # Set the robot reference after initialization

    # Initialize timing for frequency monitoring
    loop_rate = 5  # Desired frequency in Hz
    rate = rospy.Rate(loop_rate)
    last_time = time.time()

    rospy.loginfo("Started EKF SLAM launch")

    # Main loop where your SLAM steps occur
    while not rospy.is_shutdown():
        current_time = time.time()
        loop_duration = current_time - last_time
        last_time = current_time

        # Calculate current frequency
        frequency = 1 / loop_duration if loop_duration > 0 else float('inf')
        # rospy.loginfo(f"Current loop frequency: {frequency:.2f} Hz")

        # Sleep to maintain loop rate of 30 Hz
        # rate.sleep()

def main():
    # # Use cProfile to profile the main_loop
    # profiler = cProfile.Profile()
    # profiler.enable()  # Start profiling

    # try:
    #     main_loop()  # Call the main loop
    # finally:
        
    #     rospy.loginfo("Saving profiling data...")
        
    #     profiler.disable()  # Stop profiling

    #     # Save the profiling data to a .prof file for Snakeviz
    #     profiler_output_path = '/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/data/profiler_output_main.prof'
    #     profiler.dump_stats(profiler_output_path)

    #     # Optionally, print some profiling statistics to stdout
    #     stats = pstats.Stats(profiler, stream=sys.stdout)
    #     stats.strip_dirs()
    #     stats.sort_stats('cumtime')
    #     stats.print_stats(20)  # Print the top 20 functions

    #     # Also, save the stats to a text file for further inspection
    #     with open('/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/data/profiler_output_main.txt', 'w') as f:
    #         stats = pstats.Stats(profiler, stream=f)
    #         stats.strip_dirs()
    #         stats.sort_stats('cumtime')
    #         stats.print_stats(20)
    
    main_loop()


if __name__ == '__main__':
    main()
