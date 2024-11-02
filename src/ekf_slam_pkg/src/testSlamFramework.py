import subprocess
import time
import os
import shutil
import glob
import subprocess

def run_corner_extraction_script(script_path):

    print("Running corner extraction script...")
    subprocess.run(['python3', script_path], check=True)
    print("Corner extraction completed.")

def clear_directory(directory):
    files = glob.glob(f"{directory}/*")
    for f in files:
        os.remove(f)

def clear_folder_and_reset_file(folder_path, text_file_path):

    # Check and delete all subfolders in the specified folder
    if os.path.exists(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"Deleted folder: {item_path}")
            except Exception as e:
                print(f"Failed to delete {item_path}: {e}")
    else:
        print(f"Folder path {folder_path} does not exist.")
        
    # Write '0' at the start of the text file
    if os.path.exists(text_file_path):
        try:
            with open(text_file_path, 'w') as file:
                file.write("0")
            print(f"Reset file: {text_file_path} with '0'")
        except Exception as e:
            print(f"Failed to reset {text_file_path}: {e}")
    else:
        print(f"Text file path {text_file_path} does not exist.")
    
    # Clear the CSV file
    evaluation_csv = "/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/slamPlotterEvaluationPlots/SlamPlotterEvaluation.csv"
    if os.path.exists(evaluation_csv):
        try:
            with open(evaluation_csv, "w") as file:
                file.truncate(0)  # This will clear the file
            print(f"Cleared CSV file: {evaluation_csv}")
        except Exception as e:
            print(f"Failed to clear {evaluation_csv}: {e}")
    else:
        print(f"CSV file path {evaluation_csv} does not exist.")

def run_slam_pipeline(launch_file, folder_path, text_file_path, runs=10, delay=5):
    

    # Clear folder and reset text file once at the start
    if folder_path and text_file_path:
        print("Clearing folder and resetting file once at the beginning")
        clear_folder_and_reset_file(folder_path, text_file_path)
    
    clear_directory("/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/data")

    for i in range(runs):
        print(f"\nStarting SLAM pipeline run {i + 1}/{runs}")
        
        # Start the ROS launch file
        process = subprocess.Popen(['roslaunch'] + launch_file.split())
        
        # Wait for the SLAM pipeline to complete (adjust based on your pipeline's duration)
        time.sleep(480)  # Adjust duration as needed
        
        # Stop the ROS process
        process.terminate()
        process.wait()  # Ensure the process has completely terminated
        
        print(f"SLAM pipeline run {i + 1} completed")
        
        # Wait before restarting, if necessary
        time.sleep(delay)

# Example usage
launch_file = 'ekf_slam_pkg run_slam.launch'
folder_path = '/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/slamPlotterEvaluationPlots'
text_file_path = '/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/slamPlotterEvaluationPlots/run_counter.txt'

run_slam_pipeline(launch_file, runs=10, delay=5, folder_path=folder_path, text_file_path=text_file_path)
