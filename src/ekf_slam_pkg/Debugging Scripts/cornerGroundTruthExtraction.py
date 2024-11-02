import rospy
from gazebo_msgs.srv import GetModelState
import matplotlib.pyplot as plt
import numpy as np
import csv

def get_model_state(model_name):

    rospy.wait_for_service('/gazebo/get_model_state')
    try:
        get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        response = get_state(model_name, 'world')
        x = response.pose.position.x
        y = response.pose.position.y
        
        # Extract yaw from the quaternion orientation
        q = response.pose.orientation
        yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
        
        return x, y, yaw
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return None

def calculate_corners(x, y, yaw, length, width):
    half_length = length / 2.0
    half_width = width / 2.0
    
    # If there's no yaw, just compute corners directly in world frame
    if abs(yaw) < 1e-6:  # Assuming yaw is zero or negligible
        global_corners = np.array([
            [x + half_length, y + half_width],
            [x - half_length, y + half_width],
            [x - half_length, y - half_width],
            [x + half_length, y - half_width]
        ])
    else:
        # Apply rotation only if yaw is non-zero
        local_corners = np.array([
            [half_length, half_width],
            [-half_length, half_width],
            [-half_length, -half_width],
            [half_length, -half_width]
        ])
        
        rotation_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw),  np.cos(yaw)]
        ])
        
        global_corners = np.dot(local_corners, rotation_matrix.T) + np.array([x, y])
    
    return global_corners


def save_corners_to_csv(file_path, model_names, size):
    with open(file_path, 'w', newline='') as csvfile:
        fieldnames = ['Model_Name', 'Corner_Index', 'X', 'Y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for model_name in model_names:
            position = get_model_state(model_name)
            if position:
                x, y, yaw = position
                length, width = size[0], size[1]  # Assuming the same size for each model; adjust as necessary
                corners = calculate_corners(x, y, yaw, length, width)
                
                for j, corner in enumerate(corners):
                    writer.writerow({
                        'Model_Name': model_name,
                        'Corner_Index': j + 1,
                        'X': corner[0],
                        'Y': corner[1]
                    })
    print(f"Corners data saved to {file_path}")
    
def plot_boxes_from_csv(file_path):
    # Dictionary to store corners by model
    boxes = {}

    # Read the CSV file
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            model_name = row['Model_Name']
            x, y = float(row['X']), float(row['Y'])
            
            if model_name not in boxes:
                boxes[model_name] = []
            boxes[model_name].append((x, y))

    # Plot each box and its corners
    plt.figure(figsize=(10, 10))
    for model_name, corners in boxes.items():
        corners = np.array(corners)
        
        # Close the box by appending the first corner at the end
        corners = np.vstack([corners, corners[0]])

        plt.plot(corners[:, 0], corners[:, 1], '-o', label=model_name)

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Box Positions and Corners in Gazebo World")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig("/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/slamPlotterEvaluationPlots/Run_1/Run_1_Landmarks_Map_extracted.png")
    plt.close()

def main(output_csv_path, size):

    # List of model names as per the provided Gazebo world
    model_names = [
        "link_0", "link_0_clone", "link_0_clone_0", "link_0_clone_0_clone",
        "link_0_clone_0_clone_0", "link_0_clone_0_clone_1", "link_0_clone_1",
        "link_0_clone_clone", "link_0_clone_clone_clone", "link_0_clone_clone_clone_0",
        "link_0_clone_clone_clone_clone", "link_1", "link_1_clone"
    ]
    
    save_corners_to_csv(output_csv_path, model_names, size)
    plot_boxes_from_csv(output_csv_path)

# Example usage
if __name__ == "__main__":
    rospy.init_node('get_corners_from_gazebo', anonymous=True)
    
    output_csv_path = '/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/features/cornerGroundTruthPositions.csv'
    box_size = (1.0, 1.0)  # Replace with actual size (length, width) of your boxes
    
    main(output_csv_path, box_size)
