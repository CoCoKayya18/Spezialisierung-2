import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory for saving plots
def create_directory(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

# Function to load JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to extract the robot's estimated path and save the plot
def plot_robot_path(data, save_dir):
    robot_path = []
    landmarks = []
    
    for correction in data:
        final_state = correction['correction']['final_state']
        robot_x = final_state[0][0]
        robot_y = final_state[1][0]
        robot_path.append((robot_x, robot_y))
        
        if len(final_state) > 3:
            landmarks_in_state = final_state[3:]  # Landmarks are stored after robot pose
            for i in range(0, len(landmarks_in_state), 2):
                landmark_x = landmarks_in_state[i][0]
                landmark_y = landmarks_in_state[i + 1][0]
                landmarks.append((landmark_x, landmark_y))

    robot_path = np.array(robot_path)

    plt.figure(figsize=(10, 6))
    plt.plot(robot_path[:, 0], robot_path[:, 1], marker='o', label='Estimated Path')
    
    # Plot the landmarks
    if landmarks:
        landmarks = np.array(landmarks)
        plt.scatter(landmarks[:, 0], landmarks[:, 1], marker='x', color='r', label='Landmarks')
    
    
    plt.title('Robot Estimated Path')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.legend()

    # Save the plot
    plt.savefig(os.path.join(save_dir, 'robot_estimated_path.png'))
    plt.close()

# Function to plot the residuals and save the plot
def plot_residuals(data, save_dir):
    residuals = []
    for correction in data:
        observations = correction['correction']['All']['observations']
        for obs in observations:
            for landmark in obs['landmarks']:
                z_hat = np.array(landmark['z_hat'])
                z_i = np.array(landmark['z_i'])
                residual = np.linalg.norm(z_hat - z_i)
                residuals.append(residual)

    residuals = np.array(residuals)

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(residuals)), residuals, marker='o', label='Residuals (z_hat - z_i)')
    plt.title('Residuals Over Time')
    plt.xlabel('Correction Step')
    plt.ylabel('Residual Magnitude')
    plt.grid(True)
    plt.legend()

    # Save the plot
    plt.savefig(os.path.join(save_dir, 'measurement_residuals_over_time.png'))
    plt.close()

# Function to track covariance matrix growth and save the plot
def plot_covariance_growth(data, save_dir):
    covariances = []
    for correction in data:
        final_covariance = np.array(correction['correction']['final_covariance'])
        covariance_diagonal = np.diag(final_covariance)
        covariances.append(covariance_diagonal)

    # Now that we have a list of covariance diagonals, we need to handle the fact that
    # their lengths may differ (because of growing state due to new landmarks)
    max_length = max([len(cov) for cov in covariances])

    # Pad the covariance diagonals with NaN (or zero) so they have the same length
    padded_covariances = np.array([np.pad(cov, (0, max_length - len(cov)), 'constant', constant_values=np.nan) for cov in covariances])

    # Plot each state variance over time
    plt.figure(figsize=(10, 6))
    for i in range(padded_covariances.shape[1]):
        plt.plot(padded_covariances[:, i], marker='o', label=f'Variance of state {i+1}')
    
    plt.title('Variance Growth Over Time')
    plt.xlabel('Correction Step')
    plt.ylabel('Variance')
    plt.grid(True)
    plt.legend()

    # Save the plot
    plt.savefig(os.path.join(save_dir, 'variance_growth.png'))
    plt.close()


# Main function to run all analysis
def analyze_ekf_slam(file_path, output_dir):
    # Create directory for saving plots
    create_directory(output_dir)

    # Load data
    data = load_json_data(file_path)

    # Analyze and plot robot path
    plot_robot_path(data, output_dir)

    # Analyze and plot residuals
    plot_residuals(data, output_dir)

    # Analyze and plot covariance growth
    plot_covariance_growth(data, output_dir)

    print(f'Analysis complete. Plots saved in {output_dir}')

# Example usage
if __name__ == "__main__":
    # Define file paths
    json_file_path = '/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/data/correctionData.json'  # Replace with your actual JSON file path
    output_directory = '/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/correctionAnalysisOutput'

    # Run the analysis
    analyze_ekf_slam(json_file_path, output_directory)
