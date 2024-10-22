import json
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

def clear_directory(directory_path):
    """
    Clears all files in the specified directory.
    If the directory doesn't exist, it will create it.
    """
    if os.path.exists(directory_path):
        # If the directory exists, delete all its contents
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory and its contents
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        # Create the directory if it doesn't exist
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")

    print(f"Directory {directory_path} is cleared.")


# Create directory for saving plots
def create_directory(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

# Function to load JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to visualize state updates over time
def plot_state_updates_per_correction(data, save_dir):
    state_updates = []
    sum_of_updates = None  # Initialize to None, will set it to the size of state when available

    for correction in data:
        matched_observations = correction['correction'].get('Matched', {}).get('observations', [])
        
        # Check if there is a matched observation with a state update
        if matched_observations:
            state_update = matched_observations[0]['landmarks'][0].get('State update', None)

            if state_update:
                state_update = np.array(state_update).flatten()  # Convert to 1D array
                state_updates.append(state_update)

                # Initialize sum_of_updates if it hasn't been done yet
                if sum_of_updates is None:
                    sum_of_updates = np.zeros_like(state_update)
                
                sum_of_updates += state_update  # Sum the state updates
        else:
            # Skip corrections with no state update (e.g., newly detected landmarks)
            continue

    # Check if state updates exist
    if state_updates:
        state_updates = np.array(state_updates)

        # Plot each state update over time
        plt.figure(figsize=(10, 6))
        for i in range(state_updates.shape[1]):
            plt.plot(range(1, len(state_updates) + 1), state_updates[:, i], marker='o', label=f'State {i+1} Update')

        # Add the sum of all updates as a text box on the plot
        sum_text = f"Sum of all updates:\n{sum_of_updates}"
        plt.gcf().text(0.15, 0.75, sum_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

        plt.title('State Updates Per Correction Step')
        plt.xlabel('Correction Step')
        plt.ylabel('State Update Value')
        plt.grid(True)
        plt.legend()

        # Save the plot
        plt.savefig(os.path.join(save_dir, 'state_updates_per_correction.png'))
        plt.close()

        # Print the sum of all updates for reference in console
        print("Sum of all state updates:", sum_of_updates)
    else:
        print("No state updates available for plotting.")

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
    range_residuals = []
    bearing_residuals = []

    for correction in data:
        observations = correction['correction']['All']['observations']
        for obs in observations:
            for landmark in obs['landmarks']:
                measurement_residual = np.array(landmark['measurement_residual'])

                # Extract the range and bearing residuals
                range_residual = measurement_residual[0]  # Range component
                bearing_residual = measurement_residual[1]  # Bearing component
                
                range_residuals.append(range_residual)
                bearing_residuals.append(bearing_residual)

    range_residuals = np.array(range_residuals)
    bearing_residuals = np.array(bearing_residuals)

    # Plot residuals for both range and bearing
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(range_residuals)), range_residuals, marker='o', label='Range Residuals')
    plt.plot(range(len(bearing_residuals)), bearing_residuals, marker='x', label='Bearing Residuals')
    plt.title('Measurement Residuals Over Time (All Observations)')
    plt.xlabel('Correction Step')
    plt.ylabel('Residual Value')
    plt.grid(True)
    plt.legend()

    # Save the plot
    plt.savefig(os.path.join(save_dir, 'measurement_residuals_all.png'))
    plt.close()

# Function to plot measurement residuals only from the matched section
def plot_residuals_magnitude_matched(data, save_dir):
    residuals = []
    
    for correction in data:
        matched_observations = correction['correction'].get('Matched', {}).get('observations', [])
        
        # Process only if there are matched observations
        if matched_observations:
            for obs in matched_observations:
                for landmark in obs['landmarks']:
                    z_hat = np.array(landmark['z_hat'])
                    z_i = np.array(landmark['z_i'])
                    residual = np.linalg.norm(z_hat - z_i)  # Calculate the residual
                    residuals.append(residual)
        else:
            # No matched data; skip this correction step
            continue

    # Convert to numpy array for plotting
    residuals = np.array(residuals)

    # Plot residuals
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(residuals)), residuals, marker='o', label='Matched Residuals (z_hat - z_i)')
    plt.title('Matched Measurement Residuals magnitude Over Time')
    plt.xlabel('Correction Step')
    plt.ylabel('Residual Magnitude')
    plt.grid(True)
    plt.legend()

    # Save the plot
    plt.savefig(os.path.join(save_dir, 'measurement_residuals_matched.png'))
    plt.close()
    
# Function to plot residuals directly from the JSON data for matched observations
def plot_residuals_matched(data, save_dir):
    range_residuals = []
    bearing_residuals = []

    for correction in data:
        matched_observations = correction['correction'].get('Matched', {}).get('observations', [])

        # Process only if there are matched observations
        if matched_observations:
            for obs in matched_observations:
                for landmark in obs['landmarks']:
                    measurement_residual = np.array(landmark['measurement_residual'])

                    # Extract the range and bearing residuals
                    range_residual = measurement_residual[0]  # Range component
                    bearing_residual = measurement_residual[1]  # Bearing component

                    range_residuals.append(range_residual)
                    bearing_residuals.append(bearing_residual)

    range_residuals = np.array(range_residuals)
    bearing_residuals = np.array(bearing_residuals)

    # Plot residuals for both range and bearing
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(range_residuals)), range_residuals, marker='o', label='Matched Range Residuals')
    plt.plot(range(len(bearing_residuals)), bearing_residuals, marker='x', label='Matched Bearing Residuals')
    plt.title('Measurement Residuals Over Time (Matched Observations)')
    plt.xlabel('Correction Step')
    plt.ylabel('Residual Value')
    plt.grid(True)
    plt.legend()

    # Save the plot
    plt.savefig(os.path.join(save_dir, 'measurement_residuals_matched.png'))
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
    
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_z_i_and_z_hat(data, save_dir):
    # Lists to hold range and bearing values for z_i (observed) and z_hat (predicted)
    z_i_range = []
    z_i_bearing = []
    z_hat_range = []
    z_hat_bearing = []

    # Iterate through the corrections in the data
    for correction in data:
        matched_observations = correction['correction'].get('Matched', {}).get('observations', [])
        
        # Only process matched observations
        if matched_observations:
            for obs in matched_observations:
                for landmark in obs['landmarks']:
                    z_i = np.array(landmark['z_i'])      # Observed measurement
                    z_hat = np.array(landmark['z_hat'])  # Predicted measurement
                    
                    # Append the range (z_i[0] and z_hat[0]) and bearing (z_i[1] and z_hat[1])
                    z_i_range.append(z_i[0])
                    z_hat_range.append(z_hat[0])
                    z_i_bearing.append(z_i[1])
                    z_hat_bearing.append(z_hat[1])

    # Convert to numpy arrays
    z_i_range = np.array(z_i_range)
    z_hat_range = np.array(z_hat_range)
    z_i_bearing = np.array(z_i_bearing)
    z_hat_bearing = np.array(z_hat_bearing)

    # Plot the Range (z_i and z_hat)
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(z_i_range)), z_i_range, marker='o', label='Observed Range (z_i)', linestyle='--')
    plt.plot(range(len(z_hat_range)), z_hat_range, marker='x', label='Predicted Range (z_hat)')
    plt.title('Range Observed vs Predicted (z_i vs z_hat)')
    plt.xlabel('Correction Step')
    plt.ylabel('Range Value')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'range_z_i_vs_z_hat.png'))
    plt.close()

    # Plot the Bearing (z_i and z_hat)
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(z_i_bearing)), z_i_bearing, marker='o', label='Observed Bearing (z_i)', linestyle='--')
    plt.plot(range(len(z_hat_bearing)), z_hat_bearing, marker='x', label='Predicted Bearing (z_hat)')
    plt.title('Bearing Observed vs Predicted (z_i vs z_hat)')
    plt.xlabel('Correction Step')
    plt.ylabel('Bearing Value')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'bearing_z_i_vs_z_hat.png'))
    plt.close()


def plot_covariance_growth_log_scale(data, save_dir):
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

    # Plot each state variance over time with logarithmic scale
    plt.figure(figsize=(10, 6))
    for i in range(padded_covariances.shape[1]):
        plt.plot(padded_covariances[:, i], marker='o', label=f'Variance of state {i+1}')
    
    plt.yscale('log')  # Set Y-axis to log scale
    plt.title('Variance Growth Over Time (Log Scale)')
    plt.xlabel('Correction Step')
    plt.ylabel('Log Variance')
    plt.grid(True)
    plt.legend()

    # Save the plot
    plt.savefig(os.path.join(save_dir, 'variance_growth_log_scale.png'))
    plt.close()

def plot_state_after_corrections(data, save_dir):
    correction_steps = []
    x_state = []
    y_state = []
    theta_state = []
    landmark_1_x_state = []
    landmark_1_y_state = []

    for correction in data:
        correction_number = correction['correction']['number']
        
        # Check if there is matched data in this correction
        matched_observations = correction['correction'].get('Matched', {}).get('observations', [])
        
        if matched_observations:
            final_state = correction['correction']['final_state']
            correction_steps.append(correction_number)
            # Append robot state (x, y, theta)
            x_state.append(final_state[0][0])  # x
            y_state.append(final_state[1][0])  # y
            theta_state.append(final_state[2][0])  # theta
            
            # Append the first landmark state (x, y) if it exists
            if len(final_state) > 3:
                landmark_1_x_state.append(final_state[3][0])  # landmark_1_x
                landmark_1_y_state.append(final_state[4][0])  # landmark_1_y
            else:
                # In case there's no landmark information
                landmark_1_x_state.append(np.nan)  # No landmark, append NaN
                landmark_1_y_state.append(np.nan)  # No landmark, append NaN

    # Plotting each state variable
    plt.figure(figsize=(10, 6))
    
    plt.plot(correction_steps, x_state, label='x (robot)', marker='o')
    plt.plot(correction_steps, y_state, label='y (robot)', marker='o')
    plt.plot(correction_steps, theta_state, label='theta (robot)', marker='o')
    plt.plot(correction_steps, landmark_1_x_state, label='Landmark 1 x', marker='x')
    plt.plot(correction_steps, landmark_1_y_state, label='Landmark 1 y', marker='x')
    
    # Adding the linear line: y = m * (x - 10) + 0, where m is the slope
    m = 0.1  # Define the slope, you can change this value to adjust the slope
    x_line = np.array(correction_steps)
    y_line = m * (x_line - 12)  # Linear equation with intercept at (10, 0)

    plt.plot(x_line, y_line, label='Linear Line (y = m(x-10))', linestyle='--', color='black')

    plt.title('State After Each Correction (Matched Observations)')
    plt.xlabel('Correction Step')
    plt.ylabel('State Value')
    plt.grid(True)
    plt.legend()

    # Save the plot
    plt.savefig(os.path.join(save_dir, 'state_after_each_correction.png'))
    plt.close()

def plot_pi_values(data, save_dir):
    # List to hold pi values for all observations
    pi_values = []
    observation_ids = []
    
    # Iterate through the corrections in the data
    for correction in data:
        observations = correction['correction']['All']['observations']
        
        for obs in observations:
            for landmark in obs['landmarks']:
                pi_value = landmark['pi']
                pi_values.append(pi_value)
                observation_ids.append(obs['observation_id'])
    
    # Convert to numpy arrays for plotting
    pi_values = np.array(pi_values)
    observation_ids = np.array(observation_ids)

    # Plot pi values
    plt.figure(figsize=(10, 6))
    plt.plot(observation_ids, pi_values, marker='o', label='Pi Values')
    plt.title('Pi Values for All Observations')
    plt.xlabel('Observation ID')
    plt.ylabel('Pi Value')
    plt.grid(True)
    plt.legend()

    # Save the plot
    plt.savefig(os.path.join(save_dir, 'pi_values.png'))
    plt.close()

    print(f"Pi values plot saved in {save_dir}")
    
def visualize_ransac_from_json(data, save_dir):
    """
    Visualizes the RANSAC lines with inliers and outliers for each iteration based on the stored data in the JSON file.
    """
    save_dir = os.path.join(save_dir, "RansacFeatureAnaysis")
    create_directory(save_dir)
    
    clear_directory(save_dir)

    for correction in data:
        if "features" in correction['correction'] and "lines" in correction['correction']['features']:
            lines_data = correction['correction']['features']['lines']

            for line_data in lines_data:
                iteration = line_data['iteration']
                loopCounter = line_data['loopCounter']
                slope = line_data['slope']
                intercept = line_data['intercept']
                inliers = np.array(line_data['inliers'])
                outliers = np.array(line_data['outliers'])

                # Create a new figure for each iteration
                plt.figure(figsize=(10, 10))

                # Plot inliers in blue
                plt.scatter(inliers[:, 0], inliers[:, 1], c='blue', label='Inliers')

                # Plot outliers in red
                plt.scatter(outliers[:, 0], outliers[:, 1], c='red', label='Outliers')

                # Plot the detected line
                x_vals = np.array([inliers[:, 0].min(), inliers[:, 0].max()])  # x-range for the line
                y_vals = slope * x_vals + intercept  # y = mx + b for the line
                plt.plot(x_vals, y_vals, 'g-', linewidth=2, label=f'Line: y={slope:.2f}x+{intercept:.2f}')

                # Set plot details
                plt.title(f'RANSAC Line Detection - Loop {loopCounter} - Iteration {iteration}')
                plt.xlabel('X [meters]')
                plt.ylabel('Y [meters]')
                plt.legend()
                plt.grid(True)
                plt.axis('equal')

                # Save the plot for this iteration
                filename = f'ransac_loop_{loopCounter}_iteration_{iteration}.png'
                filepath = os.path.join(save_dir, filename)
                plt.savefig(filepath)
                plt.close()

                # print(f"Saved plot for Loop {loopCounter}, Iteration {iteration} to {filepath}")


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
    
    plot_residuals_magnitude_matched(data, output_dir)
    
    plot_residuals_matched(data, output_dir)
    
    plot_z_i_and_z_hat(data, output_dir)

    # Analyze and plot covariance growth
    plot_covariance_growth(data, output_dir)
    
    # Analyze and plot covariance growth with log scaling
    plot_covariance_growth_log_scale(data, output_dir)
    
    # Analyze and plot state updates
    plot_state_updates_per_correction(data, output_dir)
    
    plot_state_after_corrections(data, output_dir)
    
    plot_pi_values(data, output_dir)
    
    # visualize_ransac_from_json(data, output_dir)

    print(f'Analysis complete. Plots saved in {output_dir}')

# Example usage
if __name__ == "__main__":
    # Define file paths
    json_file_path = '/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/data/correctionData.json'  
    output_directory = '/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/correctionAnalysisOutput'

    # Run the analysis
    analyze_ekf_slam(json_file_path, output_directory)
