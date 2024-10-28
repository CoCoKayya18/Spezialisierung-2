import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
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


def create_directory(directory_name):
    # Check if the directory exists
    if os.path.exists(directory_name):
        # Clear all contents of the directory
        shutil.rmtree(directory_name)
    
    # Create a fresh directory
    os.makedirs(directory_name)

# Function to load JSON data
def load_json_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        # Read the entire content and split by double newlines, assuming this separates JSON objects
        content = file.read()
        json_blocks = content.strip().split('\n\n')

        for block in json_blocks:
            try:
                # Parse each block as a JSON object
                json_obj = json.loads(block)
                data.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON block: {e}")
    
    return data

# Function to visualize state updates over time
def plot_state_updates_per_correction(data, save_dir, threshold=0.03):
    state_updates = []
    high_state_updates = []
    high_state_update_indices = []
    sum_of_updates = None  # Initialize to None, will set it to the size of the state when available
    max_state_length = 0  # Track the maximum state vector length

    for correction_idx, correction in enumerate(data):
        matched_observations = correction['correction'].get('Matched', {}).get('observations', [])
        
        # Check if there is a matched observation with a state update
        if matched_observations:
            for obs_idx, obs in enumerate(matched_observations):
                state_update = obs['landmarks'][0].get('State update', None)

                if state_update:
                    state_update = np.array(state_update).flatten()  # Convert to 1D array
                    current_state_length = len(state_update)
                    
                    # Update the max_state_length if the current state is longer
                    if current_state_length > max_state_length:
                        # Pad all previous state updates to match the new max_state_length
                        state_updates = [np.pad(su, (0, current_state_length - len(su)), 'constant') for su in state_updates]
                        if sum_of_updates is not None:
                            sum_of_updates = np.pad(sum_of_updates, (0, current_state_length - len(sum_of_updates)), 'constant')
                        max_state_length = current_state_length
                    
                    # Initialize sum_of_updates if it hasn't been done yet
                    if sum_of_updates is None:
                        sum_of_updates = np.zeros(current_state_length)

                    # Pad the state_update if it's smaller than the current max length
                    if current_state_length < max_state_length:
                        state_update = np.pad(state_update, (0, max_state_length - current_state_length), 'constant')

                    # Append the state update to state_updates
                    state_updates.append(state_update)
                    
                    # Add the state update to the sum_of_updates
                    sum_of_updates += state_update

                    # Check if the first three entries exceed the threshold
                    if any(abs(state_update[:3]) > threshold):
                        high_state_updates.append(state_update[:3])  # Only store the first three entries
                        high_state_update_indices.append({
                            "correction_step": correction_idx,
                            "observation_id": obs["observation_id"],
                            "landmark_id": obs['landmarks'][0]['landmark_id'],
                            "state_update": state_update[:3]
                        })
        else:
            # Skip corrections with no state update (e.g., newly detected landmarks)
            continue

    # Check if state updates exist
    if state_updates:
        # Convert state_updates to a 2D array, all rows having the same length
        state_updates = np.vstack(state_updates)

        # Plot each state update over time
        plt.figure(figsize=(10, 6))
        for i in range(state_updates.shape[1]):
            plt.plot(range(1, len(state_updates) + 1), state_updates[:, i], marker='o', label=f'State {i+1} Update')

        # Add the sum of all updates as a text box on the plot
        sum_text = f"Sum of all updates:\n{sum_of_updates[:3]}"  # Display only the sum for robot state
        plt.gcf().text(0.15, 0.75, sum_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

        plt.title('State Updates Per Correction Step')
        plt.xlabel('Correction Step')
        plt.ylabel('State Update Value')
        plt.grid(True)
        plt.legend()

        # Save the plot
        plt.savefig(os.path.join(save_dir, 'state_updates_per_correction.png'))
        plt.close()

        # Print the sum of all updates for reference in the console
        print("Sum of all state updates:", sum_of_updates)
    else:
        print("No state updates available for plotting.")

    # Plot high state updates (first three entries only)
    if high_state_updates:
        high_state_updates = np.array(high_state_updates)

        plt.figure(figsize=(10, 6))
        for i in range(3):  # Only the first three entries (robot state)
            plt.plot(range(1, len(high_state_updates) + 1), high_state_updates[:, i], marker='o', label=f'High State {i+1} Update')

        plt.title(f'High State Updates Per Correction Step (Threshold: {threshold})')
        plt.xlabel('Correction Step')
        plt.ylabel('State Update Value')
        plt.grid(True)
        plt.legend()

        # Save the plot
        plt.savefig(os.path.join(save_dir, 'high_state_updates_per_correction.png'))
        plt.close()

        # Print high state update indices for reference in the console
        print("High state updates (correction step, observation ID, landmark ID):")
        for high_update in high_state_update_indices:
            print(high_update)
    else:
        print("No high state updates found.")
    
# Function to visualize summed state updates (x, y, theta) over time
def plot_summed_state_updates_per_correction(data, save_dir, threshold=0.03):
    state_updates_per_correction = []
    high_state_updates = []
    high_state_update_indices = []
    sum_of_updates = np.zeros(3)  # Track sum of all updates for x, y, theta
    max_state_length = 3  # Track the maximum state vector length (x, y, theta)

    for correction_idx, correction in enumerate(data):
        matched_observations = correction['correction'].get('Matched', {}).get('observations', [])
        correction_sum = np.zeros(3)  # Summed update for this correction step (x, y, theta)

        # Check if there is a matched observation with a state update
        if matched_observations:
            for obs_idx, obs in enumerate(matched_observations):
                state_update = obs['landmarks'][0].get('State update', None)

                if state_update:
                    state_update = np.array(state_update).flatten()  # Convert to 1D array

                    # Only consider the first three entries for x, y, theta
                    state_update_xytheta = state_update[:3]

                    # Add this observation's update to the correction sum
                    correction_sum += state_update_xytheta

            # Append the total update for this correction step to the list
            state_updates_per_correction.append(correction_sum)
            
            # Add the correction sum to the global sum_of_updates
            sum_of_updates += correction_sum

            # Check if the first three entries exceed the threshold
            if any(abs(correction_sum[:3]) > threshold):
                high_state_updates.append(correction_sum[:3])  # Only store the first three entries (x, y, theta)
                high_state_update_indices.append({
                    "correction_step": correction_idx,
                    "total_state_update": correction_sum[:3]
                })
        else:
            # Skip corrections with no state update (e.g., newly detected landmarks)
            continue

    # Check if state updates exist
    if state_updates_per_correction:
        # Convert state_updates_per_correction to a 2D array
        state_updates_per_correction = np.vstack(state_updates_per_correction)

        # Plot each summed state update (x, y, theta) over time
        plt.figure(figsize=(10, 6))
        labels = ['X State Update', 'Y State Update', 'Theta State Update']
        for i in range(3):  # Only the first three entries (x, y, theta)
            plt.plot(range(1, len(state_updates_per_correction) + 1), state_updates_per_correction[:, i], 
                     marker='o', label=labels[i])

        # Add the sum of all updates as a text box on the plot
        sum_text = f"Sum of all updates (x, y, theta):\n{sum_of_updates}"  # Display only the sum for robot state
        plt.gcf().text(0.15, 0.75, sum_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

        plt.title('Summed State Updates Per Correction Step (X, Y, Theta)')
        plt.xlabel('Correction Step')
        plt.ylabel('State Update Value')
        plt.grid(True)
        plt.legend()

        # Save the plot
        plt.savefig(os.path.join(save_dir, 'summed_robot_state_updates_per_correction.png'))
        plt.close()

        # Print the sum of all updates for reference in the console
        print("Sum of all state updates (x, y, theta):", sum_of_updates)
    else:
        print("No state updates available for plotting.")

    # Plot high state updates (first three entries only)
    if high_state_updates:
        high_state_updates = np.array(high_state_updates)

        plt.figure(figsize=(10, 6))
        for i in range(3):  # Only the first three entries (x, y, theta)
            plt.plot(range(1, len(high_state_updates) + 1), high_state_updates[:, i], 
                     marker='o', label=f'High {labels[i]} Update')

        plt.title(f'High State Updates Per Correction Step (Threshold: {threshold})')
        plt.xlabel('Correction Step')
        plt.ylabel('State Update Value')
        plt.grid(True)
        plt.legend()

        # Save the plot
        plt.savefig(os.path.join(save_dir, 'high_state_updates_per_correction.png'))
        plt.close()

        # Print high state update indices for reference in the console
        print("High state updates (correction step, total state update):")
        for high_update in high_state_update_indices:
            print(high_update)
    else:
        print("No high state updates found.")


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
    high_range_residuals = []
    high_bearing_residuals = []
    high_residual_indices = []
    
    threshold=0.18

    for idx, correction in enumerate(data):
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
                    
                    # Check if either residual is above the threshold
                    if abs(range_residual) > threshold or abs(bearing_residual) > threshold:
                        high_range_residuals.append(range_residual)
                        high_bearing_residuals.append(bearing_residual)
                        high_residual_indices.append((idx, obs["observation_id"]))  # Capture the index and observation ID

    range_residuals = np.array(range_residuals)
    bearing_residuals = np.array(bearing_residuals)
    high_range_residuals = np.array(high_range_residuals)
    high_bearing_residuals = np.array(high_bearing_residuals)

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

    # Plot only high residuals
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(high_range_residuals)), high_range_residuals, marker='o', label='High Range Residuals')
    plt.plot(range(len(high_bearing_residuals)), high_bearing_residuals, marker='x', label='High Bearing Residuals')
    plt.title(f'Matched Measurement Residuals Above Threshold {threshold}')
    plt.xlabel('Correction Step')
    plt.ylabel('Residual Value')
    plt.grid(True)
    plt.legend()

    # Save the plot for high residuals
    plt.savefig(os.path.join(save_dir, 'high_residuals_matched.png'))
    plt.close()

    # Output high residual indices
    print("Indices of high residuals (correction step, observation ID):")
    for index in high_residual_indices:
        print(index)


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
    plt.figure(figsize=(12, 8))  # Larger figure for better visibility

    # Prepare lists to hold plotted elements for toggling visibility later
    group_plots = []
    labels = []

    # Loop through each correction and group landmarks by correction
    for idx, correction in enumerate(data):
        final_state = np.array(correction['correction']['final_state'])
        
        # Extract robot and landmark positions
        robot_x = final_state[0]  # assuming robot's x is at index 0
        robot_y = final_state[1]  # assuming robot's y is at index 1
        theta = final_state[2]    # assuming robot's orientation is at index 2
        
        # Assuming landmarks start from index 3 and alternate between x and y positions
        landmarks_x = final_state[3::2]
        landmarks_y = final_state[4::2]

        # Plot robot position
        plt.scatter(robot_x, robot_y, label=f'Robot (Correction {idx})', marker='o', color='C0')
        
        # Plot landmarks as a group for each correction
        landmark_group, = plt.plot(landmarks_x, landmarks_y, 'x', label=f'Landmarks (Correction {idx})', visible=True)
        group_plots.append(landmark_group)
        labels.append(f'Landmarks (Correction {idx})')

    # Add labels, title, grid, and legend
    plt.title('State After Each Correction (Grouped Landmarks)', fontsize=14)
    plt.xlabel('State Value X', fontsize=12)
    plt.ylabel('State Value Y', fontsize=12)
    plt.grid(True)

    # Create checkboxes for toggling groups of landmarks (grouped by correction)
    rax = plt.axes([0.82, 0.2, 0.15, 0.6])  # Adjust the position and size of the checkbox
    check = CheckButtons(rax, labels, [True] * len(labels))

    # Define callback for toggling visibility
    def toggle_visibility(label):
        index = labels.index(label)
        plot_element = group_plots[index]
        plot_element.set_visible(not plot_element.get_visible())
        plt.draw()

    # Connect the checkbox to the visibility toggle function
    check.on_clicked(toggle_visibility)

    # Adjust layout to make room for checkboxes
    plt.subplots_adjust(left=0.1, right=0.75)  # Adjust margins to fit the checkbox

    # Show plot with interactivity
    # plt.show()

    # Optionally save the plot if a save directory is provided
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'grouped_landmarks_states.png'))
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
                
def analyze_anomaly_step(data, anomaly_step):

    correction = data[anomaly_step]
    print(f"Analyzing correction step: {anomaly_step}")
    
    # Extract data for analysis
    kalman_gain = correction['correction'].get('Kalman Gain')
    state_update = correction['correction'].get('State update')
    measurement_residual = correction['correction'].get('Measurement residual')
    final_covariance = correction['correction'].get('Final Covariance')
    
    # Print or plot relevant data
    print(f"Kalman Gain at step {anomaly_step}: {kalman_gain}")
    print(f"State Update at step {anomaly_step}: {state_update}")
    print(f"Measurement Residual at step {anomaly_step}: {measurement_residual}")
    print(f"Final Covariance at step {anomaly_step}: {final_covariance}")

def find_high_bearing_error(data, threshold=1.0):

    high_error_measurements = []

    for step_index, correction in enumerate(data):
        observations = correction['correction']['Matched']['observations']
        for obs in observations:
            for landmark in obs['landmarks']:
                bearing_residual = landmark['measurement_residual'][1]  # Get the bearing component
                if abs(bearing_residual) > threshold:
                    high_error_measurements.append({
                        "correction_step": step_index,
                        "observation_id": obs["observation_id"],
                        "bearing_residual": bearing_residual
                    })
    
    print(f"High errors: {high_error_measurements}")

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
    
    plot_summed_state_updates_per_correction(data, output_dir)
    
    plot_state_after_corrections(data, output_dir)
    
    plot_pi_values(data, output_dir)
    
    analyze_anomaly_step(data, 45)
    
    find_high_bearing_error(data, threshold=0.5)
    
    # visualize_ransac_from_json(data, output_dir)

    print(f'Analysis complete. Plots saved in {output_dir}')

# Example usage
if __name__ == "__main__":
    # Define file paths
    json_file_path = '/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/data/correction_data.json'  
    output_directory = '/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/correctionAnalysisOutput'

    # Run the analysis
    analyze_ekf_slam(json_file_path, output_directory)
