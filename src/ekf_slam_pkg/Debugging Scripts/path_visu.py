

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

# Load the data from CSV files
ekf_data_path = '/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/data/ekf_path.csv'  # Replace with the path to your EKF data CSV file
ground_truth_data_path = '/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/data/ground_truth_path.csv'  # Replace with the path to your ground truth data CSV file

# Read EKF data
ekf_df = pd.read_csv(ekf_data_path)

# Read ground truth data
ground_truth_df = pd.read_csv(ground_truth_data_path)

# Plotting the paths
fig, ax = plt.subplots(figsize=(10, 6))

# Plot EKF path
ekf_line, = ax.plot(ekf_df['x'], ekf_df['y'], label='EKF Path', color='blue', marker='o', markersize=4, linestyle='-', linewidth=1)

# Plot ground truth path
ground_truth_line, = ax.plot(ground_truth_df['x'], ground_truth_df['y'], label='Ground Truth Path', color='green', marker='x', markersize=4, linestyle='-', linewidth=1)

# Set labels and title
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_title('Comparison of Ground Truth Path and EKF Path')

# Add a legend
ax.legend()

# Add a grid
ax.grid(True)

# Create check buttons for toggling visibility
rax = plt.axes([0.85, 0.4, 0.1, 0.15])  # Position of the check buttons
check = CheckButtons(rax, ['EKF Path', 'Ground Truth Path'], [True, True])

# Function to toggle visibility
def toggle_visibility(label):
    if label == 'EKF Path':
        ekf_line.set_visible(not ekf_line.get_visible())
    elif label == 'Ground Truth Path':
        ground_truth_line.set_visible(not ground_truth_line.get_visible())
    plt.draw()

# Connect check buttons to the function
check.on_clicked(toggle_visibility)

# Adjust layout to prevent overlap
plt.subplots_adjust(left=0.1, right=0.8)

# Show the plot
plt.show()
