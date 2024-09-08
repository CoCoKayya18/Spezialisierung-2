import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import GPy
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import seaborn as sns

def load_scalers(scaler_x_path, scaler_y_path):
    """Load the scalers for input and output data."""
    with open(scaler_x_path, 'rb') as file:
        scaler_X = pickle.load(file)
    with open(scaler_y_path, 'rb') as file:
        scaler_Y = pickle.load(file)
    return scaler_X, scaler_Y

def load_model(model_path):
    """Load the pre-trained GPy model."""
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
        print(model)
    return model

def predict_path(odom_velocities, scaler_X, scaler_Y, model):
    """Predict the path using the model and scalers."""
    # Scale the input data
    X_test_scaled = scaler_X.transform(odom_velocities)
    
    # Predict using the model
    Y_pred_scaled, _ = model.predict(X_test_scaled)
    
    # Inverse transform to get the original scale
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
    return Y_pred

def visualize_paths_with_check_buttons(ground_truth, predicted_path, plot_dir):
    """Plot the ground truth path against the predicted path with check buttons to toggle visibility."""
    os.makedirs(plot_dir, exist_ok=True)

    # Check if the data is already cumulative or contains unexpected values
    # print("Ground Truth Data Head:\n", ground_truth[:10])  # Print first 10 entries to inspect
    # print("Predicted Path Data Head:\n", predicted_path[:10])  # Print first 10 entries to inspect

    # Plot paths
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot Ground Truth Path (assuming data is not cumulative already)
    gt_line, = ax.plot(ground_truth[:, 0], ground_truth[:, 1],
                       label='Ground Truth Path', color='blue')
    
    # Plot Predicted Path
    pred_line, = ax.plot(np.cumsum(predicted_path[:, 0]), np.cumsum(predicted_path[:, 1]),
                         label='Predicted Path', color='red', linestyle='dashed')
    
    ax.set_title('Ground Truth vs Predicted Path')
    ax.set_xlabel('X Position (meters)')
    ax.set_ylabel('Y Position (meters)')
    ax.legend()
    plt.grid(True)

    # Add check buttons for toggling visibility
    rax = plt.axes([0.85, 0.4, 0.1, 0.15])  # Position of the check buttons
    check = CheckButtons(rax, ['Ground Truth Path', 'Predicted Path'], [True, True])

    # Function to toggle visibility
    def toggle_visibility(label):
        if label == 'Ground Truth Path':
            gt_line.set_visible(not gt_line.get_visible())
        elif label == 'Predicted Path':
            pred_line.set_visible(not pred_line.get_visible())
        plt.draw()

    # Connect check buttons to the function
    check.on_clicked(toggle_visibility)

    # Adjust layout to prevent overlap
    plt.subplots_adjust(left=0.1, right=0.8)
    
    # Save plot
    plt.savefig(os.path.join(plot_dir, 'path_comparison.png'))
    plt.show()

def calculate_cumulative_path(deltas):
    """Calculate the cumulative path from delta values."""
    # Calculate the cumulative sum of deltas
    cumulative_path = np.cumsum(deltas, axis=0)
    return cumulative_path

def main(use_old_test_data=False):
    # Define file paths
    odom_velocities_path = '../Spezialisierung-2/src/ekf_slam_pkg/data/odom_velocities.csv'  # Replace with the path to your odom_velocities.csv
    ground_truth_path = '../Spezialisierung-2/src/ekf_slam_pkg/data/ground_truth_path.csv'  # Replace with the path to your ground_truth_path.csv
    old_test_data_path = '../Spezialisierung-1/src/slam_pkg/data/square_single/training/FullData/odomVel/sparse_test_data_square_odometry_single.csv'  # Replace with the path to your old test data CSV
    scaler_x_path = '../Spezialisierung-2/src/ekf_slam_pkg/Scaler/sparse_scaler_X_random3_odometry.pkl'  # Replace with the path to your scaler X file
    scaler_y_path = '../Spezialisierung-2/src/ekf_slam_pkg/Scaler/sparse_scaler_Y_random3_odometry.pkl'  # Replace with the path to your scaler Y file
    model_path = '../Spezialisierung-2/src/ekf_slam_pkg/myMLModel/sparse_gpy_model_random3_odometry.pkl'  # Replace with the path to your trained model file
    plot_dir = 'plots'  # Directory to save plots

    # Load data based on the boolean flag
    if use_old_test_data:
        # Load old test data
        old_test_data = pd.read_csv(old_test_data_path)
        odom_velocities = old_test_data[['odom_world_velocity_x', 'odom_world_velocity_y', 'odom_angular_velocity']].values
        deltas = old_test_data[['delta_position_x_world', 'delta_position_y_world', 'delta_yaw']].values
        
        # Calculate cumulative path for ground truth
        ground_truth = calculate_cumulative_path(deltas)
    else:
        # Load new test data
        odom_velocities = pd.read_csv(odom_velocities_path)[['linear_x', 'linear_y', 'angular_z']].values
        ground_truth = pd.read_csv(ground_truth_path)[['x', 'y', 'z']].values

    # Load scalers and model
    scaler_X, scaler_Y = load_scalers(scaler_x_path, scaler_y_path)
    model = load_model(model_path)

    # Predict path using the model
    predicted_path = predict_path(odom_velocities, scaler_X, scaler_Y, model)

    # Visualize the predicted path vs ground truth path with interactive check buttons
    visualize_paths_with_check_buttons(ground_truth, predicted_path, plot_dir)

if __name__ == '__main__':
    # Set the boolean flag to True or False to switch between datasets
    use_old_test_data = True  # Set to True to use the old test data
    main(use_old_test_data)