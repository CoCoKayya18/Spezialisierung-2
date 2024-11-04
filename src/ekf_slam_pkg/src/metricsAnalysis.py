import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
import numpy as np
import glob
import os

# File paths
evaluation_csv = "/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/slamPlotterEvaluationPlots/SlamPlotterEvaluation.csv"
nees_csv_folder = "/home/ubuntu/Spezialisierung-2/src/ekf_slam_pkg/slamPlotterEvaluationPlots"

# Load the main evaluation data
try:
    data = pd.read_csv(evaluation_csv)
    ate_values = data["ATE"].dropna().values
    mean_nees_values = data["Mean_NEES"].dropna().values
except Exception as e:
    print(f"Failed to load data: {e}")
    exit(1)

# Calculate statistics for ATE and NEES
ate_q1, ate_q3 = np.percentile(ate_values, [25, 75])
ate_mean = np.mean(ate_values)
ate_median = np.median(ate_values)  # Add median calculation
nees_q1, nees_q3 = np.percentile(mean_nees_values, [25, 75])
nees_mean = np.mean(mean_nees_values)
nees_median = np.median(mean_nees_values)  # Add median calculation

# Boxplots for ATE and NEES with annotated statistics
plt.figure(figsize=(10, 7))

# ATE Boxplot
plt.subplot(1, 2, 1)
plt.boxplot(ate_values)
plt.title("ATE Boxplot")
plt.ylabel("ATE")
# Add lines and text for Q1, Q3, median, and mean
plt.axhline(ate_q1, color='purple', linestyle='--', label=f'Q1: {ate_q1:.2f}')
plt.axhline(ate_q3, color='green', linestyle='--', label=f'Q3: {ate_q3:.2f}')
plt.axhline(ate_median, color='orange', linestyle='--', label=f'Median: {ate_median:.2f}')
plt.legend()

# NEES Boxplot
plt.subplot(1, 2, 2)
plt.boxplot(mean_nees_values)
plt.title("NEES Boxplot")
plt.ylabel("NEES")
# Add lines and text for Q1, Q3, median, and mean
plt.axhline(nees_q1, color='purple', linestyle='--', label=f'Q1: {nees_q1:.2f}')
plt.axhline(nees_q3, color='green', linestyle='--', label=f'Q3: {nees_q3:.2f}')
plt.axhline(nees_median, color='orange', linestyle='--', label=f'Median: {nees_median:.2f}')
plt.legend()

# Save and show the plot
plt.tight_layout()
plt.savefig(f"{nees_csv_folder}/ATE_NEES_Boxplots_with_stats_and_median.png")
print("Saved ATE and NEES boxplots with statistics as ATE_NEES_Boxplots_with_stats_and_median.png.")
plt.close()

plt.figure(figsize=(10, 7))
# ATE Boxplot
plt.subplot(1, 2, 1)
plt.boxplot(ate_values)
plt.title("ATE Boxplot")
plt.ylabel("ATE")
# Add lines and text for Q1, Q3, median, and mean
plt.axhline(ate_q1, color='purple', linestyle='--', label=f'Q1: {ate_q1:.2f}')
plt.axhline(ate_q3, color='green', linestyle='--', label=f'Q3: {ate_q3:.2f}')
plt.axhline(ate_median, color='orange', linestyle='--', label=f'Median: {ate_median:.2f}')
plt.axhline(ate_mean, color='blue', linestyle='-', label=f'Mean: {ate_mean:.2f}')
plt.legend()

# NEES Boxplot
plt.subplot(1, 2, 2)
plt.boxplot(mean_nees_values)
plt.title("NEES Boxplot")
plt.ylabel("NEES")
# Add lines and text for Q1, Q3, median, and mean
plt.axhline(nees_q1, color='purple', linestyle='--', label=f'Q1: {nees_q1:.2f}')
plt.axhline(nees_q3, color='green', linestyle='--', label=f'Q3: {nees_q3:.2f}')
plt.axhline(nees_median, color='orange', linestyle='--', label=f'Median: {nees_median:.2f}')
plt.axhline(nees_mean, color='blue', linestyle='-', label=f'Mean: {nees_mean:.2f}')
plt.legend()

# Save and show the plot
plt.tight_layout()
plt.savefig(f"{nees_csv_folder}/ATE_NEES_Boxplots_with_stats_and_median_and_mean.png")
print("Saved ATE and NEES boxplots with statistics as ATE_NEES_Boxplots_with_stats_and_median_and_mean.png.")
plt.close()


# Recursively search for NEES CSV files within each Run_* folder
nees_files = sorted(glob.glob(f"{nees_csv_folder}/Run_*/Run_*_NEES_values.csv"))

# Parameters for NEES consistency test
degrees_of_freedom = 2  # Since NEES considers x and y errors
confidence_level = 0.95

# Chi-square bounds for consistency region
upper_bound = chi2.ppf((1 + confidence_level) / 2, degrees_of_freedom)
lower_bound = chi2.ppf((1 - confidence_level) / 2, degrees_of_freedom)

# Loop through each NEES file and create a plot in the corresponding Run folder
for nees_file in nees_files:
    try:
        # Load NEES values for the current run
        nees_data = pd.read_csv(nees_file)["NEES Value"].dropna().values
        
        # Ensure there are NEES values to plot
        if len(nees_data) == 0:
            print(f"No NEES values found in {nees_file}. Skipping...")
            continue

        # Extract the folder path to save the plot in the correct Run_* directory
        run_folder = os.path.dirname(nees_file)
        output_filename = os.path.join(run_folder, "NEES_TimeSeries_Consistency.png")
        
        # Plot NEES values over time with consistency bounds
        plt.figure(figsize=(10, 6))
        plt.plot(nees_data, label="NEES values", color="blue")
        plt.axhline(upper_bound, color="red", linestyle="--", label=f"Upper 95% bound ({upper_bound:.2f})")
        plt.axhline(lower_bound, color="orange", linestyle="--", label=f"Lower 95% bound ({lower_bound:.2f})")
        plt.title("Normalized Estimation Error Squared (NEES) Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("NEES Value")
        plt.legend()
        plt.grid(True)
        
        # Save plot in the corresponding Run_* folder
        plt.savefig(output_filename)
        print(f"Saved NEES time series plot with consistency bounds for {run_folder} as {output_filename}.")
        plt.close()

    except Exception as e:
        print(f"Failed to load or plot NEES data from {nees_file}: {e}")
