# Spezialisierung-2

This repository, **Spezialisierung-2**, is developed by [CoCoKayya18](https://github.com/CoCoKayya18). The project is primarily written in Python, with additional components in CMake.

## Project Overview

**Spezialisierung-2** is structured to facilitate modular development and ease of maintenance. The repository includes:

- **Source Code (`src/`):** Contains the main Python scripts and modules.
- **EKF SLAM Code (`include/ekf_slam_pkg/`):** Contains all the scripts implementing the EKF SLAM
- **Plots (`plots/`):** Directory for generated plots and visualizations.
- **Configuration Files:**
  - `.gitignore`: Specifies files and directories to be ignored by Git.
- **Documentation:**
  - `README.md`: Provides an overview and instructions for the project.

## Getting Started

To set up and run this project locally, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/CoCoKayya18/Spezialisierung-2.git
   ```

2. Navigate to the Project Directory:

   ```bash
   cd Spezialisierung-2
   ```
   
3. Set Up the Environment:

   - Ensure Python is installed on your system.

   - Install necessary dependencies:

4. Build the Project:
   catkin_make

5. Adjust paths
   - As this directory works with absolute paths, and not relative ones (for some reason, they didnt work), adjust the paths accordingly beforehand

7. Execute the program
  - For a single launch, use the launch file "run_slam.launch" in /launch
  - For multiple runs: Modify the "testSlamFramework.py" in /src and adjust the number of runs and the duration accordingly


