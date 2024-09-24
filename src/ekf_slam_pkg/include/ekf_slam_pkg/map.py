import rospy
import numpy as np

class Map:
    def __init__(self, config):
        self.landmarks = []
        self.landmarkNumber = 0

        self.alpha = 5.991 # 95% confidence based on Chi-squared distribution
        rospy.loginfo("Map class initialized")

    def calculate_landmark_estimates(self, x, y, theta, z_i):
        """
        Update the landmark estimates given the robot's current pose and the observation.
        """
        r_i, phi_i = z_i
        mu_N_plus_1_x = x + r_i * np.cos(phi_i + theta)
        mu_N_plus_1_y = y + r_i * np.sin(phi_i + theta)
        return mu_N_plus_1_x, mu_N_plus_1_y

    def compute_F_x_k(self, num_landmarks, k):
    
         # Total state size: 3 for robot pose + 2 * n_landmarks for landmarks
        state_size = 3 + 2 * num_landmarks  # This ensures the matrix is wide enough to include the robot + landmarks

        # Initialize F_x_k as a zeros matrix of size (5, state_size)
        F_x_k = np.zeros((5, state_size))

        # Add the 3x3 identity matrix for the robot's pose (x, y, theta)
        F_x_k[:3, :3] = np.eye(3)

        # Add the 2x2 identity matrix for the specific landmark `k`
        landmark_offset = 3 + 2 * (k - 1)  # Landmark's position in the state vector
        F_x_k[3:, landmark_offset:landmark_offset + 2] = np.eye(2)

        return F_x_k

    def compute_H_k_t(self, delta_k, q_k, F_x_k):
        """
        Computes the H_k_t matrix.
        """
        # print(f"\n F_x_k : {F_x_k.shape}\n{F_x_k}")

        # As i am not using any signatures for the landmark, the H matrix gets reduced
        H_k_t = (1 / q_k) * np.array([
            [np.sqrt(q_k) * delta_k[0].item(), -np.sqrt(q_k) * delta_k[1].item(), 0, -np.sqrt(q_k) * delta_k[0].item(), np.sqrt(q_k) * delta_k[1].item()],
            [delta_k[1].item(), delta_k[0].item(), -1, -delta_k[1].item(), -delta_k[0].item()]
        ]) 

        H_k_t = H_k_t @ F_x_k

        return H_k_t

    def compute_mahalanobis_distance(self, z_i, z_hat_k, H_k_t, Sigma_t, measurement_noise):
        """
        Computes the Mahalanobis distance for data association.
        """

        # Print the dimensions and values of the inputs
        # print("\n==== Mahalanobis Distance Computation ====")
        
        # print(f"\nz_i (measurement): {z_i.shape}\n{z_i}")
        # print(f"\nz_hat_k (predicted measurement): {z_hat_k.shape}\n{z_hat_k}")
        # print(f"\nH_k_t (Jacobian): {H_k_t.shape}\n{H_k_t}")
        # print(f"\nH_k_t (Jacobian) transposed: {H_k_t.T.shape}\n{H_k_t.T}")
        # print(f"\nSigma_t (covariance matrix): {Sigma_t.shape}\n{Sigma_t}")
        # print(f"\nmeasurement_noise: {measurement_noise.shape}\n{measurement_noise}")

        Psi_k = H_k_t @ Sigma_t @ H_k_t.T + measurement_noise
        pi_k = (z_i - z_hat_k).T @ np.linalg.inv(Psi_k) @ (z_i - z_hat_k)
        return pi_k, Psi_k

    def data_association(self, pi_k):
        """
        Performs data association and returns the associated landmark index.
        """
        if np.min(pi_k) <= self.alpha:
            return np.argmin(pi_k)
        return None

    def update_state(self, x, y, theta, z_i, z_hat_k, K_i_t):
        """
        Updates the state (x, y, theta) given the Kalman gain and measurement residual.
        """
        state_update = K_i_t @ (z_i - z_hat_k)
        x += state_update[0]
        y += state_update[1]
        theta += state_update[2]
        return x, y, theta
    
    def get_landmarks(self, state_vector):

        landmarks = []
        
        # Ensure state vector has landmarks
        if len(state_vector) > 3:
            # Extract landmarks from state vector
            for i in range(3, len(state_vector), 2):
                x_landmark = state_vector[i]
                y_landmark = state_vector[i + 1]
                landmarks.append((x_landmark, y_landmark))

        # rospy.loginfo(f"Landmarks: {landmarks}")
        
        return landmarks
