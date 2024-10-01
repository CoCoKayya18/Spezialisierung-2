import rospy
import numpy as np

class Map:
    def __init__(self, config):
        self.landmarks = []
        self.landmarkNumber = 0

        self.alpha = 5.991 # 95% confidence based on Chi-squared distribution
        rospy.loginfo("Map class initialized")

    def calculate_landmark_estimates(self, x, y, theta, z_i):

        r_i, phi_i = z_i

        # r_i = z_i[0]  
        # phi_i = z_i[1]

        # rospy.loginfo(f"Z_i: {z_i}")
        # rospy.loginfo(f"r_i: {r_i}")
        # rospy.loginfo(f"phi_i: {phi_i}")

        # rospy.loginfo(f"X robot: {x}")
        # rospy.loginfo(f"Y robot: {y}")
        # rospy.loginfo(f"Theta robot: {theta}")

        # rospy.loginfo(f"Angle before: {phi_i + theta} and after normalization: {np.arctan2(np.sin(phi_i + theta), np.cos(phi_i + theta))}")

        mu_N_plus_1_x = x + r_i * np.cos(np.arctan2(np.sin(phi_i + theta), np.cos(phi_i + theta)))
        mu_N_plus_1_y = y + r_i * np.sin(np.arctan2(np.sin(phi_i + theta), np.cos(phi_i + theta)))

        # rospy.loginfo(f"mu_N_plus_1_x robot: {mu_N_plus_1_x}")
        # rospy.loginfo(f"mu_N_plus_1_y robot: {mu_N_plus_1_y}")

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

        Psi_k = H_k_t @ Sigma_t @ H_k_t.T + measurement_noise
        pi_k = (z_i - z_hat_k).T @ np.linalg.inv(Psi_k) @ (z_i - z_hat_k)
        return pi_k, Psi_k

    def data_association(self, pi_k):
        if np.min(pi_k) <= self.alpha:
            return np.argmin(pi_k)
        return None

    def update_state(self, x, y, theta, z_i, z_hat_k, K_i_t):
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
