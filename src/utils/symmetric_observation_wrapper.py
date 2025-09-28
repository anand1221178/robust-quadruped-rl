import gymnasium as gym
import numpy as np

class SymmetricObservationWrapper(gym.ObservationWrapper):
    """
    This wrapper creates a symmetric observation space for the Ant environment.
    It randomly flips the observations to enforce a symmetric policy.
    """
    def __init__(self, env, flip_prob=0.5):
        super().__init__(env)
        self.flip_prob = flip_prob
        self.single_obs_dim = 29  # Based on RealAntMujoco-v0

        # Indices for simple sign flipping
        # obs = [vx, vy, vz, z, roll_v, pitch_v, yaw_v, sin(r), sin(p), sin(y), cos(r), cos(p), cos(y), jp_1-8, jv_1-8]
        self.sign_flip_indices = [
            1,  # vy (lateral velocity)
            4,  # roll_vel
            6,  # yaw_vel
            7,  # sin(roll)
            9,  # sin(yaw)
        ]

        # Indices for joint pairs that need to be swapped and negated
        # (hip_1, hip_2), (ankle_1, ankle_2), (hip_3, hip_4), (ankle_3, ankle_4)
        # Indices are relative to the start of the 8-dim joint position/velocity blocks
        self.joint_pairs = [
            (0, 2),  # hip_1  <-> hip_2
            (1, 3),  # ankle_1 <-> ankle_2
            (4, 6),  # hip_3  <-> hip_4
            (5, 7),  # ankle_3 <-> ankle_4
        ]

    def observation(self, obs):
        """
        Applies the symmetric flip to the observation with a given probability.
        """
        if self.np_random.uniform() < self.flip_prob:
            return self._flip_observation(obs)
        return obs

    def _flip_observation(self, obs):
        """
        Flips the observation to its symmetric counterpart.
        Handles stacked observations.
        """
        flipped_obs = obs.copy()
        num_chunks = len(obs) // self.single_obs_dim

        for i in range(num_chunks):
            start_idx = i * self.single_obs_dim
            end_idx = start_idx + self.single_obs_dim
            
            chunk = flipped_obs[start_idx:end_idx]
            original_chunk = obs[start_idx:end_idx]

            # 1. Flip signs of lateral movements
            chunk[self.sign_flip_indices] *= -1

            # 2. Swap and negate joint positions
            joint_pos_start = 13
            for left_idx, right_idx in self.joint_pairs:
                left_abs_idx = joint_pos_start + left_idx
                right_abs_idx = joint_pos_start + right_idx
                chunk[left_abs_idx], chunk[right_abs_idx] = \
                    -original_chunk[right_abs_idx], -original_chunk[left_abs_idx]

            # 3. Swap and negate joint velocities
            joint_vel_start = 21
            for left_idx, right_idx in self.joint_pairs:
                left_abs_idx = joint_vel_start + left_idx
                right_abs_idx = joint_vel_start + right_idx
                chunk[left_abs_idx], chunk[right_abs_idx] = \
                    -original_chunk[right_abs_idx], -original_chunk[left_abs_idx]
        
        return flipped_obs
