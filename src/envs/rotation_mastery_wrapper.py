import gymnasium as gym
import numpy as np
import random


class RotationMasteryWrapper(gym.Wrapper):
    """
    V7.11 - Rotation Mastery Wrapper
    Teaches ankle_4 the same rotation strategy that ankle_1 uses successfully.
    Uses NaN-safe multiplicative rewards to encourage rotation-based recovery.
    """

    def __init__(self, env, config=None):
        super().__init__(env)

        # Default config
        default_config = {
            'enabled': True,
            'yaw_change_multiplier': 1.5,
            'forward_after_rotation': 2.0,
            'max_total_multiplier': 2.5,
            'min_rotation_threshold': 0.1,
            'min_forward_threshold': 0.05,
            'target_joints': ['ankle_4'],  # Focus on ankle_4, but can include ankle_1 for learning
        }

        self.config = {**default_config, **(config or {})}

        # State tracking
        self.prev_yaw = 0.0
        self.failed_joints = []
        self.episode_step = 0

        # Safety tracking to prevent reward explosion
        self.max_reward_seen = 0.0
        self.reward_history = []

        print("Rotation Mastery Wrapper: Teaching ankle_4 rotation strategy")
        print(f"   Target joints: {self.config['target_joints']}")
        print(f"   Max rotation multiplier: {self.config['yaw_change_multiplier']}")
        print(f"   Max forward multiplier: {self.config['forward_after_rotation']}")
        print(f"   Hard cap: {self.config['max_total_multiplier']}")

    def reset(self, **kwargs):
        """Reset episode state tracking"""
        obs, info = self.env.reset(**kwargs)

        # Reset tracking
        self.prev_yaw = self._extract_yaw(obs)
        self.failed_joints = []
        self.episode_step = 0

        return obs, info

    def step(self, action):
        """Apply rotation mastery rewards for target joint failures"""
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.episode_step += 1

        # Get current state
        current_yaw = self._extract_yaw(obs)
        forward_velocity = self._extract_forward_velocity(obs)

        # Track failed joints from info
        if 'dropped_joints' in info:
            self.failed_joints = info['dropped_joints']

        # Apply rotation mastery rewards if enabled and target joint failed
        if (self.config['enabled'] and
            self._target_joint_failed() and
            self.episode_step > 120):  # Only after delayed locking

            rotation_multiplier = self._calculate_rotation_multiplier(
                current_yaw, forward_velocity
            )

            # Apply NaN-safe multiplicative reward
            original_reward = reward
            reward = self._apply_safe_multiplier(reward, rotation_multiplier)

            # Debug info (optional)
            if abs(rotation_multiplier - 1.0) > 0.01:
                yaw_change = abs(current_yaw - self.prev_yaw)


        # Update state tracking
        self.prev_yaw = current_yaw

        return obs, reward, terminated, truncated, info

    def _target_joint_failed(self):
        """Check if any target joint (ankle_4, etc.) has failed"""
        target_indices = []
        joint_map = {'ankle_1': 1, 'ankle_2': 3, 'ankle_3': 5, 'ankle_4': 7,
                     'hip_1': 0, 'hip_2': 2, 'hip_3': 4, 'hip_4': 6}

        for joint_name in self.config['target_joints']:
            if joint_name in joint_map:
                target_indices.append(joint_map[joint_name])

        return any(idx in self.failed_joints for idx in target_indices)

    def _calculate_rotation_multiplier(self, current_yaw, forward_velocity):
        """Calculate NaN-safe rotation reward multiplier"""
        multiplier = 1.0

        # Calculate yaw change (handle wrapping)
        yaw_change = abs(current_yaw - self.prev_yaw)
        if yaw_change > np.pi:  # Handle 2π wrapping
            yaw_change = 2 * np.pi - yaw_change

        # Reward rotation when stuck
        if yaw_change > self.config['min_rotation_threshold']:
            rotation_bonus = min(
                self.config['yaw_change_multiplier'],
                1.0 + yaw_change * 0.5  # Scale with rotation amount
            )
            multiplier *= rotation_bonus

        # Big reward for forward movement (the goal!)
        if forward_velocity > self.config['min_forward_threshold']:
            forward_bonus = self.config['forward_after_rotation']
            multiplier *= forward_bonus

        # Hard cap to prevent NaN/overflow
        multiplier = min(multiplier, self.config['max_total_multiplier'])

        return multiplier

    def _apply_safe_multiplier(self, reward, multiplier):
        """Apply multiplier with safety checks to prevent NaN/overflow"""
        # Safety check 1: Clamp extreme rewards
        if abs(reward) > 1000:
            reward = np.sign(reward) * 1000

        # Safety check 2: Apply multiplier
        new_reward = reward * multiplier

        # Safety check 3: Check for NaN/inf
        if not np.isfinite(new_reward):
            print(f"Non-finite reward detected! Original: {reward}, Multiplier: {multiplier}")
            return reward  # Return original reward if problems

        # Safety check 4: Extreme value check
        if abs(new_reward) > 10000:
            print(f"Extreme reward detected! Clamping: {new_reward}")
            new_reward = np.sign(new_reward) * 10000

        # Track for monitoring
        self.max_reward_seen = max(self.max_reward_seen, abs(new_reward))
        self.reward_history.append(new_reward)
        if len(self.reward_history) > 1000:
            self.reward_history.pop(0)

        return new_reward

    def _extract_yaw(self, obs):
        """Extract yaw angle from observation"""
        # For RealAnt, yaw is in the orientation part of obs
        # obs format: [vx, vy, vz, z, roll_v, pitch_v, yaw_v, sin(r), sin(p), sin(y), cos(r), cos(p), cos(y), ...]
        if len(obs) >= 13:
            sin_yaw = obs[9]
            cos_yaw = obs[12]
            return np.arctan2(sin_yaw, cos_yaw)
        return 0.0

    def _extract_forward_velocity(self, obs):
        """Extract forward velocity from observation"""
        # For RealAnt walking left→right, vx is forward velocity
        if len(obs) >= 1:
            return obs[0]
        return 0.0

    def get_diagnostics(self):
        """Return diagnostic information for monitoring"""
        return {
            'max_reward_seen': self.max_reward_seen,
            'recent_avg_reward': np.mean(self.reward_history) if self.reward_history else 0.0,
            'recent_reward_std': np.std(self.reward_history) if self.reward_history else 0.0,
        }