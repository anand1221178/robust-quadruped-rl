"""
Target Walking Wrapper
Reward robot for walking toward a target position
"""

import gymnasium as gym
import numpy as np

class TargetWalkingWrapper(gym.Wrapper):
    """
    Reward walking toward a target position
    """
    def __init__(self, env, target_distance=5.0, speed_bonus_scale=2.0):
        super().__init__(env)
        self.target_distance = target_distance  # How far to place target
        self.speed_bonus_scale = speed_bonus_scale
        self.target_x = target_distance
        self.initial_distance = 0
        self.previous_distance = 0
        self.episode_steps = 0
        self.dt = env.dt if hasattr(env, 'dt') else 0.01
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Set target ahead of robot
        current_x = self.env.unwrapped.data.qpos[0]
        self.target_x = current_x + self.target_distance
        self.initial_distance = self.target_distance
        self.previous_distance = self.target_distance
        self.episode_steps = 0
        
        return obs, info
    
    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1
        
        # Get current position
        current_x = self.env.unwrapped.data.qpos[0]
        current_z = self.env.unwrapped.data.qpos[2]
        
        # Calculate distance to target
        distance_to_target = abs(self.target_x - current_x)
        
        # Progress reward - how much closer we got
        progress = self.previous_distance - distance_to_target
        progress_reward = progress * 100  # Scale up
        
        # Speed bonus - reward faster progress
        speed = progress / self.dt
        if speed > 0.5:  # Moving at decent speed
            speed_bonus = speed * self.speed_bonus_scale
        else:
            speed_bonus = 0
        
        # Distance penalty - encourage reaching target quickly
        distance_penalty = -distance_to_target * 0.1
        
        # Height reward - stay upright
        height_reward = 0
        if 0.15 < current_z < 0.35:
            height_reward = 1.0
        
        # Success bonus - reached target
        success_bonus = 0
        if distance_to_target < 0.5:  # Within 0.5m of target
            success_bonus = 100
            # Move target further
            self.target_x = current_x + self.target_distance
            self.previous_distance = self.target_distance
            print(f"Target reached! New target at x={self.target_x:.2f}")
        
        # Control penalty
        control_penalty = -0.01 * np.sum(np.square(action))
        
        # Total reward
        custom_reward = (
            progress_reward + 
            speed_bonus + 
            distance_penalty + 
            height_reward + 
            success_bonus + 
            control_penalty
        )
        
        # Early termination if fallen
        if current_z < 0.1 or current_z > 0.5:
            terminated = True
            custom_reward -= 10
        
        # Update for next step
        self.previous_distance = distance_to_target
        
        # Logging
        info['distance_to_target'] = distance_to_target
        info['progress'] = progress
        info['speed'] = speed
        info['success_bonus'] = success_bonus
        info['custom_reward'] = custom_reward
        
        return obs, custom_reward, terminated, truncated, info