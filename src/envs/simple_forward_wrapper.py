"""
Simple Forward Reward Wrapper
Just reward forward movement, nothing fancy
"""

import gymnasium as gym
import numpy as np

class SimpleForwardWrapper(gym.Wrapper):
    """
    Dead simple: Reward forward movement, penalize everything else
    """
    def __init__(self, env):
        super().__init__(env)
        self.previous_x = 0
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_x = self.env.unwrapped.data.qpos[0]
        return obs, info
    
    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        
        # Get current x position
        current_x = self.env.unwrapped.data.qpos[0]
        
        # Simple forward reward
        forward_reward = (current_x - self.previous_x) * 100  # Scale up for impact
        
        # Height bonus (stay upright)
        z_position = self.env.unwrapped.data.qpos[2]
        height_reward = 0
        if 0.15 < z_position < 0.35:
            height_reward = 1.0
        
        # Control penalty (smaller actions = smoother)
        control_penalty = -0.01 * np.sum(np.square(action))
        
        # Total reward
        reward = forward_reward + height_reward + control_penalty
        
        # Early termination if fallen
        if z_position < 0.1 or z_position > 0.5:
            terminated = True
            reward -= 10  # Fall penalty
        
        self.previous_x = current_x
        
        # Logging
        info['forward_velocity'] = forward_reward / 100  # Actual velocity
        info['height'] = z_position
        info['reward_forward'] = forward_reward
        info['reward_height'] = height_reward
        info['reward_control'] = control_penalty
        
        return obs, reward, terminated, truncated, info