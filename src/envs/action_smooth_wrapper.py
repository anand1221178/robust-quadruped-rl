"""
Action Smoothing Wrapper
Smooths actions using exponential moving average to reduce jerkiness
"""

import gymnasium as gym
import numpy as np
from collections import deque

class ActionSmoothWrapper(gym.Wrapper):
    """
    Smooths actions to reduce jerkiness in robot movement
    """
    def __init__(self, env, alpha=0.7, window_size=3):
        """
        Args:
            env: Environment to wrap
            alpha: Smoothing factor (0=no smoothing, 1=no change)
            window_size: Number of past actions to consider
        """
        super().__init__(env)
        self.alpha = alpha
        self.window_size = window_size
        self.action_history = deque(maxlen=window_size)
        self.previous_action = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.action_history.clear()
        self.previous_action = None
        return obs, info
    
    def step(self, action):
        # Smooth the action
        smoothed_action = self._smooth_action(action)
        
        # Step with smoothed action
        obs, reward, terminated, truncated, info = self.env.step(smoothed_action)
        
        # Store action metrics
        if self.previous_action is not None:
            action_change = np.linalg.norm(smoothed_action - self.previous_action)
            info['action_smoothness/change'] = action_change
            info['action_smoothness/raw_action'] = action
            info['action_smoothness/smoothed_action'] = smoothed_action
        
        self.previous_action = smoothed_action.copy()
        
        return obs, reward, terminated, truncated, info
    
    def _smooth_action(self, action):
        """Apply exponential moving average smoothing"""
        action = np.array(action)
        
        # If no history, use current action
        if len(self.action_history) == 0:
            self.action_history.append(action)
            return action
        
        # Exponential moving average
        if self.previous_action is not None:
            smoothed = self.alpha * self.previous_action + (1 - self.alpha) * action
        else:
            smoothed = action
        
        # Add to history
        self.action_history.append(smoothed)
        
        # Additional smoothing: average over window
        if len(self.action_history) >= 2:
            window_avg = np.mean(list(self.action_history), axis=0)
            smoothed = 0.7 * smoothed + 0.3 * window_avg
        
        # Clip to action space bounds
        smoothed = np.clip(smoothed, self.action_space.low, self.action_space.high)
        
        return smoothed