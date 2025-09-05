"""
Straight Line Locomotion Wrapper
Forces robot to walk in straight line (x-axis only) by penalizing lateral movement
"""

import numpy as np
import gymnasium as gym

class StraightLineWrapper(gym.Wrapper):
    """
    Wrapper that encourages straight-line locomotion along x-axis
    Penalizes y-axis movement and rotation to force forward walking
    """
    
    def __init__(
        self, 
        env,
        lateral_penalty=2.0,      # Penalty for y-axis movement
        rotation_penalty=1.0,     # Penalty for rotation (yaw)
        straight_bonus=0.5,       # Bonus for staying near y=0
        max_lateral_deviation=2.0 # Reset if too far from centerline
    ):
        super().__init__(env)
        
        self.lateral_penalty = lateral_penalty
        self.rotation_penalty = rotation_penalty  
        self.straight_bonus = straight_bonus
        self.max_lateral_deviation = max_lateral_deviation
        
        # Track position
        self.initial_pos = None
        self.last_pos = None
        
    def reset(self, **kwargs):
        """Reset and store initial position"""
        obs = self.env.reset(**kwargs)
        
        # Get initial position
        try:
            if hasattr(self.env.unwrapped, 'sim'):
                self.initial_pos = self.env.unwrapped.sim.data.qpos[:3].copy()  # x, y, z
                self.last_pos = self.initial_pos.copy()
        except:
            self.initial_pos = np.array([0.0, 0.0, 0.75])  # Default spawn
            self.last_pos = self.initial_pos.copy()
            
        return obs
    
    def step(self, action):
        """Apply straight-line reward modifications"""
        obs, reward, done, info = self.env.step(action)
        
        # Get current position and orientation
        try:
            if hasattr(self.env.unwrapped, 'sim'):
                current_pos = self.env.unwrapped.sim.data.qpos[:3].copy()  # x, y, z
                orientation = self.env.unwrapped.sim.data.qpos[3:7]  # quaternion
                
                # Calculate yaw from quaternion (rotation around z-axis)
                w, x, y, z = orientation
                yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
            else:
                # Fallback
                current_pos = self.last_pos + np.array([0.1, 0, 0])  # Assume forward movement
                yaw = 0.0
        except:
            current_pos = self.last_pos + np.array([0.1, 0, 0])
            yaw = 0.0
        
        # Calculate movement components
        if self.last_pos is not None:
            delta_pos = current_pos - self.last_pos
            x_velocity = delta_pos[0] / 0.05  # Convert to m/s (dt=0.05)
            y_velocity = abs(delta_pos[1]) / 0.05  # Absolute lateral movement
        else:
            x_velocity = 0.0
            y_velocity = 0.0
        
        # Straight-line reward modifications
        straight_reward = 0.0
        
        # 1. Penalize lateral movement (y-axis)
        if y_velocity > 0.01:  # Small threshold to avoid penalizing tiny movements
            straight_reward -= self.lateral_penalty * y_velocity
        
        # 2. Penalize rotation (yaw)
        yaw_penalty = self.rotation_penalty * abs(yaw)
        straight_reward -= yaw_penalty
        
        # 3. Bonus for staying near centerline (y ≈ 0)
        y_distance_from_center = abs(current_pos[1] - self.initial_pos[1])
        if y_distance_from_center < 0.5:  # Within 50cm of centerline
            straight_reward += self.straight_bonus * (0.5 - y_distance_from_center)
        
        # 4. Reset if too far from centerline (safety)
        if y_distance_from_center > self.max_lateral_deviation:
            done = True
            straight_reward -= 10.0  # Large penalty for going off-track
            info['reset_reason'] = 'lateral_deviation'
        
        # Apply reward modification
        reward += straight_reward
        
        # Update info with straight-line metrics
        info['x_velocity'] = x_velocity
        info['y_velocity'] = y_velocity
        info['yaw'] = yaw
        info['y_distance_from_center'] = y_distance_from_center
        info['straight_line_bonus'] = straight_reward
        
        # Update tracking
        self.last_pos = current_pos
        
        return obs, reward, done, info


class StrictStraightLineWrapper(StraightLineWrapper):
    """
    Even stricter version that heavily penalizes any deviation
    """
    
    def __init__(self, env, **kwargs):
        super().__init__(
            env,
            lateral_penalty=5.0,      # Much higher penalty
            rotation_penalty=3.0,     # Much higher penalty
            straight_bonus=1.0,       # Higher bonus
            max_lateral_deviation=1.0, # Stricter deviation limit
            **kwargs
        )
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        
        # Additional strict penalties
        y_velocity = info.get('y_velocity', 0)
        yaw = info.get('yaw', 0)
        
        # Extra penalty for any lateral movement
        if y_velocity > 0.005:  # Very small threshold
            reward -= 1.0
            
        # Extra penalty for any rotation
        if abs(yaw) > 0.1:  # Small rotation threshold
            reward -= 1.0
            
        return obs, reward, done, info