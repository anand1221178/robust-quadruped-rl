import gymnasium as gym
import numpy as np

class SuccessRewardWrapper(gym.Wrapper):
    """
    Simplified wrapper to encourage natural walking
    """
    def __init__(self, env):
        super().__init__(env)
        self.initial_x_position = 0
        self.step_count = 0
        self.previous_x_position = 0
        self.previous_action = None
        
        # targets - REALISTIC & STABLE WALKING
        self.TARGET_VELOCITY = 1.5      # m/s - Realistic target for RealAnt
        self.MAX_VELOCITY = 2.5         # m/s - Allow some faster movement
        self.MIN_VELOCITY = 0.5         # m/s - Minimum walking speed
        
        # Get timestep
        self.dt = env.dt if hasattr(env, 'dt') else 0.01
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.initial_x_position = self.env.unwrapped.data.qpos[0]
        self.previous_x_position = self.initial_x_position
        self.step_count = 0
        self.previous_action = None  # Reset previous action
        return obs, info
    
    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        
        # Get current position
        current_x_position = self.env.unwrapped.data.qpos[0]
        z_position = self.env.unwrapped.data.qpos[2]
        
        # Calculate velocity
        instant_velocity = (current_x_position - self.previous_x_position) / self.dt
        
        # SIMPLE FORWARD REWARD - Just reward ANY forward movement
        # Start with base forward reward
        custom_reward = instant_velocity * 10.0  # 10x multiplier for forward movement
        
        # Add bonus for reaching minimum speed
        if instant_velocity >= self.MIN_VELOCITY:
            custom_reward += 5.0
        
        # Add bonus for reaching target speed  
        if instant_velocity >= self.TARGET_VELOCITY:
            custom_reward += 10.0
        
        # Only penalize if going backwards
        if instant_velocity < 0:
            custom_reward = instant_velocity * 20.0  # Harsh penalty for backwards
        
        # Small penalty for not moving at all
        if abs(instant_velocity) < 0.01:
            custom_reward -= 2.0
        
        # Height bonus - maintain reasonable height (adjusted for RealAnt's smaller size)
        if 0.15 < z_position < 0.35:  # RealAnt starts at 0.235, so reasonable range
            custom_reward += 0.1
        
        # STABILITY BONUS - Penalize excessive spinning/slipping
        if hasattr(self.env.unwrapped, 'data'):
            angular_vel = np.linalg.norm(self.env.unwrapped.data.qvel[3:6])
            if angular_vel < 2.0:  # Not spinning wildly
                custom_reward += 1.0  # Reward stable movement
            else:
                custom_reward -= (angular_vel - 2.0) * 0.5  # Penalize excessive spinning
        
        # Remove action penalties for now - let it learn to walk first
        # (We'll add smoothness back with SR2L later)
        
        # termination penalty - REDUCED (robot learning to walk will fall)
        if terminated:
            custom_reward -= 5.0  # Penalty but not too harsh
        
        # Store action for next step
        self.previous_action = action.copy()
        
        self.previous_x_position = current_x_position
        
        # Logging
        info['distance_traveled'] = current_x_position - self.initial_x_position
        info['current_velocity'] = instant_velocity
        info['custom_reward'] = custom_reward
        info['original_reward'] = original_reward
        
        # Custom metrics
        info['custom_metrics/instant_velocity'] = instant_velocity
        info['custom_metrics/distance_traveled'] = current_x_position - self.initial_x_position
        info['custom_metrics/is_walking_slowly'] = instant_velocity <= self.MAX_VELOCITY
        info['custom_metrics/speed_penalty'] = max(0, instant_velocity - self.MAX_VELOCITY)
        info['custom_metrics/height'] = z_position
        info['custom_metrics/gait_quality'] = 0  # Removed complex gait analysis
        
        # Add stability metric
        if hasattr(self.env.unwrapped, 'data'):
            angular_vel = np.linalg.norm(self.env.unwrapped.data.qvel[3:6])
            info['custom_metrics/angular_velocity'] = angular_vel
            info['custom_metrics/is_stable'] = angular_vel < 2.0
        
        return obs, custom_reward, terminated, truncated, info
