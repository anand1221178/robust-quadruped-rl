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
        
        # targets
        self.TARGET_VELOCITY = 2.0      # m/s -
        self.MAX_VELOCITY = 3.0         # m/s -
        self.MIN_VELOCITY = 1         # m/s -
        
        # Get timestep
        self.dt = env.dt if hasattr(env, 'dt') else 0.01
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.initial_x_position = self.env.unwrapped.data.qpos[0]
        self.previous_x_position = self.initial_x_position
        self.step_count = 0
        return obs, info
    
    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        
        # Get current position
        current_x_position = self.env.unwrapped.data.qpos[0]
        z_position = self.env.unwrapped.data.qpos[2]
        
        # Calculate velocity
        instant_velocity = (current_x_position - self.previous_x_position) / self.dt
        
        # SIMPLE REWARD STRUCTURE - blend original with speed limit
        custom_reward = original_reward * 0.5
        
        # Add velocity shaping with proper incentives
        if instant_velocity >= self.MIN_VELOCITY and instant_velocity <= self.TARGET_VELOCITY:
            # Strong reward for target walking speed
            velocity_reward = (instant_velocity / self.TARGET_VELOCITY) * 3.0
            custom_reward += velocity_reward
        elif self.TARGET_VELOCITY < instant_velocity <= self.MAX_VELOCITY:
            # Flat reward in acceptable range
            custom_reward += 3.0
        elif instant_velocity > self.MAX_VELOCITY:
            # Gentle penalty for too fast
            excess = instant_velocity - self.MAX_VELOCITY
            custom_reward += 3.0 - (excess * 0.5)
        elif 0 < instant_velocity < self.MIN_VELOCITY:
            # Encourage movement but penalize being too slow
            custom_reward += instant_velocity * 0.5 - 1.0  # Small penalty for being too slow
        else:
            # Strong penalty for backward/stationary movement
            custom_reward -= 2.0
        
        # Height bonus - maintain reasonable height (adjusted for RealAnt's smaller size)
        if 0.15 < z_position < 0.35:  # RealAnt starts at 0.235, so reasonable range
            custom_reward += 0.1
        
        # termination penalty
        if terminated:
            custom_reward -= 5.0
        
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
        
        return obs, custom_reward, terminated, truncated, info
