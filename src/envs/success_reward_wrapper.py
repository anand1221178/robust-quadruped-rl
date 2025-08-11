import gymnasium as gym
import numpy as np

class SuccessRewardWrapper(gym.Wrapper):
    """
    Simplified wrapper to encourage natural walking like DeepMind's
    """
    def __init__(self, env):
        super().__init__(env)
        self.initial_x_position = 0
        self.step_count = 0
        self.previous_x_position = 0
        
        # More realistic targets
        self.TARGET_VELOCITY = 2.0      # m/s - much more natural walking speed
        self.MAX_VELOCITY = 3.0         # m/s - allow some flexibility
        self.MIN_VELOCITY = 0.5         # m/s - need actual movement
        
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
        # Use 50% of original reward to maintain good walking mechanics
        custom_reward = original_reward * 0.5
        
        # Add velocity shaping - gentle guidance toward target
        if 0 < instant_velocity <= self.TARGET_VELOCITY:
            # Reward proportional to velocity up to target
            velocity_reward = (instant_velocity / self.TARGET_VELOCITY) * 2.0
            custom_reward += velocity_reward
        elif self.TARGET_VELOCITY < instant_velocity <= self.MAX_VELOCITY:
            # Flat reward in acceptable range
            custom_reward += 2.0
        elif instant_velocity > self.MAX_VELOCITY:
            # Gentle penalty for too fast
            excess = instant_velocity - self.MAX_VELOCITY
            custom_reward += 2.0 - (excess * 0.3)
        else:
            # Penalty for backward/stationary
            custom_reward += instant_velocity
        
        # Height bonus - just maintain reasonable height
        if 0.5 < z_position < 1.0:
            custom_reward += 0.1
        
        # Small termination penalty
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