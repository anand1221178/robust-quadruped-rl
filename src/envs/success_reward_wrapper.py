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
        
        # targets - AGGRESSIVE - FORCE WALKING
        self.TARGET_VELOCITY = 1.5      # m/s - Higher target for real walking
        self.MAX_VELOCITY = 3.0         # m/s - Allow faster movement
        self.MIN_VELOCITY = 0.8         # m/s - FORCE minimum walking speed
        
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
        
        # AGGRESSIVE REWARD STRUCTURE - FORCE WALKING OR DIE
        custom_reward = 0  # Ignore original reward completely - focus on walking
        
        # EXTREME velocity shaping - walk or get punished
        if instant_velocity >= self.MIN_VELOCITY and instant_velocity <= self.TARGET_VELOCITY:
            # MASSIVE reward for actual walking
            velocity_reward = (instant_velocity / self.TARGET_VELOCITY) * 10.0
            custom_reward += velocity_reward
        elif self.TARGET_VELOCITY < instant_velocity <= self.MAX_VELOCITY:
            # Big reward for fast walking
            custom_reward += 10.0
        elif instant_velocity > self.MAX_VELOCITY:
            # Still reward but with penalty for being too fast
            excess = instant_velocity - self.MAX_VELOCITY
            custom_reward += 10.0 - (excess * 1.0)
        elif 0.1 < instant_velocity < self.MIN_VELOCITY:
            # HARSH penalty for slow crawling - force them to walk faster
            custom_reward -= 5.0 * (self.MIN_VELOCITY - instant_velocity)
        else:
            # SEVERE penalty for not moving
            custom_reward -= 10.0
        
        # Height bonus - maintain reasonable height (adjusted for RealAnt's smaller size)
        if 0.15 < z_position < 0.35:  # RealAnt starts at 0.235, so reasonable range
            custom_reward += 0.1
        
        # termination penalty - REDUCED (robot learning to walk will fall)
        if terminated:
            custom_reward -= 1.0  # Much gentler penalty
        
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
