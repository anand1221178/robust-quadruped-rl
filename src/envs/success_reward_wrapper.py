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
        return obs, info
    
    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        
        # Get current position
        current_x_position = self.env.unwrapped.data.qpos[0]
        z_position = self.env.unwrapped.data.qpos[2]
        
        # Calculate velocity
        instant_velocity = (current_x_position - self.previous_x_position) / self.dt
        
        # PROGRESSIVE REWARD - Encourage faster walking gradually
        custom_reward = 0
        
        # Smooth reward curve that encourages speed
        if instant_velocity <= 0.1:
            # Not moving - penalty
            custom_reward = -5.0
        elif instant_velocity < self.MIN_VELOCITY:
            # Moving but too slow - small penalty to small reward
            progress = instant_velocity / self.MIN_VELOCITY
            custom_reward = -2.0 + (progress * 7.0)  # -2 to +5
        elif instant_velocity <= self.TARGET_VELOCITY:
            # Good speed range - strong rewards
            progress = (instant_velocity - self.MIN_VELOCITY) / (self.TARGET_VELOCITY - self.MIN_VELOCITY)
            custom_reward = 5.0 + (progress * 10.0)  # +5 to +15
        elif instant_velocity <= self.MAX_VELOCITY:
            # Above target but not too fast - maximum reward
            custom_reward = 15.0
        else:
            # Too fast - reduce reward
            excess = instant_velocity - self.MAX_VELOCITY
            custom_reward = 15.0 - (excess * 2.0)
        
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
        
        # Add stability metric
        if hasattr(self.env.unwrapped, 'data'):
            angular_vel = np.linalg.norm(self.env.unwrapped.data.qvel[3:6])
            info['custom_metrics/angular_velocity'] = angular_vel
            info['custom_metrics/is_stable'] = angular_vel < 2.0
        
        return obs, custom_reward, terminated, truncated, info
