import gymnasium as gym
import numpy as np

class SuccessRewardWrapper(gym.Wrapper):
    """
    Wraps Ant-v4 to use our custom success metrics as rewards
    Encourages steady walking at target velocity, not crazy running!
    """
    def __init__(self, env):
        super().__init__(env)
        self.initial_x_position = 0
        self.step_count = 0
        self.previous_x_position = 0
        
        # Success thresholds from your metrics
        self.TARGET_VELOCITY = 0.75    # m/s (middle of 0.5-1.0 range)
        self.MAX_VELOCITY = 1.0        # m/s (penalize above this)
        self.MIN_VELOCITY = 0.5        # m/s (penalize below this)
        self.DISTANCE_THRESHOLD = 1.5  # meters
        self.TIME_THRESHOLD = 500      # timesteps (5 seconds)
        
        # Get timestep
        self.dt = env.dt if hasattr(env, 'dt') else 0.01
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Get initial x position
        self.initial_x_position = self.env.unwrapped.data.qpos[0]
        self.previous_x_position = self.initial_x_position
        self.step_count = 0
        return obs, info
    
    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        
        # Get current position
        current_x_position = self.env.unwrapped.data.qpos[0]
        
        # Calculate metrics
        distance_traveled = current_x_position - self.initial_x_position
        instant_velocity = (current_x_position - self.previous_x_position) / self.dt
        
        # Custom reward based on YOUR success metrics
        custom_reward = 0.0
        
        # 1. Velocity reward - encourage forward movement
        if self.MIN_VELOCITY <= instant_velocity <= self.MAX_VELOCITY:
            # Perfect range - full reward
            custom_reward += 2.0  # Increased reward for moving
        elif 0 < instant_velocity < self.MIN_VELOCITY:
            # Too slow but moving forward - partial reward
            custom_reward += 1.0 + (instant_velocity / self.MIN_VELOCITY)
        elif instant_velocity > self.MAX_VELOCITY:
            # Too fast - slight penalty
            excess_speed = instant_velocity - self.MAX_VELOCITY
            custom_reward += 2.0 - (excess_speed * 0.1)  # Reduced penalty
        else:
            # Backward or stationary - penalty
            custom_reward += instant_velocity  # Negative if moving backward
        
        # 2. Stability bonus (reduce jumping/flipping)
        # Penalize large vertical movements
        z_position = self.env.unwrapped.data.qpos[2]  # vertical position
        if abs(z_position - 0.75) > 0.3:  # Increased tolerance for natural gait
            custom_reward -= 0.05  # Reduced penalty
        
        # 3. Survival bonus (smaller to not dominate)
        custom_reward += 0.05
        
        # 4. Success bonus at milestone
        if self.step_count >= self.TIME_THRESHOLD and distance_traveled >= self.DISTANCE_THRESHOLD:
            custom_reward += 10.0  # Big bonus for achieving goal calmly!
        
        # 5. Penalty for falling
        if terminated:
            custom_reward -= 10.0
        
        # 6. Control cost (penalize wild movements)
        control_cost = 0.01 * np.square(action).sum()  # Much smaller penalty
        custom_reward -= control_cost
        
        self.previous_x_position = current_x_position
        
        # Log success metrics
        info['distance_traveled'] = distance_traveled
        info['current_velocity'] = instant_velocity
        info['success_achieved'] = (
            self.step_count >= self.TIME_THRESHOLD and 
            distance_traveled >= self.DISTANCE_THRESHOLD and
            instant_velocity <= self.MAX_VELOCITY  # Must be calm!
        )
        
        return obs, custom_reward, terminated, truncated, info