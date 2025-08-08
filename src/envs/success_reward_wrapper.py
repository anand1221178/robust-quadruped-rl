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
        self.MAX_VELOCITY = 1.5        # m/s (increased tolerance)
        self.MIN_VELOCITY = 0.3        # m/s (lower threshold to encourage movement)
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
        
        # Use a combination of custom and original reward
        # This helps maintain what the robot already learned
        custom_reward = original_reward * 0.3  # Keep 30% of original reward
        
        # 1. Velocity shaping - smooth reward curve
        if 0 < instant_velocity <= self.TARGET_VELOCITY:
            # Linearly increase reward up to target
            custom_reward += 2.0 * (instant_velocity / self.TARGET_VELOCITY)
        elif self.TARGET_VELOCITY < instant_velocity <= self.MAX_VELOCITY:
            # Maintain high reward in acceptable range
            custom_reward += 2.0
        elif instant_velocity > self.MAX_VELOCITY:
            # Gentle penalty for too fast
            excess = instant_velocity - self.MAX_VELOCITY
            custom_reward += 2.0 - (0.2 * excess)  # Very gentle penalty
        else:
            # Moving backward - proportional penalty
            custom_reward += instant_velocity * 0.5
        
        # 2. Progress reward (encourage forward movement)
        custom_reward += distance_traveled * 0.01  # Small constant encouragement
        
        # 3. Survival bonus (very small)
        if not terminated:
            custom_reward += 0.01
        
        # 4. Success bonus at milestone
        if (self.step_count >= self.TIME_THRESHOLD and 
            distance_traveled >= self.DISTANCE_THRESHOLD and 
            instant_velocity <= self.MAX_VELOCITY):
            custom_reward += 20.0  # Big bonus!
        
        # 5. Termination penalty
        if terminated:
            custom_reward -= 5.0  # Reduced from 10
        
        # 6. Very light control cost
        control_cost = 0.001 * np.square(action).sum()
        custom_reward -= control_cost
        
        self.previous_x_position = current_x_position
        
        # Log success metrics
        info['distance_traveled'] = distance_traveled
        info['current_velocity'] = instant_velocity
        info['custom_reward'] = custom_reward
        info['original_reward'] = original_reward
        info['success_achieved'] = (
            self.step_count >= self.TIME_THRESHOLD and 
            distance_traveled >= self.DISTANCE_THRESHOLD and
            instant_velocity <= self.MAX_VELOCITY
        )
        
        return obs, custom_reward, terminated, truncated, info