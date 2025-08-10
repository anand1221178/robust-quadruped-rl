import gymnasium as gym
import numpy as np

class SuccessRewardWrapper(gym.Wrapper):
    """
    Wraps Ant-v4 to encourage SLOW, STEADY walking
    """
    def __init__(self, env):
        super().__init__(env)
        self.initial_x_position = 0
        self.step_count = 0
        self.previous_x_position = 0
        
        # More forgiving velocity constraints
        self.TARGET_VELOCITY = 0.5      # m/s target
        self.MAX_VELOCITY = 1.0         # m/s absolute max (increased!)
        self.MIN_VELOCITY = 0.1         # m/s minimum (lowered!)
        self.DISTANCE_THRESHOLD = 1.5   # meters
        self.TIME_THRESHOLD = 500       # timesteps (5 seconds)
        
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
        
        # Calculate metrics
        distance_traveled = current_x_position - self.initial_x_position
        instant_velocity = (current_x_position - self.previous_x_position) / self.dt
        
        # Start with small base reward to encourage ANY movement
        custom_reward = 0.0
        
        # 1. Progressive velocity reward - ALWAYS reward forward movement
        if instant_velocity > 0:
            if instant_velocity <= self.TARGET_VELOCITY:
                # Linear reward up to target
                custom_reward += 2.0 * (instant_velocity / self.TARGET_VELOCITY)
            elif instant_velocity <= self.MAX_VELOCITY:
                # Constant good reward in acceptable range
                custom_reward += 2.0
            else:
                # Gentle taper off for too fast (not harsh penalty)
                excess = instant_velocity - self.MAX_VELOCITY
                custom_reward += 2.0 - (excess * 0.5)  # Much gentler!
        else:
            # Small penalty for backward/stationary
            custom_reward += instant_velocity * 0.2  # Very small penalty
        
        # 2. Forward progress bonus (encourage movement!)
        custom_reward += min(distance_traveled * 0.1, 1.0)  # Cap at 1.0
        
        # 3. Survival bonus (smaller so it doesn't dominate)
        if not terminated:
            custom_reward += 0.01
        
        # 4. Milestone bonus
        if (self.step_count >= self.TIME_THRESHOLD and 
            distance_traveled >= self.DISTANCE_THRESHOLD and
            instant_velocity <= self.MAX_VELOCITY):
            custom_reward += 5.0  # Reduced from 10
        
        # 5. Termination penalty (smaller)
        if terminated:
            custom_reward -= 5.0
        
        # 6. REMOVE action penalty - let it move freely!
        # action_penalty = np.sum(np.abs(action)) * 0.01
        # custom_reward -= action_penalty
        
        self.previous_x_position = current_x_position
        
        # Log everything for monitoring
        info['distance_traveled'] = distance_traveled
        info['current_velocity'] = instant_velocity
        info['custom_reward'] = custom_reward
        info['original_reward'] = original_reward
        info['success_achieved'] = (
            self.step_count >= self.TIME_THRESHOLD and 
            distance_traveled >= self.DISTANCE_THRESHOLD and
            instant_velocity <= self.MAX_VELOCITY
        )
        
        # Add custom metrics for W&B logging
        info['custom_metrics/instant_velocity'] = instant_velocity
        info['custom_metrics/distance_traveled'] = distance_traveled
        info['custom_metrics/is_walking_slowly'] = instant_velocity <= self.MAX_VELOCITY
        info['custom_metrics/speed_penalty'] = max(0, instant_velocity - self.MAX_VELOCITY)
        
        return obs, custom_reward, terminated, truncated, info