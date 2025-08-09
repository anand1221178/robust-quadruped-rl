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
        
        # Much stricter velocity constraints
        self.TARGET_VELOCITY = 0.5      # m/s target
        self.MAX_VELOCITY = 0.75        # m/s absolute max
        self.MIN_VELOCITY = 0.2         # m/s minimum
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
        
        # IGNORE original reward completely - it's making the robot run too fast!
        custom_reward = 0.0
        
        # 1. Velocity reward with HARSH penalties for going too fast
        if 0 < instant_velocity <= self.TARGET_VELOCITY:
            # Good speed - full reward
            custom_reward += 1.0 * (instant_velocity / self.TARGET_VELOCITY)
        elif self.TARGET_VELOCITY < instant_velocity <= self.MAX_VELOCITY:
            # Acceptable but not ideal
            custom_reward += 1.0 - 0.5 * ((instant_velocity - self.TARGET_VELOCITY) / (self.MAX_VELOCITY - self.TARGET_VELOCITY))
        elif instant_velocity > self.MAX_VELOCITY:
            # TOO FAST - heavy penalty that scales with speed
            excess = instant_velocity - self.MAX_VELOCITY
            custom_reward -= excess * 2.0  # -2 reward per m/s over limit!
        else:
            # Moving backward
            custom_reward += instant_velocity * 0.5
        
        # 2. Stability bonus - reward staying low
        z_position = self.env.unwrapped.data.qpos[2]
        if 0.5 < z_position < 0.9:  # Good height range
            custom_reward += 0.1
        
        # 3. Smoothness bonus - penalize jerky movements
        action_penalty = np.sum(np.abs(action)) * 0.01
        custom_reward -= action_penalty
        
        # 4. Small survival bonus
        if not terminated:
            custom_reward += 0.05
        
        # 5. Success bonus ONLY if walking slowly
        if (self.step_count >= self.TIME_THRESHOLD and 
            distance_traveled >= self.DISTANCE_THRESHOLD and 
            instant_velocity <= self.MAX_VELOCITY):
            custom_reward += 10.0
        
        # 6. Termination penalty
        if terminated:
            custom_reward -= 10.0
        
        self.previous_x_position = current_x_position
        
        # Log everything
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