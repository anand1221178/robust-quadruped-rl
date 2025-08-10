import gymnasium as gym
import numpy as np

class SuccessRewardWrapper(gym.Wrapper):
    """
    Wraps Ant-v4 to encourage PROPER walking at moderate speed
    """
    def __init__(self, env):
        super().__init__(env)
        self.initial_x_position = 0
        self.step_count = 0
        self.previous_x_position = 0
        self.previous_action = None
        
        # Velocity constraints
        self.TARGET_VELOCITY = 0.5      # m/s target
        self.MAX_VELOCITY = 1.0         # m/s absolute max
        self.MIN_VELOCITY = 0.1         # m/s minimum
        self.DISTANCE_THRESHOLD = 1.5   # meters
        self.TIME_THRESHOLD = 500       # timesteps (5 seconds)
        
        # Get timestep
        self.dt = env.dt if hasattr(env, 'dt') else 0.01
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.initial_x_position = self.env.unwrapped.data.qpos[0]
        self.previous_x_position = self.initial_x_position
        self.step_count = 0
        self.previous_action = None
        return obs, info
    
    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        
        # Get current position and orientation
        current_x_position = self.env.unwrapped.data.qpos[0]
        z_position = self.env.unwrapped.data.qpos[2]  # height
        
        # Get joint velocities for gait analysis
        joint_velocities = obs[13:21] if len(obs) >= 21 else np.zeros(8)
        
        # Calculate metrics
        distance_traveled = current_x_position - self.initial_x_position
        instant_velocity = (current_x_position - self.previous_x_position) / self.dt
        
        # Start with base reward
        custom_reward = 0.0
        
        # 1. Velocity reward
        if instant_velocity > 0:
            if instant_velocity <= self.TARGET_VELOCITY:
                custom_reward += 2.0 * (instant_velocity / self.TARGET_VELOCITY)
            elif instant_velocity <= self.MAX_VELOCITY:
                custom_reward += 2.0
            else:
                excess = instant_velocity - self.MAX_VELOCITY
                custom_reward += 2.0 - (excess * 0.5)
        else:
            custom_reward += instant_velocity * 0.2
        
        # 2. GAIT QUALITY REWARDS - This is the key!
        # Reward alternating leg movements (proper walking pattern)
        if self.previous_action is not None:
            # Check if actions are changing (not dragging with fixed legs)
            action_change = np.abs(action - self.previous_action).mean()
            custom_reward += min(action_change * 2.0, 0.5)  # Reward changing actions
            
            # Penalize if some joints aren't moving (dragging behavior)
            static_joints = np.sum(np.abs(action) < 0.1)  # Count near-zero actions
            custom_reward -= static_joints * 0.1  # Penalty for each static joint
        
        # 3. Height maintenance (penalize dragging on ground)
        ideal_height = 0.75  # Ant's normal walking height
        height_error = abs(z_position - ideal_height)
        if height_error < 0.2:
            custom_reward += 0.2  # Reward good posture
        else:
            custom_reward -= height_error * 0.5  # Penalize dragging/jumping
        
        # 4. Symmetry reward (all legs should contribute)
        # Check variance in joint actions - want them all active
        if len(action) >= 8:
            action_variance = np.var(np.abs(action))
            if action_variance > 0.01:  # All joints active
                custom_reward += 0.1
        
        # 5. Energy efficiency (small but not zero actions)
        # Encourage moderate torques, not extreme or zero
        avg_torque = np.mean(np.abs(action))
        if 0.2 < avg_torque < 0.8:  # Sweet spot for walking
            custom_reward += 0.1
        
        # 6. Forward progress bonus
        custom_reward += min(distance_traveled * 0.05, 0.5)
        
        # 7. Survival bonus
        if not terminated:
            custom_reward += 0.01
        
        # 8. Milestone bonus
        if (self.step_count >= self.TIME_THRESHOLD and 
            distance_traveled >= self.DISTANCE_THRESHOLD and
            instant_velocity <= self.MAX_VELOCITY):
            custom_reward += 5.0
        
        # 9. Termination penalty
        if terminated:
            custom_reward -= 5.0
        
        self.previous_x_position = current_x_position
        self.previous_action = action.copy()
        
        # Log everything
        info['distance_traveled'] = distance_traveled
        info['current_velocity'] = instant_velocity
        info['custom_reward'] = custom_reward
        info['original_reward'] = original_reward
        info['z_position'] = z_position
        info['avg_torque'] = avg_torque if 'avg_torque' in locals() else 0
        info['success_achieved'] = (
            self.step_count >= self.TIME_THRESHOLD and 
            distance_traveled >= self.DISTANCE_THRESHOLD and
            instant_velocity <= self.MAX_VELOCITY
        )
        
        # Custom metrics for W&B
        info['custom_metrics/instant_velocity'] = instant_velocity
        info['custom_metrics/distance_traveled'] = distance_traveled
        info['custom_metrics/is_walking_slowly'] = instant_velocity <= self.MAX_VELOCITY
        info['custom_metrics/speed_penalty'] = max(0, instant_velocity - self.MAX_VELOCITY)
        info['custom_metrics/height'] = z_position
        info['custom_metrics/gait_quality'] = action_change if 'action_change' in locals() else 0
        
        return obs, custom_reward, terminated, truncated, info