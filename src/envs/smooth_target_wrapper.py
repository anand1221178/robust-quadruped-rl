"""
Smooth Target Walking Wrapper
Combines goal-directed behavior with smooth locomotion
"""

import gymnasium as gym
import numpy as np

class SmoothTargetWrapper(gym.Wrapper):
    """
    Goal-directed walker that prioritizes SMOOTH locomotion
    
    Key differences from TargetWalkingWrapper:
    1. Smooth reward curves (no spikes)
    2. Action smoothness incentives  
    3. Direction guidance in observations
    4. Steady progress > frantic rushing
    """
    def __init__(self, env, target_distance=5.0):
        super().__init__(env)
        self.target_distance = target_distance
        self.target_x = target_distance
        self.initial_distance = 0
        self.previous_distance = 0
        self.previous_action = None
        self.episode_steps = 0
        self.dt = env.dt if hasattr(env, 'dt') else 0.01
        
        # Smooth walking targets (not aggressive speed targets)
        self.TARGET_VELOCITY = 0.3  # Reasonable walking speed
        self.MAX_VELOCITY = 1.0     # Don't encourage crazy speeds
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Set target ahead of robot
        current_x = self.env.unwrapped.data.qpos[0]
        self.target_x = current_x + self.target_distance
        self.initial_distance = self.target_distance
        self.previous_distance = self.target_distance
        self.previous_action = None
        self.episode_steps = 0
        
        # Add direction info to observation
        direction_to_target = np.array([1.0, 0.0])  # Always +X for now
        augmented_obs = np.concatenate([obs, direction_to_target])
        
        return augmented_obs, info
    
    @property
    def observation_space(self):
        """Extend observation space with direction info"""
        original_space = self.env.observation_space
        low = np.concatenate([original_space.low, [-1.0, -1.0]])
        high = np.concatenate([original_space.high, [1.0, 1.0]])
        return gym.spaces.Box(low=low, high=high, dtype=original_space.dtype)
    
    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1
        
        # Get current position
        current_x = self.env.unwrapped.data.qpos[0]
        current_z = self.env.unwrapped.data.qpos[2]
        
        # Calculate smooth progress metrics
        distance_to_target = abs(self.target_x - current_x)
        progress = self.previous_distance - distance_to_target
        
        # === SMOOTH REWARDS (no spikes) ===
        
        # 1. Steady progress reward (smooth curve)
        if progress > 0:
            progress_reward = np.tanh(progress * 20) * 10  # Smooth saturation
        else:
            progress_reward = progress * 5  # Linear penalty for going backwards
        
        # 2. Smooth velocity reward (target ~0.3 m/s)
        velocity = progress / self.dt if self.dt > 0 else 0
        velocity_diff = abs(velocity - self.TARGET_VELOCITY)
        velocity_reward = 5.0 * np.exp(-velocity_diff * 2)  # Gaussian around target
        
        # 3. Action smoothness bonus (KEY FOR SMOOTH WALKING!)
        smoothness_reward = 0
        if self.previous_action is not None:
            action_change = np.mean(np.abs(action - self.previous_action))
            smoothness_reward = 2.0 * np.exp(-action_change * 3)  # Reward small changes
        
        # 4. Gentle direction bonus (not urgent distance penalty)
        direction_reward = 0
        if distance_to_target < self.target_distance * 0.8:  # Getting close
            direction_reward = 1.0
        
        # 5. Height stability
        height_reward = 0
        if 0.15 < current_z < 0.35:
            height_reward = 0.5
        
        # 6. Success bonus (but not overwhelming)
        success_bonus = 0
        if distance_to_target < 0.5:  # Reached target
            success_bonus = 20  # Moderate bonus, not crazy spike
            # Move target further
            self.target_x = current_x + self.target_distance
            self.previous_distance = self.target_distance
            print(f"🎯 Target reached smoothly! New target at x={self.target_x:.2f}")
        
        # === COMBINE SMOOTH REWARDS ===
        custom_reward = (
            progress_reward +      # Smooth progress
            velocity_reward +      # Target speed
            smoothness_reward +    # Action consistency  
            direction_reward +     # Gentle guidance
            height_reward +        # Stability
            success_bonus         # Target achievement
        )
        
        # Small control penalty (don't be too aggressive)
        custom_reward -= 0.005 * np.sum(np.square(action))
        
        # Update for next step
        self.previous_distance = distance_to_target
        self.previous_action = action.copy()
        
        # Add direction info to observation
        direction_to_target = np.array([1.0, 0.0])  # Always +X for now
        augmented_obs = np.concatenate([obs, direction_to_target])
        
        # Enhanced info
        info.update({
            'distance_to_target': distance_to_target,
            'progress': progress,
            'velocity': velocity,
            'smoothness_score': smoothness_reward,
            'custom_reward': custom_reward,
            'target_reached': success_bonus > 0
        })
        
        return augmented_obs, custom_reward, terminated, truncated, info