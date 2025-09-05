"""
Permanent Domain Randomization Wrapper
Once a joint fails, it stays failed for the entire episode
This trains the robot to adapt to permanent disabilities
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class PermanentDRWrapper(gym.Wrapper):
    """
    Wrapper for PERMANENT joint failures during training.
    Once a joint fails, it remains disabled for the entire episode.
    """
    
    def __init__(
        self,
        env,
        failure_rate=0.001,  # Probability per step of a new permanent failure
        max_failed_joints=4,  # Maximum joints that can fail in one episode
        warmup_steps=1000000,  # Steps before failures begin
        curriculum_steps=10000000,  # Steps to gradually increase difficulty
        start_failures=0,  # Start with this many pre-failed joints
        end_failures=4,  # End with up to this many failures
        failure_types=['lock'],  # Only lock for permanent (no weak/noise)
        verbose=False
    ):
        super().__init__(env)
        
        self.failure_rate = failure_rate
        self.max_failed_joints = max_failed_joints
        self.warmup_steps = warmup_steps
        self.curriculum_steps = curriculum_steps
        self.start_failures = start_failures
        self.end_failures = end_failures
        self.failure_types = failure_types
        self.verbose = verbose
        
        # Get action space dimensions (number of joints)
        self.num_joints = env.action_space.shape[0]
        
        # Track permanent failures for current episode
        self.failed_joints = set()  # Set of permanently failed joint indices
        self.failure_history = {}  # When each joint failed
        
        # Track training progress
        self.total_steps = 0
        self.episode_count = 0
        
        # Statistics
        self.stats = {
            'total_failures': 0,
            'episodes_with_failures': 0,
            'max_failures_in_episode': 0,
            'avg_failures_per_episode': 0.0
        }
    
    def reset(self, **kwargs):
        """Reset environment and potentially start with some failed joints"""
        obs = self.env.reset(**kwargs)
        
        # Clear previous episode's failures
        self.failed_joints = set()
        self.failure_history = {}
        self.episode_count += 1
        
        # Curriculum learning: determine initial failures based on training progress
        if self.total_steps > self.warmup_steps:
            # Calculate curriculum progress
            curriculum_progress = min(1.0, 
                (self.total_steps - self.warmup_steps) / self.curriculum_steps)
            
            # Determine number of initial failures
            max_initial = int(self.start_failures + 
                            (self.end_failures - self.start_failures) * curriculum_progress)
            
            # Randomly fail some joints at episode start (curriculum)
            if max_initial > 0 and np.random.random() < 0.3:  # 30% chance of starting with failures
                num_initial = np.random.randint(0, min(max_initial, self.max_failed_joints) + 1)
                initial_joints = np.random.choice(self.num_joints, num_initial, replace=False)
                for joint_idx in initial_joints:
                    self.failed_joints.add(joint_idx)
                    self.failure_history[joint_idx] = 0
                
                if self.verbose and num_initial > 0:
                    print(f"Episode {self.episode_count}: Starting with {num_initial} failed joints: {initial_joints}")
        
        # Update statistics
        if self.failed_joints:
            self.stats['episodes_with_failures'] += 1
        
        return obs
    
    def step(self, action):
        """Apply permanent failures to action before stepping"""
        self.total_steps += 1
        
        # Copy action to avoid modifying original
        modified_action = action.copy()
        
        # Apply new permanent failures (if not at max)
        if len(self.failed_joints) < self.max_failed_joints:
            if self.total_steps > self.warmup_steps:
                # Check for new failures
                for joint_idx in range(self.num_joints):
                    if joint_idx not in self.failed_joints:
                        if np.random.random() < self.failure_rate:
                            # New permanent failure!
                            self.failed_joints.add(joint_idx)
                            self.failure_history[joint_idx] = self.total_steps
                            self.stats['total_failures'] += 1
                            
                            if self.verbose:
                                print(f"Step {self.total_steps}: Joint {joint_idx} PERMANENTLY FAILED! "
                                      f"Total failed: {len(self.failed_joints)}/{self.num_joints}")
        
        # Apply all permanent failures
        for joint_idx in self.failed_joints:
            # Permanently disabled joints output zero torque
            modified_action[joint_idx] = 0.0
        
        # Step the environment with modified action
        result = self.env.step(modified_action)
        if len(result) == 5:  # New gymnasium format
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:  # Old gym format
            obs, reward, done, info = result
        
        # Add failure info to the info dict
        info['permanent_failures'] = list(self.failed_joints)
        info['num_failed_joints'] = len(self.failed_joints)
        info['failure_rate'] = len(self.failed_joints) / self.num_joints
        
        # Modify reward to encourage adaptation to failures
        if len(self.failed_joints) > 0:
            # Bonus for maintaining forward progress with disabilities
            # The more failures, the higher the bonus multiplier
            disability_bonus = 1.0 + (0.2 * len(self.failed_joints))
            
            # Only apply bonus if moving forward
            if 'current_velocity' in info and info['current_velocity'] > 0:
                reward *= disability_bonus
            elif 'speed' in info and info['speed'] > 0:
                reward *= disability_bonus
        
        # Update max failures statistic
        self.stats['max_failures_in_episode'] = max(
            self.stats['max_failures_in_episode'],
            len(self.failed_joints)
        )
        
        return obs, reward, done, info
    
    def get_training_stats(self):
        """Get training statistics"""
        if self.episode_count > 0:
            self.stats['avg_failures_per_episode'] = (
                self.stats['total_failures'] / self.episode_count
            )
        return self.stats
    
    def set_training_progress(self, total_steps):
        """Update training progress for curriculum learning"""
        self.total_steps = total_steps


class PermanentDRCurriculumWrapper(PermanentDRWrapper):
    """
    Advanced version with sophisticated curriculum learning
    """
    
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        
        # Curriculum stages
        self.curriculum_stages = [
            # (steps, max_failures, failure_rate, description)
            (0,        0, 0.000, "No failures - learn basic walking"),
            (2000000,  1, 0.001, "Single joint failures"),
            (4000000,  2, 0.002, "Two joint failures"),
            (6000000,  3, 0.003, "Three joint failures"),
            (8000000,  4, 0.004, "Four joint failures"),
            (10000000, 5, 0.005, "Five joint failures (extreme)"),
        ]
        
        self.current_stage = 0
    
    def get_current_stage(self):
        """Determine current curriculum stage based on training steps"""
        for i, (steps, max_fail, rate, desc) in enumerate(self.curriculum_stages):
            if self.total_steps >= steps:
                self.current_stage = i
        
        stage = self.curriculum_stages[self.current_stage]
        return {
            'stage': self.current_stage,
            'max_failures': stage[1],
            'failure_rate': stage[2],
            'description': stage[3]
        }
    
    def reset(self, **kwargs):
        """Reset with curriculum-appropriate difficulty"""
        obs = self.env.reset(**kwargs)
        
        # Get current stage settings
        stage_info = self.get_current_stage()
        self.max_failed_joints = stage_info['max_failures']
        self.failure_rate = stage_info['failure_rate']
        
        # Clear failures
        self.failed_joints = set()
        self.failure_history = {}
        self.episode_count += 1
        
        # Start with failures based on curriculum
        if self.max_failed_joints > 0 and np.random.random() < 0.2:
            # 20% chance to start with some failures
            num_initial = np.random.randint(0, min(2, self.max_failed_joints) + 1)
            initial_joints = np.random.choice(self.num_joints, num_initial, replace=False)
            for joint_idx in initial_joints:
                self.failed_joints.add(joint_idx)
                self.failure_history[joint_idx] = 0
            
            if self.verbose:
                print(f"[Stage {self.current_stage}] Episode {self.episode_count}: "
                      f"Starting with {num_initial} failed joints")
        
        return obs