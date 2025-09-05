"""
Persistent Domain Randomization Wrapper with Realistic Failure Durations
Failures persist for extended periods (not single timesteps) to simulate real hardware issues
"""

import gymnasium as gym
import numpy as np
import random
from typing import Dict, Any, Optional, Tuple

class PersistentDRWrapper(gym.Wrapper):
    """
    Realistic Domain Randomization with persistent failures:
    - Short failures: 50-200 steps (transient issues)
    - Medium failures: 200-1000 steps (temporary damage)
    - Long failures: Entire episode (permanent for that episode)
    
    This is more realistic than single-timestep failures which don't exist in real hardware.
    """
    
    def __init__(
        self,
        env,
        config: Dict[str, Any]
    ):
        super().__init__(env)
        
        # Failure probability and severity
        self.failure_prob = config.get('failure_prob', 0.15)  # 15% chance per episode
        self.max_failed_joints = config.get('max_failed_joints', 2)
        
        # Failure duration distributions (in timesteps)
        self.duration_probs = config.get('duration_probs', [0.4, 0.4, 0.2])  # [short, medium, long]
        self.short_duration_range = config.get('short_duration', [50, 200])
        self.medium_duration_range = config.get('medium_duration', [200, 1000])
        # Long duration = entire episode
        
        # Failure types and their probabilities
        self.failure_types = config.get('failure_types', ['lock', 'weak', 'erratic'])
        self.failure_type_probs = config.get('failure_type_probs', [0.5, 0.3, 0.2])
        
        # Progressive curriculum
        self.use_curriculum = config.get('use_curriculum', True)
        self.curriculum_steps = config.get('curriculum_steps', 15000000)  # 15M steps
        self.warmup_steps = config.get('warmup_steps', 8000000)  # 8M warmup
        
        # State tracking
        self.num_joints = 8  # RealAnt has 8 joints
        self.current_failures = {}  # {joint_id: (failure_type, remaining_duration)}
        self.locked_positions = {}  # For locked joints
        self.episode_steps = 0
        self.total_steps = 0
        self.episode_count = 0
        
        # For tracking joint positions
        self.last_joint_positions = np.zeros(self.num_joints)
        
    def reset(self, **kwargs):
        """Reset environment and sample new persistent failures"""
        obs, info = self.env.reset(**kwargs)
        
        self.episode_steps = 0
        self.episode_count += 1
        self.current_failures = {}
        self.locked_positions = {}
        
        # Extract initial joint positions
        if len(obs) >= 21:
            self.last_joint_positions = obs[13:21].copy()
        
        # Sample failures for this episode (they will persist!)
        if self._should_apply_failures():
            self._sample_persistent_failures()
        
        # Add failure info
        info = info or {}
        info['persistent_failures'] = self._get_failure_summary()
        
        return obs, info
    
    def step(self, action):
        """Apply persistent failures to actions"""
        self.episode_steps += 1
        self.total_steps += 1
        
        # Update joint positions from observation
        if hasattr(self.env, 'sim') and hasattr(self.env.sim, 'data'):
            self.last_joint_positions = self.env.sim.data.qpos[7:15].copy()
        
        # Apply persistent failures to action
        modified_action = self._apply_persistent_failures(action)
        
        # Update failure durations
        self._update_failure_durations()
        
        # Take environment step
        obs, reward, terminated, truncated, info = self.env.step(modified_action)
        
        # Update joint positions from new observation
        if len(obs) >= 21:
            self.last_joint_positions = obs[13:21].copy()
        
        # Add failure info
        info = info or {}
        info['persistent_failures'] = self._get_failure_summary()
        info['failure_steps_remaining'] = {j: d for j, (_, d) in self.current_failures.items()}
        
        return obs, reward, terminated, truncated, info
    
    def _should_apply_failures(self) -> bool:
        """Determine if failures should be applied based on curriculum"""
        if not self.use_curriculum:
            return random.random() < self.failure_prob
        
        if self.total_steps < self.warmup_steps:
            # During warmup, very low failure rate
            return random.random() < 0.02
        
        # Progressive increase in failure probability
        progress = min(1.0, (self.total_steps - self.warmup_steps) / self.curriculum_steps)
        current_prob = 0.02 + (self.failure_prob - 0.02) * progress
        
        return random.random() < current_prob
    
    def _sample_persistent_failures(self):
        """Sample failures that will persist for extended durations"""
        num_failed = random.randint(1, min(self.max_failed_joints, self.num_joints))
        failed_joints = random.sample(range(self.num_joints), num_failed)
        
        for joint_id in failed_joints:
            # Sample failure type
            failure_type = np.random.choice(
                self.failure_types,
                p=self.failure_type_probs
            )
            
            # Sample duration category
            duration_category = np.random.choice(
                ['short', 'medium', 'long'],
                p=self.duration_probs
            )
            
            # Get actual duration
            if duration_category == 'short':
                duration = random.randint(*self.short_duration_range)
            elif duration_category == 'medium':
                duration = random.randint(*self.medium_duration_range)
            else:  # long = entire episode
                duration = float('inf')  # Will last entire episode
            
            self.current_failures[joint_id] = (failure_type, duration)
            
            # If locked, store the locked position
            if failure_type == 'lock':
                self.locked_positions[joint_id] = self.last_joint_positions[joint_id]
        
        # Log failures
        if self.current_failures:
            failure_desc = []
            for j, (ftype, dur) in self.current_failures.items():
                dur_str = "episode" if dur == float('inf') else f"{dur} steps"
                failure_desc.append(f"Joint{j}:{ftype}({dur_str})")
            print(f"Episode {self.episode_count}: Persistent failures = {failure_desc}")
    
    def _apply_persistent_failures(self, action):
        """Apply failures that persist over time"""
        if not self.current_failures:
            return action
        
        modified_action = action.copy()
        
        for joint_id, (failure_type, remaining_duration) in self.current_failures.items():
            if remaining_duration <= 0:
                continue  # Failure has expired
            
            if failure_type == 'lock':
                # Joint is locked at position - strong resistance to movement
                if joint_id in self.locked_positions:
                    target_pos = self.locked_positions[joint_id]
                    current_pos = self.last_joint_positions[joint_id]
                    
                    # Strong corrective torque to maintain locked position
                    position_error = target_pos - current_pos
                    modified_action[joint_id] = np.clip(position_error * 10.0, -1.0, 1.0)
                    
            elif failure_type == 'weak':
                # Reduced motor strength (20-40% of normal)
                modified_action[joint_id] *= random.uniform(0.2, 0.4)
                
            elif failure_type == 'erratic':
                # Erratic behavior - random noise added
                modified_action[joint_id] += random.uniform(-0.3, 0.3)
                modified_action[joint_id] = np.clip(modified_action[joint_id], -1.0, 1.0)
        
        return modified_action
    
    def _update_failure_durations(self):
        """Decrement failure durations"""
        expired_joints = []
        
        for joint_id, (failure_type, duration) in self.current_failures.items():
            if duration != float('inf'):  # Not an episode-long failure
                new_duration = duration - 1
                if new_duration <= 0:
                    expired_joints.append(joint_id)
                else:
                    self.current_failures[joint_id] = (failure_type, new_duration)
        
        # Remove expired failures
        for joint_id in expired_joints:
            del self.current_failures[joint_id]
            if joint_id in self.locked_positions:
                del self.locked_positions[joint_id]
            print(f"  Joint {joint_id} failure expired at step {self.episode_steps}")
    
    def _get_failure_summary(self) -> Dict[str, Any]:
        """Get summary of current failures"""
        return {
            'num_failed_joints': len(self.current_failures),
            'failed_joints': list(self.current_failures.keys()),
            'failure_types': {j: f[0] for j, f in self.current_failures.items()},
            'remaining_durations': {j: f[1] if f[1] != float('inf') else 'episode' 
                                   for j, f in self.current_failures.items()}
        }