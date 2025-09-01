"""
Robust Domain Randomization Wrapper with proper joint locking and surprise training
"""

import gymnasium as gym
import numpy as np
import random
from typing import Dict, Any, Optional, List

class RobustDRWrapper(gym.Wrapper):
    """
    Enhanced Domain Randomization with:
    - Proper joint locking (position control, not torque=0)
    - Progressive fault curriculum
    - Surprise training mode
    - Realistic fault modeling
    """
    
    def __init__(
        self,
        env,
        config: Dict[str, Any]
    ):
        super().__init__(env)
        
        # Core DR parameters
        self.joint_fault_prob = config.get('joint_fault_prob', 0.2)
        self.sensor_noise_std = config.get('sensor_noise_std', 0.02)
        self.max_faulty_joints = config.get('max_faulty_joints', 2)
        self.min_faulty_joints = config.get('min_faulty_joints', 1)
        
        # Fault types
        self.fault_types = config.get('fault_types', ['lock', 'weak', 'delay'])
        self.fault_type_probs = config.get('fault_type_probs', [0.5, 0.3, 0.2])
        
        # Surprise training mode
        self.surprise_mode = config.get('surprise_mode', False)
        self.surprise_prob = config.get('surprise_prob', 0.1)  # 10% chaos
        self.normal_fault_prob = config.get('normal_fault_prob', 0.05)  # 5% mild faults
        
        # Progressive curriculum
        self.use_curriculum = config.get('use_curriculum', True)
        self.curriculum_steps = config.get('curriculum_steps', 5000000)
        self.warmup_steps = config.get('warmup_steps', 1000000)
        
        # Initialize state
        self.num_joints = 8  # RealAnt has 8 actuated joints
        self.faulty_joints = {}  # Dict of joint_id: fault_type
        self.locked_positions = {}  # For position-controlled locked joints
        self.joint_health = np.ones(self.num_joints)  # Health factor per joint
        self.episode_count = 0
        self.total_steps = 0
        
        # For tracking original joint positions when locking
        self.last_joint_positions = None
        
    def reset(self, **kwargs):
        """Reset environment and sample new faults"""
        obs, info = self.env.reset(**kwargs)
        
        self.episode_count += 1
        self.faulty_joints = {}
        self.locked_positions = {}
        self.joint_health = np.ones(self.num_joints)
        
        # Get initial joint positions from observation
        # RealAnt obs: dims 13-20 are joint positions
        if len(obs) >= 21:
            self.last_joint_positions = obs[13:21].copy()
        else:
            self.last_joint_positions = np.zeros(self.num_joints)
        
        # Sample faults for this episode
        self._sample_episode_faults()
        
        # Apply sensor noise to initial observation
        if self.sensor_noise_std > 0:
            obs = self._add_sensor_noise(obs)
        
        # Add DR info to info dict
        info = info or {}
        info['faulty_joints'] = self.faulty_joints.copy()
        info['joint_health'] = self.joint_health.copy()
        info['sensor_noise_std'] = self.get_current_noise_std()
        
        return obs, info
    
    def step(self, action):
        """Apply domain randomization to actions"""
        self.total_steps += 1
        
        # Update last known joint positions from current observation
        if hasattr(self.env, 'sim') and hasattr(self.env.sim, 'data'):
            # Get current joint positions directly from sim
            self.last_joint_positions = self.env.sim.data.qpos[7:15].copy()  # Skip root joints
        
        # Apply joint faults to actions
        modified_action = self._apply_joint_faults(action)
        
        # Take environment step
        obs, reward, terminated, truncated, info = self.env.step(modified_action)
        
        # Update joint positions from new observation
        if len(obs) >= 21:
            self.last_joint_positions = obs[13:21].copy()
        
        # Add sensor noise
        if self.get_current_noise_std() > 0:
            obs = self._add_sensor_noise(obs)
        
        # Surprise catastrophic failure injection
        if self.surprise_mode and not self.faulty_joints:
            if random.random() < 0.001:  # 0.1% chance per step
                self._inject_surprise_failure()
                print(f"💥 SURPRISE FAILURE at step {self.total_steps}!")
        
        # Add DR info
        info = info or {}
        info['original_action'] = action
        info['modified_action'] = modified_action
        info['faulty_joints'] = self.faulty_joints.copy()
        info['joint_health'] = self.joint_health.copy()
        
        return obs, reward, terminated, truncated, info
    
    def _sample_episode_faults(self):
        """Sample faults for this episode based on curriculum"""
        
        # Get current fault probability based on curriculum
        fault_prob = self._get_curriculum_fault_prob()
        
        if random.random() < fault_prob:
            # Decide how many joints to affect
            num_faults = random.randint(self.min_faulty_joints, 
                                       min(self.max_faulty_joints, self._get_curriculum_max_faults()))
            
            # Sample which joints
            available_joints = list(range(self.num_joints))
            faulty_joint_ids = random.sample(available_joints, num_faults)
            
            # Assign fault types
            for joint_id in faulty_joint_ids:
                fault_type = np.random.choice(self.fault_types, p=self.fault_type_probs)
                self.faulty_joints[joint_id] = fault_type
                
                if fault_type == 'lock':
                    # Lock at current position
                    self.locked_positions[joint_id] = self.last_joint_positions[joint_id]
                    self.joint_health[joint_id] = 0.0  # Complete failure
                    
                elif fault_type == 'weak':
                    # Reduced strength (30-70% of normal)
                    self.joint_health[joint_id] = random.uniform(0.3, 0.7)
                    
                elif fault_type == 'delay':
                    # Delayed response (80-90% effectiveness)
                    self.joint_health[joint_id] = random.uniform(0.8, 0.9)
            
            joint_names = self._get_joint_names()
            fault_desc = [f"{joint_names[j]}:{self.faulty_joints[j]}" for j in self.faulty_joints]
            print(f"Episode {self.episode_count}: Faults = {fault_desc}")
    
    def _apply_joint_faults(self, action):
        """Apply various fault types to joint actions"""
        if not self.faulty_joints:
            return action
        
        modified_action = action.copy()
        
        for joint_id, fault_type in self.faulty_joints.items():
            if fault_type == 'lock':
                # Position control: try to maintain locked position
                # This creates resistance, not just zero torque
                if joint_id in self.locked_positions:
                    # Create corrective torque to maintain position
                    current_pos = self.last_joint_positions[joint_id]
                    locked_pos = self.locked_positions[joint_id]
                    position_error = locked_pos - current_pos
                    
                    # Strong corrective torque (like a seized joint)
                    modified_action[joint_id] = np.clip(position_error * 50.0, -1.0, 1.0)
                    
            elif fault_type == 'weak':
                # Reduced motor strength
                modified_action[joint_id] *= self.joint_health[joint_id]
                
            elif fault_type == 'delay':
                # Delayed/sluggish response (simplified version)
                modified_action[joint_id] *= self.joint_health[joint_id]
                
            elif fault_type == 'noise':
                # Noisy/jittery motor
                modified_action[joint_id] += np.random.normal(0, 0.1)
                modified_action[joint_id] = np.clip(modified_action[joint_id], -1.0, 1.0)
        
        return modified_action
    
    def _add_sensor_noise(self, observation):
        """Add realistic sensor noise to observations"""
        obs_copy = observation.copy()
        noise_std = self.get_current_noise_std()
        
        # Only add noise to joint sensors (dims 13-28 in RealAnt)
        # Position sensors (13-20) - less noise
        # Velocity sensors (21-28) - more noise
        
        if len(obs_copy) >= 29:
            # Joint positions - small noise
            position_noise = np.random.normal(0, noise_std * 0.5, 8)
            obs_copy[13:21] += position_noise
            
            # Joint velocities - larger noise (velocity sensors are noisier)
            velocity_noise = np.random.normal(0, noise_std, 8)
            obs_copy[21:29] += velocity_noise
        
        return obs_copy
    
    def _inject_surprise_failure(self):
        """Inject sudden catastrophic failure during episode"""
        # Random severe fault
        num_faults = random.randint(2, min(4, self.num_joints))
        available_joints = [j for j in range(self.num_joints) if j not in self.faulty_joints]
        
        if len(available_joints) >= num_faults:
            new_faulty = random.sample(available_joints, num_faults)
            for joint_id in new_faulty:
                # Mostly locks for catastrophic failures
                fault_type = np.random.choice(['lock', 'weak'], p=[0.7, 0.3])
                self.faulty_joints[joint_id] = fault_type
                
                if fault_type == 'lock':
                    self.locked_positions[joint_id] = self.last_joint_positions[joint_id]
                    self.joint_health[joint_id] = 0.0
                else:
                    self.joint_health[joint_id] = random.uniform(0.1, 0.3)  # Severe weakness
    
    def _get_curriculum_fault_prob(self):
        """Get fault probability based on curriculum progress"""
        if not self.use_curriculum:
            return self.joint_fault_prob
        
        if self.total_steps < self.warmup_steps:
            return 0.0  # No faults during warmup
        
        # Progressive increase
        progress = min(1.0, (self.total_steps - self.warmup_steps) / self.curriculum_steps)
        
        if self.surprise_mode:
            # In surprise mode: usually low faults, occasionally high
            if random.random() < self.surprise_prob:
                return min(0.8, self.joint_fault_prob * 3)  # Surprise! Many faults
            else:
                return self.normal_fault_prob * progress
        else:
            # Standard progressive curriculum
            return self.joint_fault_prob * progress
    
    def _get_curriculum_max_faults(self):
        """Get max simultaneous faults based on curriculum"""
        if not self.use_curriculum:
            return self.max_faulty_joints
        
        if self.total_steps < self.warmup_steps:
            return 0
        
        progress = min(1.0, (self.total_steps - self.warmup_steps) / self.curriculum_steps)
        
        # Start with 1 fault, progress to max
        return max(1, int(self.max_faulty_joints * progress))
    
    def get_current_noise_std(self):
        """Get current sensor noise level based on curriculum"""
        if not self.use_curriculum:
            return self.sensor_noise_std
        
        if self.total_steps < self.warmup_steps:
            return 0.0
        
        progress = min(1.0, (self.total_steps - self.warmup_steps) / self.curriculum_steps)
        return self.sensor_noise_std * progress
    
    def _get_joint_names(self):
        """Get human-readable joint names"""
        return [
            'front_left_hip', 'front_left_knee',
            'front_right_hip', 'front_right_knee',
            'back_left_hip', 'back_left_knee', 
            'back_right_hip', 'back_right_knee'
        ]