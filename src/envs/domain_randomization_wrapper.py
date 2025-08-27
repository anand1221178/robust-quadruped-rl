#!/usr/bin/env python3
"""
Domain Randomization Wrapper for Robust Quadruped Locomotion
Implements actuator dropout and sensor noise as per research proposal
"""

import gymnasium as gym
import numpy as np
from typing import Dict, List, Optional, Tuple
import random

class DomainRandomizationWrapper(gym.Wrapper):
    """
    Domain Randomization wrapper implementing:
    1. Joint dropout/lock (actuator failures)
    2. Sensor noise (Gaussian noise on proprioceptive signals)
    
    Follows research proposal curriculum:
    - Phase 2: Single joint dropout + mild noise
    - Phase 3: Multiple dropouts + high noise
    """
    
    def __init__(self, env, dr_config: Dict):
        super().__init__(env)
        self.dr_config = dr_config
        
        # Joint dropout settings
        self.joint_dropout_prob = dr_config.get('joint_dropout_prob', 0.1)
        self.max_dropped_joints = dr_config.get('max_dropped_joints', 2)
        self.min_dropped_joints = dr_config.get('min_dropped_joints', 1)
        
        # Sensor noise settings
        self.sensor_noise_std = dr_config.get('sensor_noise_std', 0.05)
        self.noise_joints_only = dr_config.get('noise_joints_only', True)  # Per research proposal
        
        # Current episode state
        self.dropped_joints = []
        self.episode_count = 0
        
        # Action space is 8 joint torques for RealAnt
        self.num_joints = 8
        
        print(f"Domain Randomization enabled:")
        print(f"  - Joint dropout prob: {self.joint_dropout_prob}")
        print(f"  - Max dropped joints: {self.max_dropped_joints}")
        print(f"  - Sensor noise std: {self.sensor_noise_std}")
    
    def reset(self, **kwargs):
        """Reset environment and sample new randomization parameters"""
        obs, info = self.env.reset(**kwargs)
        
        # Sample joint dropouts for this episode
        self._sample_joint_dropouts()
        
        # Apply sensor noise to initial observation
        if self.sensor_noise_std > 0:
            obs = self._add_sensor_noise(obs)
        
        self.episode_count += 1
        
        # Add DR info to info dict
        if info is None:
            info = {}
        info['dropped_joints'] = self.dropped_joints.copy()
        info['sensor_noise_std'] = self.sensor_noise_std
        
        return obs, info
    
    def step(self, action):
        """Apply domain randomization to actions and observations"""
        
        # Apply joint dropout to actions (lock/disable joints)
        modified_action = self._apply_joint_dropout(action)
        
        # Take environment step with modified actions
        obs, reward, terminated, truncated, info = self.env.step(modified_action)
        
        # Add sensor noise to observations
        if self.sensor_noise_std > 0:
            obs = self._add_sensor_noise(obs)
        
        # Add DR info
        if info is None:
            info = {}
        info['original_action'] = action
        info['modified_action'] = modified_action
        info['dropped_joints'] = self.dropped_joints.copy()
        
        return obs, reward, terminated, truncated, info
    
    def _sample_joint_dropouts(self):
        """Sample which joints to drop for this episode"""
        self.dropped_joints = []
        
        if random.random() < self.joint_dropout_prob:
            # Decide how many joints to drop
            num_to_drop = random.randint(self.min_dropped_joints, self.max_dropped_joints)
            
            # Randomly select joints to drop
            available_joints = list(range(self.num_joints))
            self.dropped_joints = random.sample(available_joints, num_to_drop)
            
            if len(self.dropped_joints) > 0:
                print(f"Episode {self.episode_count}: Dropping joints {self.dropped_joints}")
    
    def _apply_joint_dropout(self, action):
        """Apply joint dropout by setting dropped joint actions to 0 (locked)"""
        if len(self.dropped_joints) == 0:
            return action
        
        modified_action = action.copy()
        
        # Lock dropped joints (set torque to 0)
        for joint_idx in self.dropped_joints:
            modified_action[joint_idx] = 0.0
        
        return modified_action
    
    def _add_sensor_noise(self, observation):
        """Add Gaussian noise to sensor readings"""
        if self.sensor_noise_std == 0:
            return observation
        
        obs_copy = observation.copy()
        
        if self.noise_joints_only:
            # Add noise only to joint sensors (proprioceptive signals)
            # Based on RealAnt observation structure:
            # dims 13-20: joint positions, dims 21-28: joint velocities  
            joint_obs_indices = list(range(13, 29))  # 16 joint sensor values
            
            for idx in joint_obs_indices:
                if idx < len(obs_copy):
                    obs_copy[idx] += np.random.normal(0, self.sensor_noise_std)
        else:
            # Add noise to all observations
            noise = np.random.normal(0, self.sensor_noise_std, size=obs_copy.shape)
            obs_copy += noise
        
        return obs_copy


class CurriculumDRWrapper(DomainRandomizationWrapper):
    """
    Curriculum-based Domain Randomization
    Gradually increases difficulty following research proposal phases
    """
    
    def __init__(self, env, dr_config: Dict):
        super().__init__(env, dr_config)
        
        # Curriculum settings
        self.phase_2_steps = dr_config.get('phase_2_steps', 5000000)  # 5M steps
        self.phase_3_steps = dr_config.get('phase_3_steps', 10000000) # 10M steps
        self.current_timestep = 0
        
        # Phase-specific parameters
        self.phase_2_config = {
            'joint_dropout_prob': 0.2,    # 20% chance
            'max_dropped_joints': 1,      # Single joint only
            'min_dropped_joints': 1,
            'sensor_noise_std': 0.02,     # Mild noise
        }
        
        self.phase_3_config = {
            'joint_dropout_prob': 0.4,    # 40% chance  
            'max_dropped_joints': 3,      # Up to 3 joints
            'min_dropped_joints': 1,
            'sensor_noise_std': 0.05,     # High noise
        }
        
        self._update_curriculum()
        print(f"Curriculum DR initialized. Current phase parameters:")
        print(f"  Phase 2 (0-{self.phase_2_steps}): Single joint + mild noise")
        print(f"  Phase 3 ({self.phase_2_steps}+): Multiple joints + high noise")
    
    def step(self, action):
        """Override to track timesteps for curriculum"""
        self.current_timestep += 1
        self._update_curriculum()
        return super().step(action)
    
    def _update_curriculum(self):
        """Update DR parameters based on training progress"""
        if self.current_timestep < self.phase_2_steps:
            # Phase 2: Single joint dropout + mild noise
            config = self.phase_2_config
        else:
            # Phase 3: Multiple dropouts + high noise
            config = self.phase_3_config
        
        # Update parameters
        self.joint_dropout_prob = config['joint_dropout_prob']
        self.max_dropped_joints = config['max_dropped_joints'] 
        self.min_dropped_joints = config['min_dropped_joints']
        self.sensor_noise_std = config['sensor_noise_std']