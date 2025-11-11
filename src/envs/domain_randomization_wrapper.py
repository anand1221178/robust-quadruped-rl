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


class CurriculumDRWrapper(gym.Wrapper):

    
    def __init__(self, env, config: Dict):
        # Initialize the wrapper properly
        super().__init__(env)
        self.config = config
        self.dr_config = config.get('domain_randomization', {})
        
        # 🎯 3-PHASE CURRICULUM SETUP (CONFIGURABLE!)
        self.phase_1_steps = self.dr_config.get('phase_1_steps', 8000000)   # Phase 1: Clean training
        self.phase_2_steps = self.dr_config.get('phase_2_steps', 8000000)   # Phase 2: Single failures
        self.phase_3_steps = self.dr_config.get('phase_3_steps', 9000000)   # Phase 3: Multiple failures
        self.current_timestep = 0
        self.current_phase = 1
        
        # Action space is 8 joint torques for RealAnt
        self.num_joints = 8
        self.dropped_joints = []
        self.episode_count = 0
        
        # 🔥 PHASE CONFIGURATIONS (FROM CONFIG FILE!)
        self.phase_1_config = self.dr_config.get('phase_1_config', {
            'joint_dropout_prob': 0.0,    # NO FAILURES - perfect learning
            'max_dropped_joints': 0,
            'min_dropped_joints': 0,
            'sensor_noise_std': 0.0,      # NO NOISE - clean signals
        })

        self.phase_2_config = self.dr_config.get('phase_2_config', {
            'joint_dropout_prob': 0.05,   # 5% single failures
            'max_dropped_joints': 1,      # Single joint only
            'min_dropped_joints': 1,
            'sensor_noise_std': 0.01,     # Mild noise
        })

        self.phase_3_config = self.dr_config.get('phase_3_config', {
            'joint_dropout_prob': 0.15,   # 15% multiple failures
            'max_dropped_joints': 2,      # Up to 2 joints
            'min_dropped_joints': 1,
            'sensor_noise_std': 0.03,     # High noise
        })
        
        # Initialize with Phase 1 (clean training)
        self._update_curriculum()
        
        print(f"ULTIMATE 3-PHASE CURRICULUM DR INITIALIZED! 🔥")
        print(f"Phase 1 (0-{self.phase_1_steps:,} steps): CLEAN TRAINING - Perfect locomotion learning")
        print(f"Phase 2 ({self.phase_1_steps:,}-{self.phase_1_steps + self.phase_2_steps:,} steps): Single failures + mild noise")  
        print(f"Phase 3 ({self.phase_1_steps + self.phase_2_steps:,}+ steps): Multiple failures + high noise")
        print(f"Current Phase: {self.current_phase} - {self._get_phase_description()}")
    
    def step(self, action):
        """Override to track timesteps for 3-phase curriculum"""
        self.current_timestep += 1
        self._update_curriculum()
        
        # Apply joint dropout to actions (lock/disable joints)  
        modified_action = self._apply_joint_dropout(action)
        
        # Take environment step with modified actions
        obs, reward, terminated, truncated, info = self.env.step(modified_action)
        
        # Add sensor noise to observations
        if hasattr(self, 'sensor_noise_std') and self.sensor_noise_std > 0:
            obs = self._add_sensor_noise(obs)
        
        # Add DR info
        if info is None:
            info = {}
        info['original_action'] = action
        info['modified_action'] = modified_action
        info['dropped_joints'] = self.dropped_joints.copy()
        info['curriculum_phase'] = self.current_phase
        info['curriculum_progress'] = f"{self.current_timestep:,} steps"
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset environment and sample new randomization parameters"""
        obs, info = self.env.reset(**kwargs)
        
        # Sample joint dropouts for this episode (based on current phase)
        self._sample_joint_dropouts()
        
        # Apply sensor noise to initial observation
        if hasattr(self, 'sensor_noise_std') and self.sensor_noise_std > 0:
            obs = self._add_sensor_noise(obs)
        
        self.episode_count += 1
        
        # Add DR info to info dict
        if info is None:
            info = {}
        info['dropped_joints'] = self.dropped_joints.copy()
        info['sensor_noise_std'] = getattr(self, 'sensor_noise_std', 0.0)
        info['curriculum_phase'] = self.current_phase
        
        return obs, info
    
    def _update_curriculum(self):
        """🔥 ULTIMATE 3-PHASE CURRICULUM UPDATE! 🔥"""
        old_phase = self.current_phase

        # Determine current phase based on timesteps
        if self.current_timestep < self.phase_1_steps:
            # Phase 1: Clean training
            self.current_phase = 1
            config = self.phase_1_config
        elif self.current_timestep < self.phase_1_steps + self.phase_2_steps:
            # Phase 2: Single failures + mild noise
            self.current_phase = 2
            config = self.phase_2_config
        elif self.phase_3_steps > 0:
            # Phase 3: Multiple failures + high noise (only if enabled)
            self.current_phase = 3
            config = self.phase_3_config
        else:
            # Phase 3 disabled - stay in Phase 2
            self.current_phase = 2
            config = self.phase_2_config
        
        # Update parameters from current phase config
        self.joint_dropout_prob = config['joint_dropout_prob']
        self.max_dropped_joints = config['max_dropped_joints'] 
        self.min_dropped_joints = config['min_dropped_joints']
        self.sensor_noise_std = config['sensor_noise_std']
        self.noise_joints_only = config.get('noise_joints_only', True)
        
        # 🎉 PHASE TRANSITION CELEBRATION!
        if old_phase != self.current_phase:
            print(f"\nPHASE TRANSITION! {old_phase} → {self.current_phase} at {self.current_timestep:,} steps!")
            print(f"NEW PHASE: {self._get_phase_description()}")
            print(f"Joint dropout: {self.joint_dropout_prob:.1%}")
            print(f"Max joints: {self.max_dropped_joints}")  
            print(f"Sensor noise: {self.sensor_noise_std:.3f}")
            print("=" * 60)
    
    def _get_phase_description(self):
        """Get description of current phase"""
        if self.current_phase == 1:
            return "CLEAN TRAINING - Learning perfect locomotion"
        elif self.current_phase == 2:
            return "SINGLE FAILURES - Building basic robustness"
        else:
            return "MULTIPLE FAILURES - Ultimate robustness challenge"
    
    def _sample_joint_dropouts(self):
        """Sample which joints to drop for this episode"""
        self.dropped_joints = []

        if random.random() < self.joint_dropout_prob:
            if self.max_dropped_joints > 0:
                num_to_drop = random.randint(self.min_dropped_joints, self.max_dropped_joints)

                # Check for weighted joint sampling (V7.11 rotation mastery)
                current_phase_config = getattr(self, f'phase_{self.current_phase}_config', {})
                joint_weights = current_phase_config.get('joint_weights', None)

                if joint_weights:
                    # --- Weighted Joint Sampling ---
                    joint_names = ['hip_1', 'ankle_1', 'hip_2', 'ankle_2', 'hip_3', 'ankle_3', 'hip_4', 'ankle_4']
                    weights = []

                    for i, joint_name in enumerate(joint_names):
                        weights.append(joint_weights.get(joint_name, 1.0/len(joint_names)))

                    # Normalize weights
                    total_weight = sum(weights)
                    if total_weight > 0:
                        weights = [w / total_weight for w in weights]
                    else:
                        weights = [1.0/len(joint_names) for _ in joint_names]

                    # Sample based on weights
                    self.dropped_joints = random.choices(
                        range(self.num_joints),
                        weights=weights,
                        k=num_to_drop
                    )

                    # Remove duplicates if any
                    self.dropped_joints = list(set(self.dropped_joints))

                    # If we lost joints due to duplicates, sample more
                    while len(self.dropped_joints) < num_to_drop:
                        additional = random.choices(
                            range(self.num_joints),
                            weights=weights,
                            k=1
                        )[0]
                        if additional not in self.dropped_joints:
                            self.dropped_joints.append(additional)

                else:
                    # --- Original Random Sampling ---
                    available_joints = list(range(self.num_joints))
                    self.dropped_joints = random.sample(available_joints, num_to_drop)
                

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