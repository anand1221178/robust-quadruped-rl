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
    """
    🔥 ULTIMATE 3-PHASE CURRICULUM DR - RESEARCH PROPOSAL COMPLIANT 🔥
    Phase 1: Clean training (learn perfect locomotion)
    Phase 2: Single failures + mild noise  
    Phase 3: Multiple failures + high noise
    """
    
    def __init__(self, env, config: Dict):
        # Initialize the wrapper properly
        super().__init__(env)
        self.config = config
        self.dr_config = config.get('domain_randomization', {})
        self.symmetric_config = config.get('env', {}).get('symmetric_training', {})

        # V7.10: Symmetric sampling settings
        self.symmetric_failure_sampling = self.symmetric_config.get('symmetric_failure_sampling', False)
        self.paired_joints_names = self.symmetric_config.get('paired_joints', [])
        self.joint_to_idx = {
            'hip_1': 0, 'ankle_1': 1, 'hip_2': 2, 'ankle_2': 3,
            'hip_3': 4, 'ankle_3': 5, 'hip_4': 6, 'ankle_4': 7,
        }
        self.paired_joint_indices = [
            (self.joint_to_idx[p[0]], self.joint_to_idx[p[1]]) for p in self.paired_joints_names
        ]
        
        # 🎯 3-PHASE CURRICULUM SETUP (CONFIGURABLE!)
        self.phase_1_steps = self.dr_config.get('phase_1_steps', 8000000)   # Phase 1: Clean training
        self.phase_2_steps = dr_config.get('phase_2_steps', 8000000)   # Phase 2: Single failures
        self.phase_3_steps = dr_config.get('phase_3_steps', 9000000)   # Phase 3: Multiple failures
        self.current_timestep = 0
        self.current_phase = 1
        
        # Action space is 8 joint torques for RealAnt
        self.num_joints = 8
        self.dropped_joints = []
        self.episode_count = 0
        
        # 🔥 PHASE CONFIGURATIONS (FROM CONFIG FILE!)
        self.phase_1_config = dr_config.get('phase_1_config', {
            'joint_dropout_prob': 0.0,    # NO FAILURES - perfect learning
            'max_dropped_joints': 0,
            'min_dropped_joints': 0, 
            'sensor_noise_std': 0.0,      # NO NOISE - clean signals
        })
        
        self.phase_2_config = dr_config.get('phase_2_config', {
            'joint_dropout_prob': 0.05,   # 5% single failures
            'max_dropped_joints': 1,      # Single joint only
            'min_dropped_joints': 1,
            'sensor_noise_std': 0.01,     # Mild noise
        })
        
        self.phase_3_config = dr_config.get('phase_3_config', {
            'joint_dropout_prob': 0.15,   # 15% multiple failures  
            'max_dropped_joints': 2,      # Up to 2 joints
            'min_dropped_joints': 1,
            'sensor_noise_std': 0.03,     # High noise
        })
        
        # Initialize with Phase 1 (clean training)
        self._update_curriculum()
        
        print(f"🔥 ULTIMATE 3-PHASE CURRICULUM DR INITIALIZED! 🔥")
        print(f"📚 Phase 1 (0-{self.phase_1_steps:,} steps): CLEAN TRAINING - Perfect locomotion learning")
        print(f"⚡ Phase 2 ({self.phase_1_steps:,}-{self.phase_1_steps + self.phase_2_steps:,} steps): Single failures + mild noise")  
        print(f"🚀 Phase 3 ({self.phase_1_steps + self.phase_2_steps:,}+ steps): Multiple failures + high noise")
        print(f"🎯 Current Phase: {self.current_phase} - {self._get_phase_description()}")
    
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
        else:
            # Phase 3: Multiple failures + high noise
            self.current_phase = 3
            config = self.phase_3_config
        
        # Update parameters from current phase config
        self.joint_dropout_prob = config['joint_dropout_prob']
        self.max_dropped_joints = config['max_dropped_joints'] 
        self.min_dropped_joints = config['min_dropped_joints']
        self.sensor_noise_std = config['sensor_noise_std']
        self.noise_joints_only = config.get('noise_joints_only', True)
        
        # 🎉 PHASE TRANSITION CELEBRATION!
        if old_phase != self.current_phase:
            print(f"\n🚀 PHASE TRANSITION! {old_phase} → {self.current_phase} at {self.current_timestep:,} steps!")
            print(f"📊 NEW PHASE: {self._get_phase_description()}")
            print(f"⚙️  Joint dropout: {self.joint_dropout_prob:.1%}")
            print(f"🔧 Max joints: {self.max_dropped_joints}")  
            print(f"📡 Sensor noise: {self.sensor_noise_std:.3f}")
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
        """Sample which joints to drop for this episode, with symmetric sampling logic."""
        self.dropped_joints = []

        if random.random() < self.joint_dropout_prob:
            if self.max_dropped_joints > 0:
                num_to_drop = random.randint(self.min_dropped_joints, self.max_dropped_joints)

                # Check if symmetric sampling is active for the current phase
                current_phase_config = getattr(self, f'phase_{self.current_phase}_config', {})
                use_symmetric_sampling = current_phase_config.get('use_symmetric_sampling', False)
                prefer_paired_failures = current_phase_config.get('prefer_paired_failures', False)

                if self.symmetric_failure_sampling and use_symmetric_sampling and self.paired_joint_indices:
                    # --- Symmetric Sampling Logic ---
                    joints_to_drop = set()
                    if num_to_drop == 1:
                        # Pick a random pair, then a random joint from that pair
                        pair = random.choice(self.paired_joint_indices)
                        joints_to_drop.add(random.choice(pair))
                    
                    elif num_to_drop == 2 and prefer_paired_failures and random.random() < 0.5:
                        # 50% chance to drop a whole pair
                        pair_to_drop = random.choice(self.paired_joint_indices)
                        joints_to_drop.update(pair_to_drop)

                    else: # Fallback for other cases (e.g., num_to_drop > 2 or non-paired failure)
                        while len(joints_to_drop) < num_to_drop:
                            pair = random.choice(self.paired_joint_indices)
                            joints_to_drop.add(random.choice(pair))
                    
                    self.dropped_joints = list(joints_to_drop)

                else:
                    # --- Original Random Sampling ---
                    available_joints = list(range(self.num_joints))
                    self.dropped_joints = random.sample(available_joints, num_to_drop)
                
                # Optional: uncomment for debugging
                # if self.dropped_joints:
                #     phase_desc = self._get_phase_description()
                #     print(f"🔥 Episode {self.episode_count}: Phase {self.current_phase} - Dropping joints {self.dropped_joints} ({phase_desc})")
    
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