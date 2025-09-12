#!/usr/bin/env python3
"""
🎯 Systematic Joint Failure Curriculum Wrapper
Implements guaranteed, systematic joint failures for robustness training
"""

import gymnasium as gym
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

class SystematicCurriculumWrapper(gym.Wrapper):
    """
    Systematic curriculum wrapper for guaranteed joint failure training.
    
    Instead of probabilistic failures, this wrapper implements a systematic
    curriculum that guarantees specific joint failures during training phases.
    
    Phases:
    1. Single Joint Mastery: Each joint fails individually for N steps
    2. Dual Joint Combinations: Specific joint pairs fail for N steps  
    3. Triple Joint Challenge: Critical 3-joint combinations
    """
    
    def __init__(self, env, curriculum_config: Dict):
        super().__init__(env)
        self.curriculum_config = curriculum_config
        
        # Training progress tracking
        self.total_timesteps = 0
        self.current_phase = 0  # Start with Phase 0: Normal walking
        self.current_subphase = 0
        self.subphase_steps = 0
        
        # Joint failure state
        self.failed_joints = []  # Current failed joints (indices)
        self.failed_joint_names = []  # Current failed joints (names)
        
        # Action space is 8 joint torques for RealAnt
        self.num_joints = 8
        self.joint_names = ["hip_1", "ankle_1", "hip_2", "ankle_2", 
                           "hip_3", "ankle_3", "hip_4", "ankle_4"]
        
        # Curriculum configuration
        self._setup_curriculum()
        
        # Update current failure mode
        self._update_curriculum_phase()
        
        print(f"🎯 Systematic Curriculum Initialized")
        print(f"   Phase 0: Normal walking foundation")
        print(f"   Phase 1: {len(self.phase_1_schedule)} single joints")
        print(f"   Phase 2: {len(self.phase_2_schedule)} dual combinations") 
        print(f"   Phase 3: {len(self.phase_3_schedule)} triple combinations")
        print(f"   Total training steps: {self._calculate_total_steps():,}")
    
    def _setup_curriculum(self):
        """Setup the systematic curriculum schedule"""
        config = self.curriculum_config
        
        # Phase 0: Normal walking foundation (NEW!)
        self.phase_0_duration = config.get('normal_walking_duration', 10000000)  # 10M steps
        
        # Phase 1: Single joint mastery
        single_joint_duration = config.get('single_joint_duration', 3000000)
        self.phase_1_schedule = []
        
        for joint_name in self.joint_names:
            joint_idx = self.joint_names.index(joint_name)
            self.phase_1_schedule.append({
                'failed_joints': [joint_idx],
                'failed_joint_names': [joint_name],
                'duration': single_joint_duration,
                'pattern_type': 'single',
                'description': f'Single joint failure: {joint_name}'
            })
        
        # Phase 2: Strategic dual combinations
        dual_combo_duration = config.get('dual_combo_duration', 3000000)
        self.phase_2_schedule = []
        
        # Get combinations from config
        anatomical = config.get('anatomical_combinations', [
            ["hip_1", "ankle_1"], ["hip_2", "ankle_2"], 
            ["hip_3", "ankle_3"], ["hip_4", "ankle_4"]
        ])
        diagonal = config.get('diagonal_combinations', [
            ["hip_1", "hip_4"], ["hip_2", "hip_3"]
        ])
        functional = config.get('functional_combinations', [
            ["hip_1", "hip_2"], ["hip_3", "hip_4"],
            ["ankle_1", "ankle_2"], ["ankle_3", "ankle_4"]
        ])
        
        # Add anatomical combinations
        for combo in anatomical:
            joint_indices = [self.joint_names.index(name) for name in combo]
            self.phase_2_schedule.append({
                'failed_joints': joint_indices,
                'failed_joint_names': combo,
                'duration': dual_combo_duration,
                'pattern_type': 'anatomical',
                'description': f'Anatomical failure: {" + ".join(combo)}'
            })
        
        # Add diagonal combinations  
        for combo in diagonal:
            joint_indices = [self.joint_names.index(name) for name in combo]
            self.phase_2_schedule.append({
                'failed_joints': joint_indices,
                'failed_joint_names': combo,
                'duration': dual_combo_duration,
                'pattern_type': 'diagonal',
                'description': f'Diagonal failure: {" + ".join(combo)}'
            })
            
        # Add functional combinations
        for combo in functional:
            joint_indices = [self.joint_names.index(name) for name in combo]
            self.phase_2_schedule.append({
                'failed_joints': joint_indices,
                'failed_joint_names': combo,
                'duration': dual_combo_duration,
                'pattern_type': 'functional',
                'description': f'Functional failure: {" + ".join(combo)}'
            })
        
        # Phase 3: Triple joint challenge (optional)
        triple_combo_duration = config.get('triple_combo_duration', 3000000)
        self.phase_3_schedule = []
        
        critical_triples = config.get('critical_triple_combinations', [
            ["hip_1", "ankle_1", "hip_3"],  # Front limb + rear support
            ["hip_1", "hip_4", "ankle_2"],  # Diagonal + stability
            ["hip_1", "hip_2", "hip_3"]     # Three hip cascade
        ])
        
        for combo in critical_triples:
            joint_indices = [self.joint_names.index(name) for name in combo]
            self.phase_3_schedule.append({
                'failed_joints': joint_indices,
                'failed_joint_names': combo,
                'duration': triple_combo_duration,
                'pattern_type': 'triple',
                'description': f'Triple failure: {" + ".join(combo)}'
            })
    
    def _calculate_total_steps(self):
        """Calculate total training steps across all phases"""
        total = self.phase_0_duration  # Add Phase 0: Normal walking
        for phase in self.phase_1_schedule:
            total += phase['duration']
        for phase in self.phase_2_schedule:
            total += phase['duration']
        for phase in self.phase_3_schedule:
            total += phase['duration']
        return total
    
    def _update_curriculum_phase(self):
        """Update current curriculum phase based on training progress"""
        # Calculate phase boundaries
        phase_0_end = self.phase_0_duration
        phase_1_end = phase_0_end + sum(p['duration'] for p in self.phase_1_schedule)
        phase_2_end = phase_1_end + sum(p['duration'] for p in self.phase_2_schedule)
        phase_3_end = phase_2_end + sum(p['duration'] for p in self.phase_3_schedule)
        
        # Determine current phase
        if self.total_timesteps < phase_0_end:
            # Phase 0: Normal walking foundation
            self.current_phase = 0
            self.failed_joints = []  # No joint failures in Phase 0
            self.failed_joint_names = []
            return
        elif self.total_timesteps < phase_1_end:
            self.current_phase = 1
            schedule = self.phase_1_schedule
            phase_start = phase_0_end
        elif self.total_timesteps < phase_2_end:
            self.current_phase = 2
            schedule = self.phase_2_schedule
            phase_start = phase_1_end
        elif self.total_timesteps < phase_3_end:
            self.current_phase = 3
            schedule = self.phase_3_schedule
            phase_start = phase_2_end
        else:
            # Training complete
            self.current_phase = 4
            self.failed_joints = []
            self.failed_joint_names = []
            return
        
        # Determine current subphase within the phase
        phase_progress = self.total_timesteps - phase_start
        cumulative_duration = 0
        
        for i, subphase in enumerate(schedule):
            if phase_progress < cumulative_duration + subphase['duration']:
                # We're in this subphase
                if self.current_subphase != i:
                    # Transitioning to new subphase
                    self.current_subphase = i
                    self.subphase_steps = 0
                    self.failed_joints = subphase['failed_joints'].copy()
                    self.failed_joint_names = subphase['failed_joint_names'].copy()
                    
                    print(f"\\n🎯 CURRICULUM TRANSITION")
                    print(f"   Phase {self.current_phase}, Subphase {i+1}/{len(schedule)}")
                    print(f"   {subphase['description']}")
                    print(f"   Failed joints: {self.failed_joint_names}")
                    print(f"   Pattern type: {subphase['pattern_type']}")
                    print(f"   Duration: {subphase['duration']:,} steps")
                
                break
            cumulative_duration += subphase['duration']
    
    def step(self, action):
        """Apply systematic joint failures and track progress"""
        # Apply joint failures (lock specific joints to 0)
        modified_action = self._apply_joint_failures(action)
        
        # Take environment step
        obs, reward, terminated, truncated, info = self.env.step(modified_action)
        
        # Update training progress
        self.total_timesteps += 1
        self.subphase_steps += 1
        
        # Check for curriculum transitions
        self._update_curriculum_phase()
        
        # Add curriculum info to info dict
        if info is None:
            info = {}
        
        info.update({
            'systematic_curriculum': True,
            'curriculum_phase': self.current_phase,
            'curriculum_subphase': self.current_subphase + 1,
            'failed_joints': self.failed_joints.copy(),
            'failed_joint_names': self.failed_joint_names.copy(),
            'pattern_type': self._get_current_pattern_type(),
            'subphase_progress': self.subphase_steps,
            'total_timesteps': self.total_timesteps,
            'original_action': action,
            'modified_action': modified_action
        })
        
        return obs, reward, terminated, truncated, info
    
    def _apply_joint_failures(self, action):
        """Apply joint failures by setting failed joint actions to 0"""
        if len(self.failed_joints) == 0:
            return action
        
        modified_action = action.copy()
        
        # Lock failed joints (set torque to 0)
        for joint_idx in self.failed_joints:
            if joint_idx < len(modified_action):
                modified_action[joint_idx] = 0.0
        
        return modified_action
    
    def _get_current_pattern_type(self):
        """Get the pattern type for the current subphase"""
        if self.current_phase == 1:
            return self.phase_1_schedule[self.current_subphase]['pattern_type']
        elif self.current_phase == 2:
            return self.phase_2_schedule[self.current_subphase]['pattern_type']
        elif self.current_phase == 3:
            return self.phase_3_schedule[self.current_subphase]['pattern_type']
        else:
            return 'complete'
    
    def reset(self, **kwargs):
        """Reset environment and maintain current curriculum state"""
        obs, info = self.env.reset(**kwargs)
        
        # Add curriculum info to reset info
        if info is None:
            info = {}
            
        info.update({
            'systematic_curriculum': True,
            'curriculum_phase': self.current_phase,
            'curriculum_subphase': self.current_subphase + 1,
            'failed_joints': self.failed_joints.copy(),
            'failed_joint_names': self.failed_joint_names.copy(),
            'pattern_type': self._get_current_pattern_type(),
            'subphase_progress': self.subphase_steps,
            'total_timesteps': self.total_timesteps
        })
        
        return obs, info
    
    def get_curriculum_status(self):
        """Get detailed curriculum status for logging"""
        if self.current_phase <= 3:
            if self.current_phase == 1:
                schedule = self.phase_1_schedule
            elif self.current_phase == 2:
                schedule = self.phase_2_schedule
            else:
                schedule = self.phase_3_schedule
                
            current_subphase_info = schedule[self.current_subphase]
            
            return {
                'phase': self.current_phase,
                'subphase': self.current_subphase + 1,
                'total_subphases': len(schedule),
                'failed_joints': self.failed_joint_names,
                'pattern_type': current_subphase_info['pattern_type'],
                'description': current_subphase_info['description'],
                'subphase_progress': self.subphase_steps,
                'subphase_duration': current_subphase_info['duration'],
                'total_progress': self.total_timesteps,
                'completion_percentage': (self.total_timesteps / self._calculate_total_steps()) * 100
            }
        else:
            return {
                'phase': 'complete',
                'total_progress': self.total_timesteps,
                'completion_percentage': 100.0
            }