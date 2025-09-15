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

    V5 SMART VECNORMALIZE SUPPORT:
    - Optionally resets VecNormalize reward statistics at phase transitions
    - Prevents reward normalization corruption between phases
    - Maintains stable PPO learning throughout curriculum
    """

    def __init__(self, env, curriculum_config: Dict):
        super().__init__(env)
        self.curriculum_config = curriculum_config

        # V5: Smart VecNormalize reward stats reset
        self.reset_reward_stats_on_phase_transition = curriculum_config.get(
            'reset_reward_stats_on_phase_transition', False
        )
        self.vec_normalize_env = None  # Will be set when VecNormalize is detected
        self.last_phase = None  # Track phase changes for reset triggering (None = uninitialized)
        
        # Training progress tracking
        self.total_timesteps = 0
        # V2 FIX: Start at Phase 1 if Phase 0 duration is 0 (handled by PhaseSwitchCallback)
        phase_0_duration = curriculum_config.get('normal_walking_duration', 10000000)
        if phase_0_duration == 0:
            self.current_phase = 1  # Skip Phase 0, start joint failure training
            print("🎯 V2 MODE: Starting at Phase 1 (Phase 0 handled by PhaseSwitchCallback)")
        else:
            self.current_phase = 0  # V1 MODE: Start with Phase 0
        self.current_subphase = -1  # Initialize to -1 to force first subphase transition
        self.subphase_steps = 0

        # V3 INTERLEAVING SUPPORT
        self.interleave_mode = curriculum_config.get('interleave_mode', None)
        self.normal_episode_ratio = curriculum_config.get('normal_episode_ratio', 1.0)
        self.failure_episode_ratio = curriculum_config.get('failure_episode_ratio', 0.0)
        self.episode_count = 0
        self.current_episode_type = 'normal'  # 'normal' or 'failure'

        # V3 mode detection
        self.is_v3_interleaved = self.interleave_mode == 'episode' and self.failure_episode_ratio > 0
        if self.is_v3_interleaved:
            print(f"🔄 V3 INTERLEAVED MODE ENABLED!")
            print(f"   Normal episodes: {self.normal_episode_ratio:.1%}")
            print(f"   Failure episodes: {self.failure_episode_ratio:.1%}")
            print(f"   Interleave method: {self.interleave_mode}")
            print(f"   🎯 Benefits: Prevents skill forgetting + builds robustness")
        
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

        # V5: Initialize VecNormalize detection
        if self.reset_reward_stats_on_phase_transition:
            self._detect_vecnormalize()
            if self.vec_normalize_env:
                print(f"🔥 V5 SMART VECNORMALIZE: Reward stats reset enabled at phase transitions!")
            else:
                print(f"⚠️  V5 WARNING: VecNormalize not detected - reward stats reset disabled")

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

    def _detect_vecnormalize(self):
        """V5: Detect VecNormalize wrapper in the environment stack"""
        # VecNormalize is applied AFTER SystematicCurriculumWrapper in the training script
        # So we can't detect it during __init__. It will be set manually via set_vecnormalize_env()
        self.vec_normalize_env = None
        print(f"🔍 V5: VecNormalize detection deferred to manual connection")

    def _reset_reward_stats(self):
        """V5: Reset VecNormalize reward statistics to prevent corruption"""
        if not self.vec_normalize_env:
            return

        try:
            # Reset reward running mean and std
            if hasattr(self.vec_normalize_env, 'ret_rms'):
                # Check if we're dealing with RunningMeanStd from stable-baselines3
                rms = self.vec_normalize_env.ret_rms

                # Reset the running mean and variance for rewards
                # Handle both torch tensors and numpy arrays
                if hasattr(rms.mean, 'fill_'):
                    # PyTorch tensor
                    rms.mean.fill_(0.0)
                    rms.var.fill_(1.0)
                else:
                    # Numpy array/scalar
                    if hasattr(rms, 'mean'):
                        rms.mean = 0.0
                    if hasattr(rms, 'var'):
                        rms.var = 1.0

                # Reset count to small value to avoid division by zero
                rms.count = 1e-4

                print(f"🔄 V5 REWARD STATS RESET: VecNormalize reward statistics reset at phase transition")
                mean_val = rms.mean.item() if hasattr(rms.mean, 'item') else rms.mean
                var_val = rms.var.item() if hasattr(rms.var, 'item') else rms.var
                print(f"   New reward mean: {mean_val:.3f}")
                print(f"   New reward std: {var_val**0.5:.3f}")

            else:
                print(f"⚠️  V5 WARNING: VecNormalize ret_rms not found - cannot reset reward stats")

        except Exception as e:
            print(f"❌ V5 ERROR: Failed to reset VecNormalize reward stats: {e}")
            # Disable further reset attempts
            self.reset_reward_stats_on_phase_transition = False

    def set_vecnormalize_env(self, vec_env):
        """V5: Manually set VecNormalize environment reference for reward stats reset"""
        self.vec_normalize_env = vec_env
        if vec_env and self.reset_reward_stats_on_phase_transition:
            print(f"✅ V5: VecNormalize environment manually set - reward stats reset ready!")
    
    def _update_curriculum_phase(self):
        """Update current curriculum phase based on training progress"""
        # V2 FIX: Handle case where Phase 0 is skipped (duration = 0)
        if self.phase_0_duration == 0:
            # V2 Mode: Phase 0 handled externally, start from Phase 1
            phase_1_end = sum(p['duration'] for p in self.phase_1_schedule)
            phase_2_end = phase_1_end + sum(p['duration'] for p in self.phase_2_schedule)
            phase_3_end = phase_2_end + sum(p['duration'] for p in self.phase_3_schedule)

            if self.total_timesteps < phase_1_end:
                phase_start = 0
                schedule = self.phase_1_schedule
                self.current_phase = 1
            elif self.total_timesteps < phase_2_end:
                phase_start = phase_1_end
                schedule = self.phase_2_schedule
                self.current_phase = 2
            elif self.total_timesteps < phase_3_end:
                phase_start = phase_2_end
                schedule = self.phase_3_schedule
                self.current_phase = 3
            else:
                self.current_phase = 4  # Complete
                self.failed_joints = []
                self.failed_joint_names = []

                # V5: Handle phase tracking for V2 completion early return
                phase_changed = (self.last_phase is not None and self.last_phase != self.current_phase)
                if self.reset_reward_stats_on_phase_transition:
                    print(f"🔍 V5 DEBUG (V2 Complete): last_phase={self.last_phase}, current_phase={self.current_phase}, phase_changed={phase_changed}")
                if self.reset_reward_stats_on_phase_transition and phase_changed:
                    print(f"🔄 V5: Phase transition detected! {self.last_phase} → {self.current_phase}")
                    self._reset_reward_stats()
                self.last_phase = self.current_phase
                return
        else:
            # V1 Mode: Original logic with Phase 0
            phase_0_end = self.phase_0_duration
            phase_1_end = phase_0_end + sum(p['duration'] for p in self.phase_1_schedule)
            phase_2_end = phase_1_end + sum(p['duration'] for p in self.phase_2_schedule)
            phase_3_end = phase_2_end + sum(p['duration'] for p in self.phase_3_schedule)

            if self.total_timesteps < phase_0_end:
                # Phase 0: Normal walking foundation
                self.current_phase = 0
                self.failed_joints = []  # No joint failures in Phase 0
                self.failed_joint_names = []

                # V5: Handle phase tracking for Phase 0 early return
                phase_changed = (self.last_phase is not None and self.last_phase != self.current_phase)
                if self.reset_reward_stats_on_phase_transition and phase_changed:
                    print(f"🔄 V5: Phase transition detected! {self.last_phase} → {self.current_phase}")
                    self._reset_reward_stats()
                self.last_phase = self.current_phase
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

                # V5: Handle phase tracking for completion early return
                phase_changed = (self.last_phase is not None and self.last_phase != self.current_phase)
                if self.reset_reward_stats_on_phase_transition:
                    print(f"🔍 V5 DEBUG (Complete): last_phase={self.last_phase}, current_phase={self.current_phase}, phase_changed={phase_changed}")
                if self.reset_reward_stats_on_phase_transition and phase_changed:
                    print(f"🔄 V5: Phase transition detected! {self.last_phase} → {self.current_phase}")
                    self._reset_reward_stats()
                self.last_phase = self.current_phase
                return
        
        # Determine current subphase within the phase
        phase_progress = self.total_timesteps - phase_start
        cumulative_duration = 0

        # DEBUG: Add detailed logging for Clean V2 debugging
        if len(schedule) == 0:
            print(f"🚨 BUG: Schedule is empty for phase {self.current_phase}!")
            return

        # DEBUG: Print progress info every 10k steps for first subphase
        if phase_progress % 10000 < 10 and phase_progress > 0:
            print(f"🔍 DEBUG: Phase {self.current_phase} progress: {phase_progress:,} steps")
            print(f"   Schedule length: {len(schedule)} subphases")
            print(f"   Current subphase: {self.current_subphase}")
            print(f"   First subphase duration: {schedule[0]['duration']:,} steps")

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

        # V5: Check for phase transitions and reset reward stats if enabled
        # Need to check BEFORE updating last_phase
        phase_changed = (self.last_phase is not None and self.last_phase != self.current_phase)

        # Debug logging can be enabled for troubleshooting
        # if self.reset_reward_stats_on_phase_transition:
        #     print(f"🔍 V5 DEBUG: last_phase={self.last_phase}, current_phase={self.current_phase}, phase_changed={phase_changed}")

        if self.reset_reward_stats_on_phase_transition and phase_changed:
            print(f"🔄 V5: Phase transition detected! {self.last_phase} → {self.current_phase}")
            self._reset_reward_stats()

        # Update last phase tracker (after checking for transitions)
        self.last_phase = self.current_phase
    
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

        # Note: Robot position metrics now handled by RobotPositionCallback
        # (removed duplicate tracking to avoid confusion)

        # V3: Use episode-specific failed joints for info tracking
        active_failed_joints = getattr(self, 'episode_failed_joints', self.failed_joints)
        active_failed_joint_names = getattr(self, 'episode_failed_joint_names', self.failed_joint_names)

        info.update({
            'systematic_curriculum': True,
            'curriculum_phase': self.current_phase,
            'curriculum_subphase': self.current_subphase + 1,
            'failed_joints': active_failed_joints.copy(),
            'failed_joint_names': active_failed_joint_names.copy(),
            'pattern_type': self._get_current_pattern_type(),
            'subphase_progress': self.subphase_steps,
            'total_timesteps': self.total_timesteps,
            'original_action': action,
            'modified_action': modified_action,
            'episode_type': getattr(self, 'current_episode_type', 'systematic'),  # V3: episode type
            'episode_count': getattr(self, 'episode_count', 0)  # V3: episode count
        })
        
        return obs, reward, terminated, truncated, info
    
    def _determine_episode_type(self):
        """V3: Determine whether this episode should have failures or be normal"""
        if not self.is_v3_interleaved or self.current_phase == 0:
            return 'systematic'

        # Use episode count to determine type (weighted random based on ratios)
        import random
        random.seed(self.episode_count)  # Reproducible based on episode number

        total_ratio = self.normal_episode_ratio + self.failure_episode_ratio
        normal_prob = self.normal_episode_ratio / total_ratio

        if random.random() < normal_prob:
            return 'normal'
        else:
            return 'failure'

    def _apply_joint_failures(self, action):
        """Apply joint failures by setting failed joint actions to 0"""
        # V3: Use episode-specific failed joints instead of phase failed joints
        active_failed_joints = getattr(self, 'episode_failed_joints', self.failed_joints)

        if len(active_failed_joints) == 0:
            return action
        
        modified_action = action.copy()

        # Lock failed joints (set torque to 0)
        for joint_idx in active_failed_joints:
            if joint_idx < len(modified_action):
                modified_action[joint_idx] = 0.0

        return modified_action
    
    def _get_current_pattern_type(self):
        """Get the pattern type for the current subphase"""
        if self.current_phase == 0:
            return 'normal'  # Phase 0: Normal walking foundation
        elif self.current_phase == 1:
            return self.phase_1_schedule[self.current_subphase]['pattern_type']
        elif self.current_phase == 2:
            return self.phase_2_schedule[self.current_subphase]['pattern_type']
        elif self.current_phase == 3:
            return self.phase_3_schedule[self.current_subphase]['pattern_type']
        else:
            return 'complete'
    
    def reset(self, **kwargs):
        """Reset environment and determine episode type for V3 interleaving"""
        obs, info = self.env.reset(**kwargs)

        # V3 INTERLEAVING: Determine episode type for this episode
        if self.is_v3_interleaved and self.current_phase > 0:
            self.episode_count += 1
            self.current_episode_type = self._determine_episode_type()

            # V3: Override joint failures based on episode type
            if self.current_episode_type == 'normal':
                # Normal episode: no joint failures (preserve skills)
                self.episode_failed_joints = []
                self.episode_failed_joint_names = []
            else:
                # Failure episode: use systematic curriculum failures
                self.episode_failed_joints = self.failed_joints.copy()
                self.episode_failed_joint_names = self.failed_joint_names.copy()
        else:
            # V4 mode or Phase 0: use systematic curriculum as-is
            self.episode_failed_joints = self.failed_joints.copy()
            self.episode_failed_joint_names = self.failed_joint_names.copy()
            self.current_episode_type = 'systematic'

        # Add curriculum info to reset info
        if info is None:
            info = {}

        info.update({
            'systematic_curriculum': True,
            'curriculum_phase': self.current_phase,
            'curriculum_subphase': self.current_subphase + 1,
            'failed_joints': self.episode_failed_joints.copy(),
            'failed_joint_names': self.episode_failed_joint_names.copy(),
            'pattern_type': self._get_current_pattern_type(),
            'subphase_progress': self.subphase_steps,
            'total_timesteps': self.total_timesteps,
            'episode_type': self.current_episode_type,  # V3: track episode type
            'episode_count': self.episode_count  # V3: track episode number
        })

        return obs, info
    
    def get_curriculum_status(self):
        """Get detailed curriculum status for logging"""
        if self.current_phase == 0:
            # Phase 0: Normal walking foundation
            return {
                'phase': 0,
                'subphase': 0,
                'total_subphases': 0,
                'failed_joints': [],
                'failed_joint_names': [],
                'pattern_type': 'normal',
                'description': 'Phase 0: Normal walking foundation',
                'subphase_progress': self.total_timesteps,
                'subphase_duration': self.phase_0_duration,
                'total_progress': self.total_timesteps,
                'completion_percentage': (self.total_timesteps / self._calculate_total_steps()) * 100
            }
        elif self.current_phase <= 3:
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