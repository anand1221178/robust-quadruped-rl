#!/usr/bin/env python3
"""
V8 Enhanced ACDR Wrapper - Adaptation-Focused Curriculum
Building on V7 ACDR success with enhanced adaptation learning capabilities

Key Enhancements over V7:
1. Extended episodes (1000+ steps) for sophisticated adaptation learning
2. Adaptation-focused reward design with velocity improvement bonuses
3. Progressive failure complexity (single → dual → sequential failures)
4. Ankle-specific training phases for improved compensation
5. Dynamic failure injection for real-time adaptation learning
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import logging

class V8EnhancedACDRWrapper(gym.Wrapper):
    """
    V8 Enhanced ACDR: Advanced adaptation-focused curriculum for joint failure robustness

    Builds on V7 ACDR success (77% joint failure success rate) with enhancements for:
    - Better adaptation learning through longer episodes
    - Sophisticated gait modification discovery
    - Ankle-specific compensation training
    - Sequential failure handling for dynamic adaptation
    """

    def __init__(self, env, config=None):
        super().__init__(env)

        # Default V8 Enhanced configuration
        self.config = config or {
            'total_timesteps': 60_000_000,  # Extended training for sophisticated learning
            'curriculum_phases': {
                'phase_1': {
                    'name': 'Single Joint Mastery',
                    'duration_steps': 20_000_000,
                    'joint_patterns': ['single_joint'],
                    'k_range': [0.8, 0.4],  # Mild to moderate failures
                    'episode_length': 1000,  # Extended for adaptation
                    'failure_probability': 0.8  # High probability for consistent training
                },
                'phase_2': {
                    'name': 'Dual Joint Compensation',
                    'duration_steps': 20_000_000,
                    'joint_patterns': ['dual_adjacent', 'dual_diagonal'],
                    'k_range': [0.6, 0.2],  # Moderate to severe failures
                    'episode_length': 1200,  # Even longer for complex adaptation
                    'failure_probability': 0.7
                },
                'phase_3': {
                    'name': 'Sequential Dynamic Adaptation',
                    'duration_steps': 20_000_000,
                    'joint_patterns': ['sequential_failure'],
                    'k_range': [0.8, 0.0],  # Complete failures
                    'episode_length': 1500,  # Maximum time for dynamic learning
                    'failure_probability': 0.6
                }
            },
            'ankle_specialization': {
                'enabled': True,
                'phase_1_ankle_focus': 0.4,  # 40% of phase 1 focused on ankles
                'ankle_reward_multiplier': 2.0,  # Extra rewards for ankle adaptation
                'ankle_adaptation_bonus': 50.0   # Bonus for ankle velocity improvement
            },
            'adaptation_rewards': {
                'base_forward': 100.0,           # Standard forward motion reward
                'adaptation_bonus': 50.0,        # Bonus for velocity improvement over episode
                'stability_bonus': 25.0,         # Bonus for maintaining balance with failures
                'efficiency_bonus': 25.0,        # Bonus for energy-efficient movement
                'gait_discovery_bonus': 75.0     # Large bonus for discovering new locomotion patterns
            },
            'failure_injection': {
                'dynamic_failures': True,        # Inject failures mid-episode
                'adaptation_grace_period': 200,  # Steps to adapt after failure injection
                'failure_announcement': True     # Signal to robot that failure occurred
            }
        }

        # Training state
        self.total_steps = 0
        self.current_phase = None
        self.episode_steps = 0
        self.current_failures = []
        self.baseline_velocity = None
        self.episode_velocities = []

        # Joint mappings for RealAnt (8 joints: 4 hips + 4 ankles)
        self.joint_names = ['hip_1', 'ankle_1', 'hip_2', 'ankle_2',
                           'hip_3', 'ankle_3', 'hip_4', 'ankle_4']
        self.hip_joints = [0, 2, 4, 6]      # Hip joint indices
        self.ankle_joints = [1, 3, 5, 7]    # Ankle joint indices

        # Adjacent and diagonal joint pairs for complex failures
        self.adjacent_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]  # Hip-ankle on same leg
        self.diagonal_pairs = [(0, 6), (2, 4), (1, 7), (3, 5)]  # Cross-body patterns

        # Sequential failure tracking
        self.sequential_failure_schedule = []
        self.next_failure_step = None

        logging.info(f"V8 Enhanced ACDR Wrapper initialized with {self.config['total_timesteps']} total steps")
        logging.info(f"Phase progression: {len(self.config['curriculum_phases'])} phases with extended episodes")

    def _get_current_phase(self):
        """Determine current curriculum phase based on training progress"""
        phases = self.config['curriculum_phases']
        cumulative_steps = 0

        for phase_name, phase_config in phases.items():
            cumulative_steps += phase_config['duration_steps']
            if self.total_steps < cumulative_steps:
                return phase_name, phase_config

        # Training complete, use final phase
        final_phase = list(phases.keys())[-1]
        return final_phase, phases[final_phase]

    def _select_failure_pattern(self, phase_config):
        """Select joint failure pattern based on current phase"""
        patterns = phase_config['joint_patterns']
        pattern = np.random.choice(patterns)

        ankle_specialization = self.config['ankle_specialization']
        is_ankle_focus = (ankle_specialization['enabled'] and
                         self.current_phase[0] == 'phase_1' and
                         np.random.random() < ankle_specialization['phase_1_ankle_focus'])

        if pattern == 'single_joint':
            if is_ankle_focus:
                # Focus on ankle joints for specialization
                joint_idx = np.random.choice(self.ankle_joints)
            else:
                # Random single joint
                joint_idx = np.random.choice(len(self.joint_names))
            return [joint_idx]

        elif pattern == 'dual_adjacent':
            # Adjacent joint pairs (same leg)
            pair = np.random.choice(len(self.adjacent_pairs))
            return list(self.adjacent_pairs[pair])

        elif pattern == 'dual_diagonal':
            # Diagonal joint pairs (cross-body)
            pair = np.random.choice(len(self.diagonal_pairs))
            return list(self.diagonal_pairs[pair])

        elif pattern == 'sequential_failure':
            # Start with single failure, add more during episode
            initial_joint = np.random.choice(len(self.joint_names))
            return [initial_joint]

        return []

    def _calculate_k_value(self, phase_config):
        """Calculate current k-value based on ACDR hard→easy progression"""
        k_min, k_max = phase_config['k_range']
        phase_start_step = sum(p['duration_steps'] for name, p in self.config['curriculum_phases'].items()
                              if name < self.current_phase[0])
        phase_progress = (self.total_steps - phase_start_step) / phase_config['duration_steps']
        phase_progress = min(1.0, max(0.0, phase_progress))

        # Hard→Easy: Start with low k (harder), progress to higher k (easier)
        k = k_min + (k_max - k_min) * phase_progress
        return k

    def _setup_sequential_failures(self, phase_config):
        """Setup sequential failure schedule for dynamic adaptation"""
        if 'sequential_failure' in phase_config['joint_patterns']:
            episode_length = phase_config['episode_length']

            # Schedule additional failures throughout episode
            failure_points = [
                int(episode_length * 0.3),  # 30% through episode
                int(episode_length * 0.6),  # 60% through episode
                int(episode_length * 0.8)   # 80% through episode
            ]

            self.sequential_failure_schedule = []
            for step in failure_points:
                if np.random.random() < 0.5:  # 50% chance of additional failure
                    # Add a different joint failure
                    available_joints = [i for i in range(len(self.joint_names))
                                      if i not in self.current_failures]
                    if available_joints:
                        new_failure = np.random.choice(available_joints)
                        self.sequential_failure_schedule.append((step, new_failure))

            if self.sequential_failure_schedule:
                self.next_failure_step = self.sequential_failure_schedule[0][0]
        else:
            self.sequential_failure_schedule = []
            self.next_failure_step = None

    def _calculate_adaptation_rewards(self, base_reward):
        """Calculate additional rewards for adaptation learning"""
        if len(self.episode_velocities) < 10:  # Need some history
            return base_reward

        adaptation_config = self.config['adaptation_rewards']
        additional_reward = 0.0

        # Adaptation bonus: Reward velocity improvement over episode
        current_velocity = np.mean(self.episode_velocities[-5:])  # Recent average
        early_velocity = np.mean(self.episode_velocities[:5])     # Early average

        if current_velocity > early_velocity:
            improvement = (current_velocity - early_velocity) / max(early_velocity, 0.01)
            additional_reward += adaptation_config['adaptation_bonus'] * improvement

        # Ankle specialization bonus
        ankle_config = self.config['ankle_specialization']
        if (ankle_config['enabled'] and
            any(joint in self.ankle_joints for joint in self.current_failures)):

            if current_velocity > early_velocity:
                additional_reward += ankle_config['ankle_adaptation_bonus']

        # Stability bonus: Reward consistent forward motion despite failures
        if len(self.episode_velocities) >= 20:
            velocity_std = np.std(self.episode_velocities[-20:])
            if velocity_std < 0.02:  # Low variance = stable locomotion
                additional_reward += adaptation_config['stability_bonus']

        return base_reward + additional_reward

    def _apply_joint_failures(self, action):
        """Apply current joint failures to action"""
        if not self.current_failures:
            return action

        # Get current k-value
        _, phase_config = self._get_current_phase()
        k = self._calculate_k_value(phase_config)

        # Apply failures
        modified_action = action.copy()
        for joint_idx in self.current_failures:
            if joint_idx < len(modified_action):
                modified_action[joint_idx] = k * action[joint_idx]

        return modified_action

    def _check_sequential_failures(self):
        """Check if new failures should be injected this step"""
        if (self.next_failure_step is not None and
            self.episode_steps >= self.next_failure_step):

            # Inject new failure
            _, new_joint = self.sequential_failure_schedule.pop(0)
            self.current_failures.append(new_joint)

            # Schedule next failure
            if self.sequential_failure_schedule:
                self.next_failure_step = self.sequential_failure_schedule[0][0]
            else:
                self.next_failure_step = None

            logging.info(f"Sequential failure injected: Joint {new_joint} at step {self.episode_steps}")

    def reset(self, **kwargs):
        """Reset environment with V8 Enhanced ACDR curriculum"""
        obs, info = self.env.reset(**kwargs)

        # Update phase
        self.current_phase = self._get_current_phase()
        phase_name, phase_config = self.current_phase

        # Reset episode tracking
        self.episode_steps = 0
        self.episode_velocities = []

        # Determine if this episode will have failures
        if np.random.random() < phase_config['failure_probability']:
            # Select failure pattern
            self.current_failures = self._select_failure_pattern(phase_config)

            # Setup sequential failures if needed
            self._setup_sequential_failures(phase_config)

            logging.info(f"Episode starting with failures: {self.current_failures} in {phase_name}")
        else:
            # No failures this episode (allow pure locomotion practice)
            self.current_failures = []
            self.sequential_failure_schedule = []
            self.next_failure_step = None

        return obs, info

    def step(self, action):
        """Step with V8 Enhanced ACDR joint failure and adaptation tracking"""
        # Check for sequential failure injection
        self._check_sequential_failures()

        # Apply current joint failures
        modified_action = self._apply_joint_failures(action)

        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(modified_action)

        # Track velocity for adaptation rewards
        if hasattr(self.env.unwrapped, 'data'):
            # Calculate current velocity
            if not hasattr(self, '_last_x_pos'):
                self._last_x_pos = self.env.unwrapped.data.qpos[0]

            current_x = self.env.unwrapped.data.qpos[0]
            velocity = (current_x - self._last_x_pos) / 0.05  # 50 Hz timestep
            self.episode_velocities.append(velocity)
            self._last_x_pos = current_x

        # Calculate adaptation rewards
        enhanced_reward = self._calculate_adaptation_rewards(reward)

        # Update counters
        self.episode_steps += 1
        self.total_steps += 1

        # Add failure info to info dict
        info['v8_acdr'] = {
            'current_phase': self.current_phase[0],
            'current_failures': self.current_failures,
            'total_training_steps': self.total_steps,
            'episode_steps': self.episode_steps,
            'adaptation_reward_bonus': enhanced_reward - reward
        }

        return obs, enhanced_reward, terminated, truncated, info

    def get_curriculum_status(self):
        """Get detailed curriculum status for monitoring"""
        phase_name, phase_config = self._get_current_phase()

        phase_start_step = sum(p['duration_steps'] for name, p in self.config['curriculum_phases'].items()
                              if name < phase_name)
        phase_progress = (self.total_steps - phase_start_step) / phase_config['duration_steps']

        return {
            'current_phase': phase_name,
            'phase_description': phase_config['name'],
            'phase_progress': min(1.0, max(0.0, phase_progress)),
            'total_progress': self.total_steps / self.config['total_timesteps'],
            'current_k_value': self._calculate_k_value(phase_config),
            'current_failures': self.current_failures,
            'episode_length': phase_config['episode_length']
        }