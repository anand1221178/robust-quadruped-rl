"""
Specialist Training Wrapper for V6 Ensemble Approach

This wrapper enables training specialist controllers for specific joint failure patterns.
Unlike previous approaches that tried to teach one policy everything, this creates
focused experts that excel at specific failure scenarios.

Key Innovation: Separation > Integration for robust locomotion
"""

import gymnasium as gym
import numpy as np
from typing import List, Optional, Dict, Any


class SpecialistTrainingWrapper(gym.Wrapper):
    """
    Trains specialist controllers for specific joint failure patterns.

    Each specialist focuses on ONE specific failure scenario:
    - hip_specialist: Only hip joint failures
    - ankle_specialist: Only ankle joint failures
    - multi_joint_specialist: Complex multi-joint failures
    - normal_specialist: Perfect locomotion (no failures)
    """

    SPECIALIST_CONFIGS = {
        'normal': {
            'description': 'Perfect locomotion specialist - no failures',
            'failure_joints': [],
            'failure_probability': 0.0
        },
        'hip_specialist': {
            'description': 'Hip joint failure specialist',
            'failure_joints': [0, 1, 2, 3],  # Hip joints only
            'failure_probability': 0.5,  # 50% of episodes have hip failure
            'max_failures': 2  # Up to 2 hip joints can fail
        },
        'ankle_specialist': {
            'description': 'Ankle joint failure specialist',
            'failure_joints': [4, 5, 6, 7],  # Ankle joints only
            'failure_probability': 0.5,  # 50% of episodes have ankle failure
            'max_failures': 2  # Up to 2 ankle joints can fail
        },
        'multi_joint_specialist': {
            'description': 'Complex multi-joint failure specialist',
            'failure_joints': [0, 1, 2, 3, 4, 5, 6, 7],  # All joints
            'failure_probability': 0.7,  # 70% of episodes have failures
            'max_failures': 3  # Up to 3 joints can fail
        },
        'single_joint_specialist': {
            'description': 'Single joint failure specialist',
            'failure_joints': [0, 1, 2, 3, 4, 5, 6, 7],  # All joints
            'failure_probability': 0.5,  # 50% of episodes have single failure
            'max_failures': 1  # Exactly 1 joint fails
        }
    }

    def __init__(self,
                 env: gym.Env,
                 specialist_type: str = 'normal',
                 training_phase: str = 'specialist',  # 'baseline' or 'specialist'
                 verbose: bool = True):
        """
        Initialize specialist training wrapper.

        Args:
            env: Base environment
            specialist_type: Type of specialist to train
            training_phase: 'baseline' for normal walking, 'specialist' for failure training
            verbose: Whether to print training information
        """
        super().__init__(env)

        if specialist_type not in self.SPECIALIST_CONFIGS:
            raise ValueError(f"Unknown specialist type: {specialist_type}. "
                           f"Choose from: {list(self.SPECIALIST_CONFIGS.keys())}")

        self.specialist_type = specialist_type
        self.config = self.SPECIALIST_CONFIGS[specialist_type]
        self.training_phase = training_phase
        self.verbose = verbose

        # Episode tracking
        self.current_failed_joints = []
        self.episode_count = 0
        self.failure_episodes = 0

        if self.verbose:
            print(f"🎯 SPECIALIST TRAINING WRAPPER INITIALIZED")
            print(f"   Specialist Type: {specialist_type}")
            print(f"   Description: {self.config['description']}")
            print(f"   Training Phase: {training_phase}")
            if specialist_type != 'normal':
                print(f"   Failure Probability: {self.config['failure_probability']*100:.0f}%")
                print(f"   Max Failures: {self.config.get('max_failures', 0)}")

    def reset(self, **kwargs):
        """Reset environment and determine if this episode will have failures."""
        obs, info = self.env.reset(**kwargs)
        self.episode_count += 1

        # Determine if this episode will have failures
        self.current_failed_joints = []

        if self.training_phase == 'specialist' and self.specialist_type != 'normal':
            # Specialist training - apply failures based on configuration
            if np.random.random() < self.config['failure_probability']:
                # This episode will have failures
                available_joints = self.config['failure_joints']
                max_failures = min(self.config.get('max_failures', 1), len(available_joints))
                num_failures = np.random.randint(1, max_failures + 1)

                # Randomly select joints to fail
                self.current_failed_joints = np.random.choice(
                    available_joints,
                    size=num_failures,
                    replace=False
                ).tolist()

                self.failure_episodes += 1

                if self.verbose and self.episode_count % 100 == 0:
                    print(f"   Episode {self.episode_count}: Failed joints {self.current_failed_joints}")

        # Add failure info to the info dict
        info['failed_joints'] = self.current_failed_joints.copy()
        info['specialist_type'] = self.specialist_type
        info['training_phase'] = self.training_phase

        return obs, info

    def step(self, action):
        """Step with potential joint failures based on specialist configuration."""
        # Apply joint failures if any
        modified_action = action.copy()

        if len(self.current_failed_joints) > 0:
            # Zero out actions for failed joints
            for joint_idx in self.current_failed_joints:
                if joint_idx < len(modified_action):
                    modified_action[joint_idx] = 0.0

        # Step the environment with modified action
        obs, reward, terminated, truncated, info = self.env.step(modified_action)

        # Add specialist info
        info['failed_joints'] = self.current_failed_joints.copy()
        info['specialist_type'] = self.specialist_type
        info['action_modified'] = not np.array_equal(action, modified_action)

        # Log statistics periodically
        if self.verbose and terminated and self.episode_count % 1000 == 0:
            failure_rate = (self.failure_episodes / self.episode_count) * 100
            print(f"📊 Training Statistics (Episode {self.episode_count}):")
            print(f"   Failure Rate: {failure_rate:.1f}%")
            print(f"   Specialist: {self.specialist_type}")

        return obs, reward, terminated, truncated, info

    def get_specialist_info(self) -> Dict[str, Any]:
        """Get information about current specialist training."""
        return {
            'specialist_type': self.specialist_type,
            'training_phase': self.training_phase,
            'config': self.config,
            'episode_count': self.episode_count,
            'failure_episodes': self.failure_episodes,
            'failure_rate': (self.failure_episodes / max(1, self.episode_count)) * 100
        }


class EnsembleController:
    """
    Runtime controller that selects the appropriate specialist based on detected failures.

    This is the inference-time component that uses the trained specialists.
    """

    def __init__(self, specialists: Dict[str, Any], verbose: bool = True):
        """
        Initialize ensemble controller with trained specialists.

        Args:
            specialists: Dictionary mapping specialist types to trained models
            verbose: Whether to print selection information
        """
        self.specialists = specialists
        self.verbose = verbose
        self.selection_history = []

        if 'normal' not in specialists:
            raise ValueError("Must have a 'normal' specialist for baseline locomotion")

        print(f"🎮 ENSEMBLE CONTROLLER INITIALIZED")
        print(f"   Available Specialists: {list(specialists.keys())}")

    def select_specialist(self, detected_failures: List[int]) -> tuple:
        """
        Select the appropriate specialist based on detected failures.

        Args:
            detected_failures: List of detected failed joint indices

        Returns:
            (specialist_name, specialist_model)
        """
        if len(detected_failures) == 0:
            # No failures - use normal specialist
            selected = 'normal'

        elif len(detected_failures) == 1:
            # Single joint failure
            joint_idx = detected_failures[0]
            if joint_idx in [0, 1, 2, 3]:
                selected = 'hip_specialist' if 'hip_specialist' in self.specialists else 'single_joint_specialist'
            elif joint_idx in [4, 5, 6, 7]:
                selected = 'ankle_specialist' if 'ankle_specialist' in self.specialists else 'single_joint_specialist'
            else:
                selected = 'single_joint_specialist'

        elif len(detected_failures) >= 2:
            # Multiple joint failures
            hip_failures = [j for j in detected_failures if j in [0, 1, 2, 3]]
            ankle_failures = [j for j in detected_failures if j in [4, 5, 6, 7]]

            if len(hip_failures) > 0 and len(ankle_failures) == 0:
                # Only hip failures
                selected = 'hip_specialist' if 'hip_specialist' in self.specialists else 'multi_joint_specialist'
            elif len(ankle_failures) > 0 and len(hip_failures) == 0:
                # Only ankle failures
                selected = 'ankle_specialist' if 'ankle_specialist' in self.specialists else 'multi_joint_specialist'
            else:
                # Mixed or complex failures
                selected = 'multi_joint_specialist'

        else:
            selected = 'normal'

        # Fallback to normal if specialist not available
        if selected not in self.specialists:
            if self.verbose:
                print(f"⚠️ Specialist '{selected}' not available, using 'normal'")
            selected = 'normal'

        self.selection_history.append((detected_failures, selected))

        if self.verbose and len(self.selection_history) % 100 == 0:
            print(f"   Selected {selected} for failures {detected_failures}")

        return selected, self.specialists[selected]

    def predict(self, obs, detected_failures: Optional[List[int]] = None):
        """
        Predict action using appropriate specialist.

        Args:
            obs: Observation from environment
            detected_failures: Detected failed joints (if None, assumes no failures)

        Returns:
            action: Action to take
        """
        if detected_failures is None:
            detected_failures = []

        specialist_name, specialist_model = self.select_specialist(detected_failures)
        action, _ = specialist_model.predict(obs, deterministic=True)

        return action

    def get_selection_stats(self) -> Dict[str, int]:
        """Get statistics about specialist selection."""
        stats = {}
        for _, specialist in self.selection_history:
            stats[specialist] = stats.get(specialist, 0) + 1
        return stats