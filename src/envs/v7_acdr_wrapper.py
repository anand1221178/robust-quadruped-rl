"""
V7 ACDR Wrapper - Adaptive Curriculum Dynamics Randomization

Based on "Reinforcement Learning with Adaptive Curriculum Dynamics Randomization
for Fault-Tolerant Robot Control" (2111.10005v1)

Key Innovation: Hard2Easy curriculum - start with complete failures (k=0),
gradually improve to mild failures (k=1.5). This is the OPPOSITE of traditional
curriculum learning and proven to work for quadruped locomotion!
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class V7ACDRWrapper(gym.Wrapper):
    """
    Adaptive Curriculum Dynamics Randomization (ACDR) Wrapper

    Implements the hard2easy curriculum from the ACDR paper that successfully
    achieved fault-tolerant quadruped locomotion. Unlike our failed V1-V5
    approaches that used easy2hard, this starts with DEAD joints (k=0)
    and gradually improves.

    Key differences from failed approaches:
    - Starts with k=0 (complete failures) instead of k=1.0 (perfect)
    - Gradually increases k (easier) instead of decreasing (harder)
    - Never returns to k=0 at end of training (preserves locomotion)
    """

    def __init__(self,
                 env: gym.Env,
                 curriculum_type: str = 'hard2easy',
                 initial_L: float = 0.0,
                 initial_U: float = 0.0,
                 target_L: float = 1.0,
                 target_U: float = 1.5,
                 update_step: float = 0.01,
                 performance_window: int = 100,
                 performance_threshold: float = None,
                 # V7.2 Dual-Phase Parameters
                 phase_1_target: float = None,
                 phase_2_minimum: float = None,
                 # V7.3 Multi-Objective Parameters
                 speed_weight: float = 0.7,
                 robustness_weight: float = 0.3,
                 adaptive_weights: bool = False,
                 rewind_threshold: float = None,
                 rewind_steps: int = 2,
                 consolidation_period: int = 1000000,
                 adaptive_curriculum: bool = False,
                 verbose: bool = True):
        """
        Initialize V7 ACDR Wrapper with hard2easy curriculum.

        Args:
            env: Base environment
            curriculum_type: 'hard2easy' (proven) or 'easy2hard' (fails)
            initial_L: Initial lower bound (0.0 for hard2easy)
            initial_U: Initial upper bound (0.0 for hard2easy)
            target_L: Target lower bound (1.0 for hard2easy)
            target_U: Target upper bound (1.5 for hard2easy)
            update_step: How much to adjust interval (0.01 from paper)
            performance_window: Episodes to average for adaptation
            performance_threshold: Initial threshold (adaptive)
            verbose: Whether to print curriculum updates
        """
        super().__init__(env)

        self.curriculum_type = curriculum_type
        self.verbose = verbose

        # Curriculum interval [L, U] for failure coefficient k
        self.L = initial_L  # Lower bound
        self.U = initial_U  # Upper bound
        self.target_L = target_L
        self.target_U = target_U
        self.update_step = update_step

        # Performance tracking for adaptive curriculum
        self.performance_buffer = []
        self.performance_window = performance_window
        self.performance_threshold = performance_threshold

        # Episode tracking
        self.episode_count = 0
        self.episode_return = 0.0
        self.current_failed_leg = None
        self.current_k = None
        self.curriculum_updates = 0

        # V7.2 Dual-Phase tracking
        self.phase_1_target = phase_1_target
        self.phase_2_minimum = phase_2_minimum
        self.current_phase = 1 if phase_1_target is not None else None

        # V7.3 Multi-Objective parameters
        self.speed_weight = speed_weight
        self.robustness_weight = robustness_weight
        self.adaptive_weights = adaptive_weights
        self.rewind_threshold = rewind_threshold
        self.rewind_steps = rewind_steps
        self.consolidation_period = consolidation_period
        self.adaptive_curriculum = adaptive_curriculum

        # Curriculum rewind tracking
        self.curriculum_history = [(self.L, self.U)]
        self.consolidation_steps = 0
        self.in_consolidation = False

        # Joint configuration (8 joints: 4 legs × 2 joints/leg)
        self.num_legs = 4
        self.joints_per_leg = 2
        self.joint_mapping = {
            0: [0, 1],  # Leg 0: hip and ankle
            1: [2, 3],  # Leg 1: hip and ankle
            2: [4, 5],  # Leg 2: hip and ankle
            3: [6, 7],  # Leg 3: hip and ankle
        }

        if self.verbose:
            print(f"🚀 V7 ACDR WRAPPER INITIALIZED - {curriculum_type.upper()} CURRICULUM")
            print(f"   Initial Interval: [{self.L:.2f}, {self.U:.2f}]")
            print(f"   Target Interval: [{target_L:.2f}, {target_U:.2f}]")
            if curriculum_type == 'hard2easy':
                print("   ✅ Using PROVEN hard2easy approach from ACDR paper")
                print("   Starting with DEAD joints (k=0), will gradually improve")
            else:
                print("   ⚠️ WARNING: easy2hard shown to fail in research!")

    def reset(self, **kwargs):
        """Reset environment and sample new failure for this episode."""
        obs, info = self.env.reset(**kwargs)

        # Track episode
        self.episode_count += 1
        self.episode_return = 0.0

        # Sample failure for this episode (one random leg fails)
        self.current_failed_leg = np.random.randint(0, self.num_legs)

        # Sample failure coefficient k from current interval
        self.current_k = np.random.uniform(self.L, self.U)

        # Add failure info
        info['failed_leg'] = self.current_failed_leg
        info['failed_joints'] = self.joint_mapping[self.current_failed_leg]
        info['failure_coefficient'] = self.current_k
        info['curriculum_interval'] = [self.L, self.U]
        info['curriculum_type'] = self.curriculum_type
        info['curriculum_updates'] = self.curriculum_updates

        if self.verbose and self.episode_count % 1000 == 0:
            print(f"   Episode {self.episode_count}: Leg {self.current_failed_leg} "
                  f"fails with k={self.current_k:.3f}, Interval=[{self.L:.2f}, {self.U:.2f}]")

        return obs, info

    def step(self, action):
        """Step environment with failed actuator."""
        # Apply actuator failure to selected leg
        modified_action = action.copy()

        # Get joints for failed leg
        failed_joints = self.joint_mapping[self.current_failed_leg]

        # Apply failure coefficient k to failed joints
        # k=0: joint completely dead, k=1: normal operation
        for joint_idx in failed_joints:
            if joint_idx < len(modified_action):
                modified_action[joint_idx] = self.current_k * action[joint_idx]

        # Step environment with modified action
        obs, reward, terminated, truncated, info = self.env.step(modified_action)

        # Track episode return for curriculum adaptation
        self.episode_return += reward

        # Add failure info
        info['failed_leg'] = self.current_failed_leg
        info['failed_joints'] = failed_joints
        info['failure_coefficient'] = self.current_k
        info['action_modified'] = not np.array_equal(action, modified_action)

        # Update curriculum when episode ends
        if terminated or truncated:
            self._update_curriculum()

        return obs, reward, terminated, truncated, info

    def _update_curriculum(self):
        """Update curriculum based on performance (Algorithm 1 from paper)."""
        # Add episode return to buffer
        self.performance_buffer.append(self.episode_return)

        # Keep buffer size limited
        if len(self.performance_buffer) > self.performance_window:
            self.performance_buffer.pop(0)

        # Check if we have enough data to update
        if len(self.performance_buffer) >= self.performance_window:
            # Calculate average performance
            avg_performance = np.mean(self.performance_buffer)

            # V7.3: Check for rewind condition
            if self.adaptive_curriculum and self.rewind_threshold is not None:
                if avg_performance < self.rewind_threshold and not self.in_consolidation:
                    self._rewind_curriculum()
                    return

            # Handle consolidation period after rewind
            if self.in_consolidation:
                self.consolidation_steps += 1
                if self.consolidation_steps >= self.consolidation_period:
                    self.in_consolidation = False
                    self.consolidation_steps = 0
                    if self.verbose:
                        print("✅ CONSOLIDATION COMPLETE - Resuming curriculum progression")
                return

            # Initialize threshold if not set
            if self.performance_threshold is None:
                self.performance_threshold = avg_performance
                if self.verbose:
                    print(f"   Initial performance threshold: {self.performance_threshold:.2f}")

            # Update curriculum if performance exceeds threshold
            if avg_performance >= self.performance_threshold:
                if self.curriculum_type == 'hard2easy':
                    # Hard2Easy: Increase k (make it easier) as performance improves
                    if self.U < self.target_U:
                        old_L, old_U = self.L, self.U
                        self.L = min(self.L + self.update_step, self.target_L)
                        self.U = min(self.U + self.update_step, self.target_U)
                        self.curriculum_updates += 1

                        # Store curriculum history for potential rewind (V7.3)
                        self.curriculum_history.append((self.L, self.U))

                        # Update threshold (adaptive)
                        self.performance_threshold = avg_performance

                        if self.verbose:
                            phase_info = f" [Phase {self.current_phase}]" if self.current_phase else ""
                            print(f"📈 CURRICULUM UPDATE {self.curriculum_updates}{phase_info}:")
                            print(f"   Performance: {avg_performance:.2f} > {self.performance_threshold:.2f}")
                            print(f"   New Interval: [{self.L:.2f}, {self.U:.2f}] (was [{old_L:.2f}, {old_U:.2f}])")
                            print(f"   Progress: {(self.U / self.target_U) * 100:.1f}% to target")

                elif self.curriculum_type == 'easy2hard':
                    # Easy2Hard: Decrease k (make it harder) - shown to fail!
                    if self.L > 0.0:
                        self.L = max(self.L - self.update_step, 0.0)
                        self.U = max(self.U - self.update_step, 0.0)
                        self.curriculum_updates += 1

                        # Update threshold
                        self.performance_threshold = avg_performance

                        if self.verbose:
                            print(f"📉 CURRICULUM UPDATE {self.curriculum_updates} (easy2hard - likely to fail!):")
                            print(f"   New Interval: [{self.L:.2f}, {self.U:.2f}]")

                # Clear buffer after update
                self.performance_buffer = []

    def _rewind_curriculum(self):
        """V7.3: Rewind curriculum when performance drops too low."""
        if len(self.curriculum_history) >= self.rewind_steps + 1:
            # Rewind to earlier curriculum state
            rewind_idx = -(self.rewind_steps + 1)
            self.L, self.U = self.curriculum_history[rewind_idx]

            # Enter consolidation period
            self.in_consolidation = True
            self.consolidation_steps = 0

            if self.verbose:
                print(f"🔄 CURRICULUM REWIND:")
                print(f"   Performance too low - rewinding {self.rewind_steps} steps")
                print(f"   New Interval: [{self.L:.2f}, {self.U:.2f}]")
                print(f"   Entering consolidation for {self.consolidation_period:,} steps")

    def get_curriculum_info(self) -> Dict[str, Any]:
        """Get current curriculum status."""
        return {
            'curriculum_type': self.curriculum_type,
            'current_interval': [self.L, self.U],
            'target_interval': [self.target_L, self.target_U],
            'episode_count': self.episode_count,
            'curriculum_updates': self.curriculum_updates,
            'performance_threshold': self.performance_threshold,
            'recent_performance': np.mean(self.performance_buffer) if self.performance_buffer else 0.0,
            'progress_percentage': (self.U / self.target_U) * 100 if self.target_U > 0 else 0.0
        }

    def render(self, mode='human'):
        """Render with curriculum info overlay if available."""
        frame = self.env.render(mode='rgb_array') if hasattr(self.env, 'render') else None

        if frame is not None and self.verbose:
            # Could add text overlay showing k value and failed leg
            # For now, just return the frame
            return frame

        return self.env.render(mode=mode) if hasattr(self.env, 'render') else None


class V7LinearCurriculumDR(gym.Wrapper):
    """
    Linear Curriculum DR for comparison (as described in ACDR paper Appendix A).

    This is the non-adaptive version that updates at fixed timesteps rather
    than based on performance. Used as a baseline comparison to show that
    adaptive curriculum (V7ACDR) is superior.
    """

    def __init__(self,
                 env: gym.Env,
                 curriculum_type: str = 'hard2easy',
                 total_timesteps: int = 25_000_000,
                 num_updates: int = 11,
                 verbose: bool = False):
        """
        Initialize Linear Curriculum DR (LCDR) for comparison.

        Args:
            env: Base environment
            curriculum_type: 'hard2easy' or 'easy2hard'
            total_timesteps: Total training timesteps
            num_updates: Number of curriculum updates (N=11 in paper)
            verbose: Whether to print updates
        """
        super().__init__(env)

        self.curriculum_type = curriculum_type
        self.verbose = verbose

        # Fixed update schedule
        self.update_interval = total_timesteps // num_updates
        self.timesteps = 0
        self.last_update_timestep = 0

        # Curriculum parameters
        if curriculum_type == 'hard2easy':
            self.L = 0.0
            self.U = 0.0
            self.delta = 1.5 / (num_updates - 1)  # Reach 1.5 at end
        else:  # easy2hard
            self.L = 1.5
            self.U = 1.5
            self.delta = -1.5 / (num_updates - 1)  # Reach 0.0 at end

        # Current failure state
        self.current_failed_leg = None
        self.current_k = None

    def reset(self, **kwargs):
        """Reset with new failure sampling."""
        obs, info = self.env.reset(**kwargs)

        # Sample failure
        self.current_failed_leg = np.random.randint(0, 4)
        self.current_k = np.random.uniform(self.L, self.U)

        info['failed_leg'] = self.current_failed_leg
        info['failure_coefficient'] = self.current_k
        info['curriculum_interval'] = [self.L, self.U]

        return obs, info

    def step(self, action):
        """Step with failure and fixed curriculum updates."""
        # Apply failure
        modified_action = action.copy()
        failed_joints = [self.current_failed_leg * 2, self.current_failed_leg * 2 + 1]

        for joint_idx in failed_joints:
            if joint_idx < len(modified_action):
                modified_action[joint_idx] = self.current_k * action[joint_idx]

        obs, reward, terminated, truncated, info = self.env.step(modified_action)

        # Update timesteps and check for curriculum update
        self.timesteps += 1
        if self.timesteps - self.last_update_timestep >= self.update_interval:
            self.L += self.delta
            self.U += self.delta
            self.last_update_timestep = self.timesteps

            if self.verbose:
                print(f"LCDR Update at {self.timesteps}: [{self.L:.2f}, {self.U:.2f}]")

        return obs, reward, terminated, truncated, info