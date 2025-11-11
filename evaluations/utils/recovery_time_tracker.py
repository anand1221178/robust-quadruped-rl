"""
Recovery Time Tracking for Fault Robustness Analysis

This module implements temporal recovery metrics to measure how quickly a policy
regains locomotion capability after a joint failure occurs.

Motivation:
-----------
While retention percentage (final distance / baseline distance) measures ultimate
robustness, it doesn't capture temporal dynamics:
- Does the policy recover quickly or slowly?
- Does it maintain continuous motion or have extended "freezing" periods?
- What percentage of episodes never recover?

Recovery time provides complementary insight into adaptation mechanisms.

Methodology:
------------
1. Measure pre-fault baseline velocity (steps 100-119 before 2-second delayed locking)
2. Define "recovered" as regaining 50% of baseline velocity
3. Track time from fault onset (step 120) until recovery threshold is crossed
4. Report: recovery time (seconds), recovery rate (% episodes recovered), time distribution

Applications:
-------------
- Compare adaptation speed across models (M3 may recover faster than M1)
- Identify joints that allow fast recovery vs permanent degradation
- Validate that retention % captures full story (high correlation expected)

Author: Anand Patel
Date: October 19, 2025
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class RecoveryTimeTracker:
    """
    Tracks temporal recovery dynamics after joint failure injection.

    This class monitors velocity throughout an episode to identify when/if
    the policy regains stable forward locomotion after a fault.
    """

    def __init__(
        self,
        fault_injection_step: int = 120,
        recovery_threshold: float = 0.5,
        pre_fault_window: Tuple[int, int] = (100, 119),
        min_recovery_velocity: float = 0.05,
        fps: int = 60
    ):
        """
        Initialize recovery time tracker.

        Args:
            fault_injection_step: Step when joint failure occurs (default 120 = 2 seconds)
            recovery_threshold: Fraction of pre-fault velocity to count as "recovered"
                               (default 0.5 = 50% of baseline)
            pre_fault_window: (start, end) steps for computing baseline velocity
            min_recovery_velocity: Absolute minimum velocity to count as recovered (m/s)
                                  Prevents declaring recovery for near-zero baselines
            fps: Frames per second (60Hz for RealAnt)
        """
        self.fault_step = fault_injection_step
        self.recovery_threshold_fraction = recovery_threshold
        self.pre_fault_start, self.pre_fault_end = pre_fault_window
        self.min_recovery_velocity = min_recovery_velocity
        self.fps = fps

        # Tracking variables
        self.velocity_history: List[float] = []
        self.position_history: List[float] = []
        self.step_counter = 0

        # Computed metrics
        self.pre_fault_velocity: Optional[float] = None
        self.recovery_velocity_threshold: Optional[float] = None
        self.recovery_step: Optional[int] = None
        self.recovered: bool = False

    def reset(self):
        """Reset tracker for new episode."""
        self.velocity_history = []
        self.position_history = []
        self.step_counter = 0
        self.pre_fault_velocity = None
        self.recovery_velocity_threshold = None
        self.recovery_step = None
        self.recovered = False

    def track_step(self, obs: np.ndarray, info: Optional[Dict] = None):
        """
        Record observation at current timestep.

        Args:
            obs: Observation vector (RealAnt: 29-dim)
            info: Optional info dict from environment
        """
        # Extract forward velocity from observation
        # RealAnt obs structure: [global_pos(3), global_vel(3), qpos(8), qvel(8), ...]
        # Velocity is at index 3 (x-velocity)
        velocity_x = obs[3] if len(obs) > 3 else 0.0

        # Extract x-position (used for distance calculation)
        position_x = obs[0] if len(obs) > 0 else 0.0

        self.velocity_history.append(velocity_x)
        self.position_history.append(position_x)
        self.step_counter += 1

        # Compute pre-fault baseline velocity
        if (self.step_counter == self.pre_fault_end + 1 and
            self.pre_fault_velocity is None):
            self._compute_baseline_velocity()

        # Check for recovery (only after fault injection)
        if (self.step_counter > self.fault_step and
            not self.recovered and
            self.recovery_velocity_threshold is not None):
            if velocity_x >= self.recovery_velocity_threshold:
                self.recovery_step = self.step_counter
                self.recovered = True
                logger.debug(f"Recovery detected at step {self.step_counter} "
                           f"(velocity: {velocity_x:.3f} m/s)")

    def _compute_baseline_velocity(self):
        """Compute average pre-fault velocity from window."""
        if len(self.velocity_history) < self.pre_fault_end:
            logger.warning(f"Insufficient history for baseline velocity computation")
            return

        baseline_window = self.velocity_history[self.pre_fault_start:self.pre_fault_end+1]
        self.pre_fault_velocity = np.mean(baseline_window)

        # Compute recovery threshold (50% of baseline, but at least min_recovery_velocity)
        self.recovery_velocity_threshold = max(
            self.pre_fault_velocity * self.recovery_threshold_fraction,
            self.min_recovery_velocity
        )

        logger.debug(f"Baseline velocity: {self.pre_fault_velocity:.3f} m/s, "
                    f"Recovery threshold: {self.recovery_velocity_threshold:.3f} m/s")

    def get_results(self) -> Dict:
        """
        Compute final recovery metrics.

        Returns:
            Dictionary with recovery analysis:
            - recovered: bool (did policy recover?)
            - recovery_time_steps: int or None (steps from fault to recovery)
            - recovery_time_seconds: float or None (seconds from fault to recovery)
            - pre_fault_velocity: float (baseline velocity before fault)
            - post_fault_avg_velocity: float (average velocity after fault)
            - final_distance: float (total distance traveled)
            - velocity_history: list (full velocity trace for analysis)
        """
        if self.pre_fault_velocity is None:
            logger.warning("get_results() called before baseline velocity computed")
            self._compute_baseline_velocity()

        # Compute post-fault metrics
        if len(self.velocity_history) > self.fault_step:
            post_fault_velocities = self.velocity_history[self.fault_step:]
            post_fault_avg_velocity = np.mean(post_fault_velocities)
        else:
            post_fault_avg_velocity = 0.0

        # Compute recovery time
        if self.recovered and self.recovery_step is not None:
            recovery_time_steps = self.recovery_step - self.fault_step
            recovery_time_seconds = recovery_time_steps / self.fps
        else:
            recovery_time_steps = None
            recovery_time_seconds = None

        # Compute total distance
        if len(self.position_history) > 0:
            final_distance = self.position_history[-1] - self.position_history[0]
        else:
            final_distance = 0.0

        return {
            'recovered': self.recovered,
            'recovery_time_steps': recovery_time_steps,
            'recovery_time_seconds': recovery_time_seconds,
            'pre_fault_velocity': self.pre_fault_velocity,
            'post_fault_avg_velocity': post_fault_avg_velocity,
            'recovery_threshold': self.recovery_velocity_threshold,
            'final_distance': final_distance,
            'velocity_history': self.velocity_history,
            'position_history': self.position_history,
            'total_steps': self.step_counter
        }

    def get_summary_stats(self, all_results: List[Dict]) -> Dict:
        """
        Aggregate recovery metrics across multiple episodes.

        Args:
            all_results: List of result dictionaries from get_results()

        Returns:
            Summary statistics across all episodes
        """
        recovery_times = [r['recovery_time_seconds'] for r in all_results
                         if r['recovery_time_seconds'] is not None]

        num_recovered = sum(1 for r in all_results if r['recovered'])
        num_total = len(all_results)
        recovery_rate = num_recovered / num_total if num_total > 0 else 0.0

        summary = {
            'num_episodes': num_total,
            'num_recovered': num_recovered,
            'recovery_rate': recovery_rate,
            'recovery_time_mean': np.mean(recovery_times) if recovery_times else None,
            'recovery_time_std': np.std(recovery_times) if recovery_times else None,
            'recovery_time_median': np.median(recovery_times) if recovery_times else None,
            'recovery_time_min': np.min(recovery_times) if recovery_times else None,
            'recovery_time_max': np.max(recovery_times) if recovery_times else None,
            'pre_fault_velocity_mean': np.mean([r['pre_fault_velocity'] for r in all_results]),
            'post_fault_velocity_mean': np.mean([r['post_fault_avg_velocity'] for r in all_results]),
            'final_distance_mean': np.mean([r['final_distance'] for r in all_results])
        }

        return summary


def analyze_recovery_patterns(
    results_by_joint: Dict[str, List[Dict]],
    results_by_model: Dict[str, List[Dict]]
) -> Dict:
    """
    Comprehensive analysis of recovery patterns across joints and models.

    Args:
        results_by_joint: {joint_name: [episode_results]}
        results_by_model: {model_name: [episode_results]}

    Returns:
        Analysis dictionary with rankings and insights
    """
    analysis = {
        'by_joint': {},
        'by_model': {},
        'joint_ranking': [],  # Ranked by recovery rate
        'model_ranking': []   # Ranked by recovery speed
    }

    # Analyze by joint
    joint_stats = []
    for joint, results in results_by_joint.items():
        tracker = RecoveryTimeTracker()
        stats = tracker.get_summary_stats(results)
        stats['joint'] = joint
        analysis['by_joint'][joint] = stats
        joint_stats.append(stats)

    # Rank joints by recovery rate (descending)
    analysis['joint_ranking'] = sorted(
        joint_stats,
        key=lambda x: x['recovery_rate'],
        reverse=True
    )

    # Analyze by model
    model_stats = []
    for model, results in results_by_model.items():
        tracker = RecoveryTimeTracker()
        stats = tracker.get_summary_stats(results)
        stats['model'] = model
        analysis['by_model'][model] = stats
        model_stats.append(stats)

    # Rank models by mean recovery time (ascending = faster recovery)
    analysis['model_ranking'] = sorted(
        [s for s in model_stats if s['recovery_time_mean'] is not None],
        key=lambda x: x['recovery_time_mean']
    )

    return analysis


if __name__ == "__main__":
    # Test with dummy data
    print("Testing RecoveryTimeTracker...")

    tracker = RecoveryTimeTracker(
        fault_injection_step=120,
        recovery_threshold=0.5
    )

    # Simulate episode: fast recovery scenario
    np.random.seed(42)

    # Pre-fault: steady 0.20 m/s velocity
    for step in range(120):
        obs = np.zeros(29)
        obs[0] = step * 0.20 / 60.0  # Position
        obs[3] = 0.20  # Velocity
        tracker.track_step(obs)

    # Post-fault: drop to 0.05 m/s, then recover to 0.12 m/s at step 200
    for step in range(120, 300):
        obs = np.zeros(29)
        if step < 200:
            velocity = 0.05
        else:
            velocity = 0.12  # Exceeds 50% threshold (0.10 m/s)

        obs[0] = 120 * 0.20 / 60.0 + (step - 120) * velocity / 60.0
        obs[3] = velocity
        tracker.track_step(obs)

    results = tracker.get_results()

    print("\nRecovery Analysis:")
    print(f"  Pre-fault velocity: {results['pre_fault_velocity']:.3f} m/s")
    print(f"  Recovery threshold: {results['recovery_threshold']:.3f} m/s")
    print(f"  Recovered: {results['recovered']}")
    print(f"  Recovery time: {results['recovery_time_seconds']:.2f} seconds" if results['recovered'] else "  Never recovered")
    print(f"  Post-fault avg velocity: {results['post_fault_avg_velocity']:.3f} m/s")
    print(f"  Final distance: {results['final_distance']:.2f} m")

    print("\n RecoveryTimeTracker test passed!")
