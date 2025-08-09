#!/usr/bin/env python3
"""
Phase 1.3: Define Success Metrics for Robust Quadruped RL
This script defines and tests all evaluation metrics from the proposal
"""

import gymnasium as gym
import numpy as np
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Suppress Ant-v4 deprecation warning
warnings.filterwarnings("ignore", message=".*The environment Ant-v4 is out of date.*")

@dataclass
class SuccessMetrics:
    """All metrics we'll track for the project"""
    # From Section 4.1 of your proposal
    success_rate: float = 0.0          # Fraction maintaining forward locomotion
    cumulative_reward: float = 0.0     # Sum of episode rewards
    recovery_time: float = 0.0         # Time to resume after fault
    failure_rate: float = 0.0          # % episodes with collapse/spin/stuck
    
    # Additional useful metrics
    avg_velocity: float = 0.0          # Average forward velocity
    distance_traveled: float = 0.0     # Total distance in episode
    episode_length: int = 0              # Steps before termination
    time_standing: float = 0.0         # Time robot stayed upright
    
    # Thresholds (from proposal)
    VELOCITY_THRESHOLD: float = 0.5    # Target: 0.5-1.0 m/s
    DISTANCE_THRESHOLD: float = 1.5    # Target: 1.5m in 5 seconds
    TIME_THRESHOLD: float = 5.0        # Success if maintains for 5 seconds
    STUCK_THRESHOLD: float = 2.0       # Failure if stuck for 2 seconds


class MetricsEvaluator:
    """Evaluates robot performance according to success criteria"""
    
    def __init__(self, env_name: str = 'Ant-v4'):
        self.env = gym.make(env_name)
        self.dt = 0.01  # Timestep (10ms for Ant-v4)
        
    def evaluate_episode(self, policy=None, render: bool = False) -> SuccessMetrics:
        """
        Evaluate one episode with given policy (or random if None)
        """
        metrics = SuccessMetrics()
        
        # Reset environment
        obs, info = self.env.reset()
        
        # Track state over time
        positions = []
        velocities = []
        stuck_counter = 0
        initial_x_pos = self._get_x_position()
        last_x_pos = initial_x_pos
        
        # Episode loop
        done = False
        step = 0
        start_time = time.time()
        
        while not done and step < 500:  # Max 500 steps (5 seconds)
            # Get action
            if policy is None:
                action = self.env.action_space.sample()  # Random
            else:
                action = policy(obs)
            
            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Update metrics
            metrics.cumulative_reward += reward
            step += 1
            
            # Track position
            current_x = self._get_x_position()
            positions.append(current_x)
            
            # Calculate instantaneous velocity (for stuck detection)
            instant_velocity = (current_x - last_x_pos) / self.dt
            last_x_pos = current_x
            
            # Check if stuck
            if abs(instant_velocity) < 0.01:  # Nearly stationary
                stuck_counter += 1
            else:
                stuck_counter = 0
            
            # Check failure conditions
            if terminated:  # Robot fell
                metrics.failure_rate = 1.0
                break
                
            if stuck_counter > metrics.STUCK_THRESHOLD / self.dt:
                metrics.failure_rate = 1.0
                break
        
        # Calculate final metrics
        metrics.episode_length = step
        metrics.time_standing = step * self.dt
        
        if positions and step > 0:
            # Calculate AVERAGE velocity over entire episode
            metrics.distance_traveled = positions[-1] - initial_x_pos
            total_time = step * self.dt
            metrics.avg_velocity = metrics.distance_traveled / total_time if total_time > 0 else 0.0
        
        # Check success criteria
        # Success: Travel > 1.5m in 5 seconds with velocity > 0.5 m/s
        if (metrics.distance_traveled > metrics.DISTANCE_THRESHOLD and 
            metrics.avg_velocity > metrics.VELOCITY_THRESHOLD and
            metrics.time_standing >= metrics.TIME_THRESHOLD):
            metrics.success_rate = 1.0
        
        return metrics

    def _get_x_position(self) -> float:
        """Get x-position of the robot"""
        # For Ant-v4, we need to access the underlying MuJoCo data
        # This is a bit hacky but works
        try:
            return self.env.unwrapped.data.qpos[0]
        except:
            return 0.0
    
    def evaluate_batch(self, n_episodes: int = 100, policy=None) -> Dict[str, float]:
        """
        Evaluate multiple episodes and return aggregated metrics
        
        Args:
            n_episodes: Number of episodes to evaluate
            policy: Policy function (None for random)
            
        Returns:
            Dictionary with mean and std of all metrics
        """
        print(f"Evaluating {n_episodes} episodes...")
        
        all_metrics = []
        
        for i in range(n_episodes):
            metrics = self.evaluate_episode(policy)
            all_metrics.append(metrics)
            
            # Progress update
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{n_episodes} episodes")
        
        # Aggregate results
        results = {
            'success_rate': np.mean([m.success_rate for m in all_metrics]),
            'avg_cumulative_reward': np.mean([m.cumulative_reward for m in all_metrics]),
            'std_cumulative_reward': np.std([m.cumulative_reward for m in all_metrics]),
            'avg_distance': np.mean([m.distance_traveled for m in all_metrics]),
            'avg_velocity': np.mean([m.avg_velocity for m in all_metrics]),
            'failure_rate': np.mean([m.failure_rate for m in all_metrics]),
            'avg_episode_length': np.mean([m.episode_length for m in all_metrics]),
        }
        
        return results


def test_random_baseline():
    """Test metrics with random policy as baseline"""
    print("=" * 60)
    print("Testing Success Metrics with Random Policy")
    print("=" * 60)
    
    evaluator = MetricsEvaluator()
    
    # Test single episode
    print("\n1. Single Episode Test:")
    metrics = evaluator.evaluate_episode()
    
    print(f"\nResults:")
    print(f"  Success: {'✅' if metrics.success_rate > 0 else '❌'}")
    print(f"  Distance traveled: {metrics.distance_traveled:.2f} m")
    print(f"  Average velocity: {metrics.avg_velocity:.2f} m/s")
    print(f"  Time standing: {metrics.time_standing:.2f} s")
    print(f"  Cumulative reward: {metrics.cumulative_reward:.2f}")
    print(f"  Failed: {'Yes' if metrics.failure_rate > 0 else 'No'}")
    
    # Test batch evaluation
    print("\n2. Batch Evaluation (10 episodes):")
    results = evaluator.evaluate_batch(n_episodes=10)
    
    print(f"\nAggregate Results:")
    print(f"  Success rate: {results['success_rate']*100:.1f}%")
    print(f"  Average reward: {results['avg_cumulative_reward']:.2f} ± {results['std_cumulative_reward']:.2f}")
    print(f"  Average distance: {results['avg_distance']:.2f} m")
    print(f"  Average velocity: {results['avg_velocity']:.2f} m/s")
    print(f"  Failure rate: {results['failure_rate']*100:.1f}%")
    
    print("\n💡 Insights:")
    print("  - Random policy typically has 0% success rate")
    print("  - This establishes our baseline to beat")
    print("  - Trained PPO should achieve >90% success rate")


def create_evaluation_script():
    """Create a reusable evaluation script for later use"""
    
    script_content = '''# evaluation.py - Reusable evaluation script
"""
Evaluation script for robust quadruped locomotion
Use this to evaluate any trained policy
"""

from define_success_metrics import MetricsEvaluator, SuccessMetrics
import numpy as np

def evaluate_policy(policy_function, n_episodes=100, verbose=True):
    """
    Evaluate a trained policy
    
    Args:
        policy_function: Function that takes obs -> action
        n_episodes: Number of evaluation episodes
        verbose: Print results
        
    Returns:
        Dictionary of results
    """
    evaluator = MetricsEvaluator()
    results = evaluator.evaluate_batch(n_episodes, policy_function)
    
    if verbose:
        print(f"\\nEvaluation over {n_episodes} episodes:")
        print(f"  Success rate: {results['success_rate']*100:.1f}%")
        print(f"  Avg reward: {results['avg_cumulative_reward']:.2f}")
        print(f"  Avg distance: {results['avg_distance']:.2f} m")
        print(f"  Avg velocity: {results['avg_velocity']:.2f} m/s")
    
    return results

# Example usage:
# from stable_baselines3 import PPO
# model = PPO.load("models/ppo_baseline.zip")
# results = evaluate_policy(lambda obs: model.predict(obs)[0])
'''
    
    with open('evaluation.py', 'w') as f:
        f.write(script_content)
    
    print("\n✅ Created evaluation.py for future use")


def main():
    """Run all tests and create evaluation tools"""
    
    # Test metrics with random baseline
    test_random_baseline()
    
    # Create reusable evaluation script
    create_evaluation_script()
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Phase 1.3 Complete: Success Metrics Defined")
    print("=" * 60)
    
    print("\nSuccess Criteria Summary:")
    print("  ✓ Target velocity: 0.5-1.0 m/s")
    print("  ✓ Success: >1.5m in 5 seconds")
    print("  ✓ Episode length: 500 timesteps (5 seconds)")
    print("  ✓ Failure: Fall, spin, or stuck >2 seconds")
    
    print("\nMetrics to Track:")
    print("  1. Success rate (target >90% for baseline)")
    print("  2. Cumulative reward")
    print("  3. Recovery time (for fault scenarios)")
    print("  4. Failure rate")
    
    print("\nNext Steps:")
    print("  → Phase 2: Implement PPO baseline training")
    print("  → Use these metrics to evaluate progress")
    print("  → Log all metrics to W&B during training")


if __name__ == "__main__":
    main()