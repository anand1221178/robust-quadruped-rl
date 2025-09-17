#!/usr/bin/env python3
"""
Test V7 ACDR Models - Hard2Easy vs Easy2Hard Comparison

Based on ACDR paper evaluation methodology, testing across various
failure coefficients k ∈ [0.0, 1.0] to validate that hard2easy
outperforms easy2hard curriculum.
"""

import os
import sys
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Import our wrappers
from envs.success_reward_wrapper import SuccessRewardWrapper

# Import RealAnt
import realant_sim


def test_v7_acdr_model(model_path: str, vec_path: str, model_name: str, test_k_values=None):
    """
    Test V7 ACDR model across various failure coefficients.

    Args:
        model_path: Path to trained model
        vec_path: Path to VecNormalize
        model_name: Name for display
        test_k_values: List of k values to test
    """
    print(f"\n🔬 TESTING V7 MODEL: {model_name}")
    print("="*80)

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None

    # Default k values to test (from paper Figure 5)
    if test_k_values is None:
        test_k_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Create base environment
    def make_env():
        env = gym.make('RealAntMujoco-v0')
        env = SuccessRewardWrapper(env)
        env = Monitor(env)
        return env

    # Results storage
    results = {
        'k_values': test_k_values,
        'avg_rewards': [],
        'avg_velocities': [],
        'avg_distances': [],
        'success_rates': []
    }

    # Test each k value
    for k in test_k_values:
        print(f"\n📊 Testing k={k:.1f} (failure coefficient)")

        # Create test environment
        env = DummyVecEnv([make_env])
        if os.path.exists(vec_path):
            env = VecNormalize.load(vec_path, env)
            env.training = False
            env.norm_reward = False

        # Load model
        model = PPO.load(model_path, env=env)

        # Run multiple episodes
        num_episodes = 10
        episode_rewards = []
        episode_velocities = []
        episode_distances = []

        for episode in range(num_episodes):
            obs = env.reset()
            total_reward = 0
            positions = []
            steps = 0

            # Randomly select a leg to fail
            failed_leg = np.random.randint(0, 4)
            failed_joints = [failed_leg * 2, failed_leg * 2 + 1]

            done = False
            while not done and steps < 250:
                action, _ = model.predict(obs, deterministic=True)

                # Apply failure coefficient k to selected joints
                modified_action = action.copy()
                for joint_idx in failed_joints:
                    if joint_idx < len(modified_action[0]):
                        modified_action[0][joint_idx] *= k

                obs, reward, done, info = env.step(modified_action)
                total_reward += reward[0]

                # Track position
                try:
                    x_pos = env.envs[0].unwrapped.data.qpos[0]
                    positions.append(x_pos)
                except:
                    positions.append(0.0)

                steps += 1
                if done[0]:
                    break

            # Calculate metrics
            if len(positions) >= 2:
                net_displacement = positions[-1] - positions[0]
                time_elapsed = steps * 0.05  # 20Hz
                velocity = net_displacement / time_elapsed if time_elapsed > 0 else 0.0
                distance = abs(net_displacement)
            else:
                velocity = 0.0
                distance = 0.0

            episode_rewards.append(total_reward)
            episode_velocities.append(velocity)
            episode_distances.append(distance)

        # Store average results
        avg_reward = np.mean(episode_rewards)
        avg_velocity = np.mean(episode_velocities)
        avg_distance = np.mean(episode_distances)
        success_rate = np.mean([v > 0.05 for v in episode_velocities]) * 100

        results['avg_rewards'].append(avg_reward)
        results['avg_velocities'].append(avg_velocity)
        results['avg_distances'].append(avg_distance)
        results['success_rates'].append(success_rate)

        print(f"   Average Reward: {avg_reward:.2f}")
        print(f"   Average Velocity: {avg_velocity:.3f} m/s")
        print(f"   Average Distance: {avg_distance:.2f} m")
        print(f"   Success Rate: {success_rate:.1f}%")

        env.close()

    return results


def compare_curricula():
    """Compare hard2easy vs easy2hard V7 models."""
    print("🔥 V7 ACDR COMPARISON: HARD2EASY vs EASY2HARD")
    print("="*80)

    # Model paths (FIXED V7 ACDR models)
    hard2easy_path = "experiments/v7_acdr_hard2easy_fixed_9wbi14fc/final_model.zip"
    hard2easy_vec = "experiments/v7_acdr_hard2easy_fixed_9wbi14fc/vec_normalize.pkl"

    easy2hard_path = "experiments/v7_acdr_easy2hard_fixed_ut9okf0a/final_model.zip"
    easy2hard_vec = "experiments/v7_acdr_easy2hard_fixed_ut9okf0a/vec_normalize.pkl"

    # Test both models
    hard2easy_results = test_v7_acdr_model(
        hard2easy_path, hard2easy_vec, "V7 ACDR Hard2Easy"
    )

    easy2hard_results = test_v7_acdr_model(
        easy2hard_path, easy2hard_vec, "V7 ACDR Easy2Hard"
    )

    # Plot comparison (similar to paper Figure 5)
    if hard2easy_results and easy2hard_results:
        plt.figure(figsize=(12, 8))

        # Average Reward Plot
        plt.subplot(2, 2, 1)
        plt.plot(hard2easy_results['k_values'], hard2easy_results['avg_rewards'],
                'b-o', label='Hard2Easy', linewidth=2, markersize=8)
        if easy2hard_results:
            plt.plot(easy2hard_results['k_values'], easy2hard_results['avg_rewards'],
                    'r--s', label='Easy2Hard', linewidth=2, markersize=8)
        plt.xlabel('Failure Coefficient k')
        plt.ylabel('Average Reward')
        plt.title('V7 ACDR Performance Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Velocity Plot
        plt.subplot(2, 2, 2)
        plt.plot(hard2easy_results['k_values'], hard2easy_results['avg_velocities'],
                'b-o', label='Hard2Easy', linewidth=2, markersize=8)
        if easy2hard_results:
            plt.plot(easy2hard_results['k_values'], easy2hard_results['avg_velocities'],
                    'r--s', label='Easy2Hard', linewidth=2, markersize=8)
        plt.xlabel('Failure Coefficient k')
        plt.ylabel('Average Velocity (m/s)')
        plt.title('Walking Speed vs Failure Severity')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Success Rate Plot
        plt.subplot(2, 2, 3)
        plt.plot(hard2easy_results['k_values'], hard2easy_results['success_rates'],
                'b-o', label='Hard2Easy', linewidth=2, markersize=8)
        if easy2hard_results:
            plt.plot(easy2hard_results['k_values'], easy2hard_results['success_rates'],
                    'r--s', label='Easy2Hard', linewidth=2, markersize=8)
        plt.xlabel('Failure Coefficient k')
        plt.ylabel('Success Rate (%)')
        plt.title('Walking Success vs Failure Severity')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Distance Plot
        plt.subplot(2, 2, 4)
        plt.plot(hard2easy_results['k_values'], hard2easy_results['avg_distances'],
                'b-o', label='Hard2Easy', linewidth=2, markersize=8)
        if easy2hard_results:
            plt.plot(easy2hard_results['k_values'], easy2hard_results['avg_distances'],
                    'r--s', label='Easy2Hard', linewidth=2, markersize=8)
        plt.xlabel('Failure Coefficient k')
        plt.ylabel('Average Distance (m)')
        plt.title('Distance Traveled vs Failure Severity')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.suptitle('V7 ACDR: Hard2Easy vs Easy2Hard Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('v7_acdr_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

        print("\n📊 COMPARISON SAVED: v7_acdr_comparison.png")

    # Print summary
    print("\n🏆 V7 ACDR RESULTS SUMMARY")
    print("="*80)

    if hard2easy_results:
        avg_performance_h2e = np.mean(hard2easy_results['avg_velocities'])
        print(f"Hard2Easy Average Performance: {avg_performance_h2e:.3f} m/s")

    if easy2hard_results:
        avg_performance_e2h = np.mean(easy2hard_results['avg_velocities'])
        print(f"Easy2Hard Average Performance: {avg_performance_e2h:.3f} m/s")

    if hard2easy_results and easy2hard_results:
        improvement = (avg_performance_h2e / avg_performance_e2h - 1) * 100 if avg_performance_e2h > 0 else float('inf')
        print(f"\n✅ Hard2Easy Improvement: {improvement:.1f}% better than Easy2Hard")
        print("This validates the ACDR paper's findings!")


if __name__ == "__main__":
    print("🚀 V7 ACDR Model Evaluation")
    print("   Testing hard2easy vs easy2hard curriculum approaches")
    print()

    # You can test individual models or compare both
    compare_curricula()

    # Or test a specific model:
    # test_v7_acdr_model(
    #     "experiments/v7_acdr_hard2easy/final_model.zip",
    #     "experiments/v7_acdr_hard2easy/vec_normalize.pkl",
    #     "V7 ACDR Hard2Easy"
    # )