#!/usr/bin/env python3
"""
Test V1 Fixed Systematic Curriculum Model
Tests the completed ppo_systematic_curriculum_fixed_64M model performance
"""

import os
import sys
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import json
from datetime import datetime

# Add src to path
sys.path.append('/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/src')
sys.path.append('/Users/anandpatel/Documents/4th Year/robust-quadruped-rl')

# Import RealAnt environments
import realant_sim
from envs.success_reward_wrapper import SuccessRewardWrapper
from envs.domain_randomization_wrapper import DomainRandomizationWrapper

def test_systematic_curriculum_model():
    """Test the V1 fixed systematic curriculum model"""

    print("🧪 TESTING V1 FIXED SYSTEMATIC CURRICULUM MODEL")
    print("="*60)

    # Model paths
    model_path = "experiments/ppo_systematic_curriculum_fixed_64M_ugz1q24t/final_model.zip"
    vec_normalize_path = "experiments/ppo_systematic_curriculum_fixed_64M_ugz1q24t/vec_normalize.pkl"

    print(f"📁 Model: {model_path}")
    print(f"📁 VecNormalize: {vec_normalize_path}")

    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    if not os.path.exists(vec_normalize_path):
        print(f"❌ VecNormalize file not found: {vec_normalize_path}")
        return

    print("✅ Model files found!")

    # Create baseline environment for testing
    def make_env():
        env = gym.make('RealAntMujoco-v0')
        env = SuccessRewardWrapper(env)
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize.load(vec_normalize_path, env)
    env.training = False  # Disable training mode for evaluation

    print("✅ Environment created!")

    # Load model
    model = PPO.load(model_path)
    print("✅ Model loaded!")

    # Test 1: Baseline Performance (No Failures)
    print("\n🎯 TEST 1: BASELINE PERFORMANCE (NO FAILURES)")
    baseline_results = test_performance(model, env, "Baseline", num_episodes=10)

    # Test 2: Single Joint Failures
    print("\n🎯 TEST 2: SINGLE JOINT FAILURE ROBUSTNESS")
    joint_results = {}

    joints_to_test = ["hip_1", "ankle_1", "hip_2", "ankle_2", "hip_3", "ankle_3", "hip_4", "ankle_4"]

    for joint in joints_to_test[:4]:  # Test first 4 joints
        print(f"\n   Testing {joint} failure...")

        # Create DR environment with specific joint failure
        def make_failure_env():
            env = gym.make('RealAntMujoco-v0')
            env = SuccessRewardWrapper(env)

            # Configure specific joint failure
            dr_config = {
                'joint_failure_prob': 1.0,  # 100% failure rate
                'max_failed_joints': 1,
                'min_failed_joints': 1,
                'specific_joints': [joint],  # Only fail this specific joint
                'persistent_failures': True
            }
            env = DomainRandomizationWrapper(env, dr_config)
            env = Monitor(env)
            return env

        failure_env = DummyVecEnv([make_failure_env])
        failure_env = VecNormalize.load(vec_normalize_path, failure_env)
        failure_env.training = False

        joint_results[joint] = test_performance(model, failure_env, f"{joint} Failure", num_episodes=5)
        failure_env.close()

    # Test 3: Dual Joint Failures (sample)
    print("\n🎯 TEST 3: DUAL JOINT FAILURE ROBUSTNESS")
    dual_combinations = [
        ["hip_1", "ankle_1"],  # Anatomical
        ["hip_1", "hip_4"],    # Diagonal
        ["hip_1", "hip_2"]     # Functional
    ]

    dual_results = {}
    for combo in dual_combinations:
        combo_name = "+".join(combo)
        print(f"\n   Testing {combo_name} failure...")

        def make_dual_failure_env():
            env = gym.make('RealAntMujoco-v0')
            env = SuccessRewardWrapper(env)

            dr_config = {
                'joint_failure_prob': 1.0,
                'max_failed_joints': 2,
                'min_failed_joints': 2,
                'specific_joints': combo,
                'persistent_failures': True
            }
            env = DomainRandomizationWrapper(env, dr_config)
            env = Monitor(env)
            return env

        dual_env = DummyVecEnv([make_dual_failure_env])
        dual_env = VecNormalize.load(vec_normalize_path, dual_env)
        dual_env.training = False

        dual_results[combo_name] = test_performance(model, dual_env, f"{combo_name} Failure", num_episodes=5)
        dual_env.close()

    # Compile results
    print("\n" + "="*60)
    print("📊 SYSTEMATIC CURRICULUM V1 FIXED RESULTS SUMMARY")
    print("="*60)

    all_results = {
        'baseline': baseline_results,
        'single_joint_failures': joint_results,
        'dual_joint_failures': dual_results,
        'test_date': datetime.now().isoformat(),
        'model_path': model_path
    }

    # Performance summary
    baseline_velocity = baseline_results['avg_velocity']
    print(f"🏃 BASELINE PERFORMANCE: {baseline_velocity:.3f} m/s")

    print(f"\n🦵 SINGLE JOINT ROBUSTNESS:")
    single_retentions = []
    for joint, result in joint_results.items():
        velocity = result['avg_velocity']
        retention = (velocity / baseline_velocity) * 100 if baseline_velocity > 0 else 0
        single_retentions.append(retention)
        print(f"   {joint:8s}: {velocity:.3f} m/s ({retention:5.1f}% retention)")

    print(f"\n👥 DUAL JOINT ROBUSTNESS:")
    dual_retentions = []
    for combo, result in dual_results.items():
        velocity = result['avg_velocity']
        retention = (velocity / baseline_velocity) * 100 if baseline_velocity > 0 else 0
        dual_retentions.append(retention)
        print(f"   {combo:12s}: {velocity:.3f} m/s ({retention:5.1f}% retention)")

    # Overall robustness scores
    avg_single_retention = np.mean(single_retentions) if single_retentions else 0
    avg_dual_retention = np.mean(dual_retentions) if dual_retentions else 0
    overall_robustness = (avg_single_retention + avg_dual_retention) / 2

    print(f"\n🎯 ROBUSTNESS SUMMARY:")
    print(f"   Average Single Joint Retention: {avg_single_retention:.1f}%")
    print(f"   Average Dual Joint Retention: {avg_dual_retention:.1f}%")
    print(f"   Overall Robustness Score: {overall_robustness:.1f}%")

    # Comparison to baseline
    baseline_comparison = {
        'baseline_model': 0.224,  # Known baseline velocity
        'systematic_v1_fixed': baseline_velocity,
        'performance_ratio': baseline_velocity / 0.224 if baseline_velocity > 0 else 0
    }

    print(f"\n📈 BASELINE COMPARISON:")
    print(f"   Original Baseline: 0.224 m/s")
    print(f"   Systematic V1: {baseline_velocity:.3f} m/s")
    print(f"   Performance Ratio: {baseline_comparison['performance_ratio']:.2f}x")

    # Save results
    results_file = f"V1_Fixed_Systematic_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")

    # Final verdict
    print(f"\n🎊 FINAL VERDICT:")
    if baseline_velocity > 0.15:
        if overall_robustness > 60:
            print("   🏆 EXCELLENT: Strong locomotion + good robustness!")
        elif overall_robustness > 40:
            print("   ✅ GOOD: Decent locomotion + moderate robustness!")
        else:
            print("   ⚠️  PARTIAL: Good locomotion but limited robustness")
    elif baseline_velocity > 0.05:
        print("   ⚠️  LIMITED: Some locomotion but needs improvement")
    else:
        print("   ❌ FAILED: No meaningful locomotion achieved")

    env.close()
    return all_results

def test_performance(model, env, test_name, num_episodes=5):
    """Test model performance in given environment"""

    distances = []
    velocities = []
    rewards = []

    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        positions = []

        for step in range(999):  # Standard episode length
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]

            # Track position
            try:
                # Get position from environment
                base_env = env.envs[0]
                while hasattr(base_env, 'env'):
                    base_env = base_env.env

                if hasattr(base_env, 'unwrapped') and hasattr(base_env.unwrapped, 'data'):
                    x_pos = float(base_env.unwrapped.data.qpos[0])
                    positions.append(x_pos)
            except:
                pass

            if done[0]:
                break

        # Calculate distance and velocity
        if len(positions) > 1:
            distance = positions[-1] - positions[0]  # Net displacement
            velocity = distance / (len(positions) * 0.05)  # 50ms timesteps
        else:
            distance = 0.0
            velocity = 0.0

        distances.append(abs(distance))
        velocities.append(abs(velocity))
        rewards.append(episode_reward)

    # Statistics
    avg_distance = np.mean(distances)
    avg_velocity = np.mean(velocities)
    avg_reward = np.mean(rewards)
    std_velocity = np.std(velocities)

    print(f"   {test_name:20s}: {avg_velocity:.3f} ± {std_velocity:.3f} m/s | Distance: {avg_distance:.1f}m | Reward: {avg_reward:.0f}")

    return {
        'avg_distance': avg_distance,
        'avg_velocity': avg_velocity,
        'avg_reward': avg_reward,
        'std_velocity': std_velocity,
        'distances': distances,
        'velocities': velocities,
        'rewards': rewards
    }

if __name__ == "__main__":
    try:
        results = test_systematic_curriculum_model()
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()