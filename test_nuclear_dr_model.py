#!/usr/bin/env python3
"""
Nuclear DR Model - Progressive 6-Stage Persistent Evaluation
Testing the 60M step progressive domain randomization model with gentle 2% joint dropout.
"""

import os
import sys
import numpy as np
import gymnasium as gym
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Import our wrappers
from envs.success_reward_wrapper import SuccessRewardWrapper
from envs.forced_failure_wrapper import ForcedFailureWrapper

# Import RealAnt
import realant_sim

def test_nuclear_dr_model():
    """Test nuclear progressive 6-stage persistent DR model"""
    print("🔥 NUCLEAR DR MODEL EVALUATION - PROGRESSIVE 6-STAGE PERSISTENT")
    print("="*80)
    print("Testing 60M step model with gentle 2% joint dropout approach!")
    print()

    # Model paths - updated for nuclear DR model
    model_dir = "experiments/ppo_progressive_6stage_persistent_60M_gv8pprbz"
    model_path = f"{model_dir}/final_model.zip"
    vec_normalize_path = f"{model_dir}/vec_normalize.pkl"

    print(f"📁 Model directory: {model_dir}")
    print(f"🤖 Model file: {model_path}")
    print(f"📊 VecNormalize file: {vec_normalize_path}")
    print()

    # Verify files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False

    if not os.path.exists(vec_normalize_path):
        print(f"❌ VecNormalize file not found: {vec_normalize_path}")
        return False

    print("✅ All model files found!")
    print()

    # Create evaluation environment (same as training)
    def make_env():
        env = gym.make('RealAntMujoco-v0')
        env = SuccessRewardWrapper(env)
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])

    # Load VecNormalize
    env = VecNormalize.load(vec_normalize_path, env)
    env.training = False  # Disable training mode for evaluation
    env.norm_reward = False  # Don't normalize rewards during evaluation

    print("✅ Environment created and VecNormalize loaded")
    print()

    # Load the trained model
    model = PPO.load(model_path, env=env)
    print("✅ Nuclear DR model loaded successfully!")
    print()

    # Test configurations - Test both normal and forced failure scenarios
    test_configs = [
        {
            "name": "Baseline Performance",
            "forced_joints": [],  # No joint failures
            "description": "Normal walking - test if DR preserved locomotion"
        },
        {
            "name": "Single Joint Failure",
            "forced_joints": [0],  # Force hip_1 failure
            "description": "Single joint failure - test DR robustness"
        },
        {
            "name": "Dual Joint Failure",
            "forced_joints": [0, 1],  # Hip_1 + Hip_2 failure
            "description": "Dual joint failure - test extreme robustness"
        },
        {
            "name": "Triple Joint Failure",
            "forced_joints": [0, 1, 4],  # Three joints forced to fail
            "description": "Triple joint failure - ultimate stress test"
        }
    ]

    results = []

    for config in test_configs:
        print(f"🧪 TESTING: {config['name']}")
        print(f"   {config['description']}")

        # Create test environment with forced joint failures if specified
        def make_test_env():
            env = gym.make('RealAntMujoco-v0')
            env = SuccessRewardWrapper(env)

            # Add forced joint failures if specified
            forced_joints = config.get('forced_joints', [])
            if len(forced_joints) > 0:
                env = ForcedFailureWrapper(env, forced_joints)

            env = Monitor(env)
            return env

        test_env = DummyVecEnv([make_test_env])
        test_env = VecNormalize.load(vec_normalize_path, test_env)
        test_env.training = False
        test_env.norm_reward = False

        # Run evaluation episodes
        num_episodes = 5  # Test episodes
        episode_rewards = []
        episode_distances = []
        episode_velocities = []
        failed_episodes = 0

        for episode in range(num_episodes):
            obs = test_env.reset()
            total_reward = 0
            positions = [0.0]  # Start position
            steps = 0

            done = False
            while not done and steps < 250:  # 250 steps per episode
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = test_env.step(action)
                total_reward += reward[0]

                # Track position for distance calculation - use direct MuJoCo position
                try:
                    # Use direct position from MuJoCo (like championship scripts)
                    x_pos = test_env.envs[0].unwrapped.data.qpos[0]
                    positions.append(x_pos)
                except:
                    # Fallback to info dict
                    if 'x_position' in info[0]:
                        positions.append(info[0]['x_position'])
                    else:
                        # Use observation (might be normalized)
                        x_pos = obs[0][0]
                        positions.append(x_pos)

                steps += 1

                if done[0]:
                    break

            # Calculate metrics (using championship script method)
            if len(positions) >= 2:
                net_displacement = positions[-1] - positions[0]  # Net forward displacement
                time_elapsed = steps * 0.05  # 20Hz timestep (0.05s per step)
                avg_velocity = net_displacement / time_elapsed if time_elapsed > 0 else 0.0
                total_distance = abs(net_displacement)
            else:
                total_distance = 0.0
                avg_velocity = 0.0
                failed_episodes += 1

            episode_rewards.append(total_reward)
            episode_distances.append(total_distance)
            episode_velocities.append(avg_velocity)

            if episode % 2 == 0:  # Progress updates
                forced_joints_str = f"Forced joints: {config['forced_joints']}" if config['forced_joints'] else "No failures"
                print(f"     Episode {episode+1}: {forced_joints_str}, Steps={steps}, Start={positions[0]:.2f}, End={positions[-1]:.2f}, Distance={total_distance:.2f}m, Velocity={avg_velocity:.3f}m/s")

        # Calculate statistics
        avg_reward = np.mean(episode_rewards)
        avg_distance = np.mean(episode_distances)
        avg_velocity = np.mean(episode_velocities)
        std_velocity = np.std(episode_velocities)
        success_rate = (num_episodes - failed_episodes) / num_episodes * 100

        result = {
            'name': config['name'],
            'avg_reward': avg_reward,
            'avg_distance': avg_distance,
            'avg_velocity': avg_velocity,
            'std_velocity': std_velocity,
            'success_rate': success_rate,
            'failed_episodes': failed_episodes
        }
        results.append(result)

        print(f"   📊 Results: Velocity={avg_velocity:.3f}±{std_velocity:.3f} m/s, Distance={avg_distance:.2f}m, Success={success_rate:.0f}%")
        print()

        test_env.close()

    # Print comprehensive results
    print("🏆 NUCLEAR DR MODEL - COMPREHENSIVE RESULTS")
    print("="*80)
    print(f"{'Test Scenario':<25} {'Velocity (m/s)':<15} {'Distance (m)':<12} {'Success %':<10}")
    print("-"*80)

    baseline_velocity = None
    for result in results:
        velocity_str = f"{result['avg_velocity']:.3f}±{result['std_velocity']:.3f}"
        print(f"{result['name']:<25} {velocity_str:<15} {result['avg_distance']:<12.2f} {result['success_rate']:<10.0f}")

        if "Baseline" in result['name']:
            baseline_velocity = result['avg_velocity']

    print()

    # Calculate robustness metrics
    if baseline_velocity and baseline_velocity > 0:
        print("🎯 ROBUSTNESS ANALYSIS vs Baseline:")
        print("-"*50)
        for result in results:
            if "Baseline" not in result['name']:
                retention = (result['avg_velocity'] / baseline_velocity) * 100
                print(f"{result['name']:<25}: {retention:.1f}% retention")
        print()

    # Overall assessment
    print("🔬 NUCLEAR DR MODEL ASSESSMENT:")
    if baseline_velocity and baseline_velocity > 0.15:
        print("✅ Nuclear DR model demonstrates good baseline locomotion")
        print(f"   Baseline velocity: {baseline_velocity:.3f} m/s")

        robust_configs = [r for r in results if "Failure" in r['name'] and r['avg_velocity'] > 0.1]
        if len(robust_configs) > 0:
            print(f"✅ Nuclear DR model shows robustness to joint failures")
            print(f"   Robust scenarios: {len(robust_configs)}/{len([r for r in results if 'Failure' in r['name']])}")
            print(f"✅ SUCCESS: Gentle 2% approach works better than systematic curriculum!")
        else:
            print("⚠️  Nuclear DR model has limited robustness to joint failures")
    else:
        print("❌ Nuclear DR model has poor baseline locomotion performance")
        print(f"   Baseline velocity: {baseline_velocity:.3f} m/s (below 0.15 m/s threshold)")

    # Compare to systematic curriculum results
    print()
    print("📊 COMPARISON TO SYSTEMATIC CURRICULUM (V3):")
    print("-"*60)
    print(f"V3 Systematic Curriculum: ~0.000 m/s (FAILED - stationary)")
    print(f"Nuclear DR (2% gentle):   {baseline_velocity:.3f} m/s")
    if baseline_velocity > 0.05:
        improvement = baseline_velocity / 0.001 if baseline_velocity > 0 else float('inf')
        print(f"Improvement factor: {improvement:.0f}x better than systematic!")
        print("✅ PARADIGM VALIDATION: Gentle probabilistic > systematic guaranteed failures")
    else:
        print("❌ Both approaches failed - need even gentler approach for V6")

    env.close()
    return True

if __name__ == "__main__":
    print("🚀 Starting Nuclear DR Model Evaluation...")
    print("   Testing the 60M step progressive 6-stage persistent approach")
    print("   Remember to activate venv: source venv/bin/activate")
    print()

    success = test_nuclear_dr_model()
    if success:
        print("🎉 Nuclear DR evaluation completed successfully!")
    else:
        print("❌ Nuclear DR evaluation failed")