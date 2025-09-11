#!/usr/bin/env python3
"""
Comprehensive Robustness Test Suite
Tests models under various failure conditions with proper metrics
"""
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
from envs.domain_randomization_wrapper import DomainRandomizationWrapper
import realant_sim
import argparse
from datetime import datetime
import json

def test_robustness_condition(model_path, vec_normalize_path, model_name, 
                            test_name, dr_config=None, episodes=10):
    """Test model under specific robustness condition"""
    print(f"\n🔬 TESTING: {model_name} - {test_name}")
    print("=" * 80)
    
    def make_env():
        env = gym.make('RealAntMujoco-v0')
        env = SuccessRewardWrapper(env)
        
        if dr_config:
            env = DomainRandomizationWrapper(env, dr_config)
            print(f"  🎲 DR Config: {dr_config}")
        
        return env
    
    env = DummyVecEnv([make_env])
    
    # Load VecNormalize
    try:
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
        print("  ✅ VecNormalize loaded")
    except:
        print("  ⚠️  No VecNormalize")
    
    # Load model
    model = PPO.load(model_path)
    print(f"  ✅ Model loaded")
    
    # Run episodes
    results = []
    
    print(f"\n  Running {episodes} episodes...")
    print("  " + "-" * 70)
    
    for episode in range(episodes):
        obs = env.reset()
        
        positions = []
        rewards = []
        joint_failures = []
        
        for step in range(1000):
            # Get action
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Track position
            x_pos = env.envs[0].unwrapped.data.qpos[0]
            positions.append(x_pos)
            rewards.append(reward[0])
            
            # Track joint failures if DR is enabled
            if dr_config:
                if hasattr(env.envs[0], 'failed_joints'):
                    failed_joints = getattr(env.envs[0], 'failed_joints', [])
                    joint_failures.append(failed_joints.copy() if failed_joints else [])
                else:
                    # Try to get from wrapper
                    for wrapper in [env.envs[0]] + [getattr(env.envs[0], 'env', None)]:
                        if wrapper and hasattr(wrapper, 'failed_joints'):
                            failed_joints = getattr(wrapper, 'failed_joints', [])
                            joint_failures.append(failed_joints.copy() if failed_joints else [])
                            break
                    else:
                        joint_failures.append([])
            
            if done[0]:
                break
        
        # Calculate metrics
        if len(positions) >= 2:
            # Total distance traveled
            total_distance = sum(abs(positions[i] - positions[i-1]) for i in range(1, len(positions)))
            
            # Net displacement
            displacement = positions[-1] - positions[0]
            
            # Max distance from origin
            max_distance = max(abs(p) for p in positions)
            
            total_reward = sum(rewards)
            episode_length = len(positions)
            
            # Joint failure analysis
            unique_failures = set()
            failure_rate = 0
            if joint_failures:
                for step_failures in joint_failures:
                    unique_failures.update(step_failures)
                failure_rate = len([f for f in joint_failures if f]) / len(joint_failures)
            
            result = {
                'episode': episode + 1,
                'total_distance': total_distance,
                'displacement': displacement,
                'max_distance': max_distance,
                'total_reward': total_reward,
                'episode_length': episode_length,
                'unique_failed_joints': list(unique_failures),
                'failure_rate': failure_rate,
                'fell': done[0]
            }
            
            results.append(result)
            
            print(f"  Episode {episode+1:2d}: TotalDist={total_distance:6.1f}m, "
                  f"Disp={displacement:5.1f}m, Reward={total_reward:6.0f}, "
                  f"Failures={list(unique_failures) if unique_failures else 'None'}")
        
        else:
            print(f"  Episode {episode+1:2d}: FAILED (too short)")
    
    env.close()
    
    # Calculate statistics
    if results:
        distances = [r['total_distance'] for r in results]
        displacements = [r['displacement'] for r in results]
        rewards = [r['total_reward'] for r in results]
        
        stats = {
            'test_name': test_name,
            'episodes': len(results),
            'avg_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'avg_displacement': np.mean(displacements),
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'success_rate': len([r for r in results if r['total_distance'] > 2]) / len(results),
            'all_failed_joints': list(set().union(*[r['unique_failed_joints'] for r in results])),
            'avg_failure_rate': np.mean([r['failure_rate'] for r in results])
        }
        
        print(f"\n  📊 SUMMARY:")
        print(f"    Average Distance:   {stats['avg_distance']:6.1f} ± {stats['std_distance']:.1f} m")
        print(f"    Average Displacement: {stats['avg_displacement']:6.1f} m")
        print(f"    Average Reward:     {stats['avg_reward']:6.0f} ± {stats['std_reward']:.0f}")
        print(f"    Success Rate:       {stats['success_rate']*100:5.1f}% (episodes >2m)")
        
        if dr_config:
            print(f"    Joint Failure Rate: {stats['avg_failure_rate']*100:5.1f}%")
            print(f"    Failed Joints:      {stats['all_failed_joints']}")
        
        # Performance rating
        if stats['avg_distance'] > 15:
            rating = "🏆 EXCELLENT"
        elif stats['avg_distance'] > 8:
            rating = "✅ GOOD"
        elif stats['avg_distance'] > 3:
            rating = "⚠️  MODERATE"  
        else:
            rating = "❌ POOR"
        
        print(f"    Performance:        {rating}")
        
        return stats
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Comprehensive robustness test')
    parser.add_argument('--model', type=str, required=True, help='Path to model.zip')
    parser.add_argument('--vec', type=str, required=True, help='Path to vec_normalize.pkl')
    parser.add_argument('--name', type=str, required=True, help='Model name')
    parser.add_argument('--episodes', type=int, default=10, help='Episodes per test')
    args = parser.parse_args()
    
    print("🚀 COMPREHENSIVE ROBUSTNESS TEST SUITE")
    print(f"Testing: {args.name}")
    print("=" * 80)
    
    all_results = []
    
    # Test 1: Baseline (no failures)
    result = test_robustness_condition(
        args.model, args.vec, args.name,
        "BASELINE (No Failures)",
        dr_config=None,
        episodes=args.episodes
    )
    if result: all_results.append(result)
    
    # Test 2: Low joint failure rate
    result = test_robustness_condition(
        args.model, args.vec, args.name,
        "LOW FAILURES (2% joint dropout)",
        dr_config={
            'joint_dropout_prob': 0.02,
            'max_dropped_joints': 1,
            'min_dropped_joints': 0,
            'sensor_noise_std': 0.0,
            'noise_joints_only': True
        },
        episodes=args.episodes
    )
    if result: all_results.append(result)
    
    # Test 3: Moderate joint failure rate
    result = test_robustness_condition(
        args.model, args.vec, args.name,
        "MODERATE FAILURES (5% joint dropout)",
        dr_config={
            'joint_dropout_prob': 0.05,
            'max_dropped_joints': 1,
            'min_dropped_joints': 0,
            'sensor_noise_std': 0.0,
            'noise_joints_only': True
        },
        episodes=args.episodes
    )
    if result: all_results.append(result)
    
    # Test 4: High joint failure rate
    result = test_robustness_condition(
        args.model, args.vec, args.name,
        "HIGH FAILURES (10% joint dropout)",
        dr_config={
            'joint_dropout_prob': 0.10,
            'max_dropped_joints': 2,
            'min_dropped_joints': 0,
            'sensor_noise_std': 0.0,
            'noise_joints_only': True
        },
        episodes=args.episodes
    )
    if result: all_results.append(result)
    
    # Test 5: Sensor noise only
    result = test_robustness_condition(
        args.model, args.vec, args.name,
        "SENSOR NOISE (1% noise, no failures)",
        dr_config={
            'joint_dropout_prob': 0.0,
            'max_dropped_joints': 0,
            'min_dropped_joints': 0,
            'sensor_noise_std': 0.01,
            'noise_joints_only': True
        },
        episodes=args.episodes
    )
    if result: all_results.append(result)
    
    # Test 6: Combined failures + noise
    result = test_robustness_condition(
        args.model, args.vec, args.name,
        "COMBINED (5% failures + 0.5% noise)",
        dr_config={
            'joint_dropout_prob': 0.05,
            'max_dropped_joints': 1,
            'min_dropped_joints': 0,
            'sensor_noise_std': 0.005,
            'noise_joints_only': True
        },
        episodes=args.episodes
    )
    if result: all_results.append(result)
    
    # Final comparison
    print("\n" + "=" * 80)
    print("🏆 FINAL ROBUSTNESS REPORT")
    print("=" * 80)
    
    if all_results:
        baseline_distance = all_results[0]['avg_distance'] if all_results else 10
        
        for result in all_results:
            retention = (result['avg_distance'] / baseline_distance) * 100 if baseline_distance > 0 else 0
            success_pct = result['success_rate'] * 100
            
            print(f"{result['test_name']:35} | "
                  f"Dist: {result['avg_distance']:5.1f}m | "
                  f"Retention: {retention:5.1f}% | "
                  f"Success: {success_pct:5.1f}%")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"robustness_results_{args.name.replace('-', '_')}_{timestamp}.json"
        
        # Convert numpy types to native Python types for JSON serialization
        serializable_results = []
        for result in all_results:
            serializable_result = {}
            for key, value in result.items():
                if hasattr(value, 'item'):  # numpy scalar
                    serializable_result[key] = value.item()
                elif isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                else:
                    serializable_result[key] = value
            serializable_results.append(serializable_result)
        
        with open(filename, 'w') as f:
            json.dump({
                'model_name': args.name,
                'test_date': timestamp,
                'results': serializable_results
            }, f, indent=2)
        
        print(f"\n📁 Detailed results saved to: {filename}")
        
        # Overall assessment
        avg_retention = np.mean([
            (r['avg_distance'] / baseline_distance) * 100 
            for r in all_results[1:] if baseline_distance > 0
        ])
        
        print(f"\n🎯 OVERALL ROBUSTNESS SCORE: {avg_retention:.1f}%")
        
        if avg_retention > 80:
            print("🏆 VERDICT: EXCELLENT robustness!")
        elif avg_retention > 60:
            print("✅ VERDICT: GOOD robustness")
        elif avg_retention > 40:
            print("⚠️  VERDICT: MODERATE robustness")
        else:
            print("❌ VERDICT: POOR robustness")

if __name__ == "__main__":
    main()