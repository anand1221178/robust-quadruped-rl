#!/usr/bin/env python3
"""
Comprehensive evaluation of DR v2 model performance
Tests robustness under various joint failure conditions
"""

import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
import realant_sim
import os

def test_with_failures(model, env, failure_rate=0.0, num_episodes=3):
    """Test model with simulated joint failures"""
    velocities = []
    distances = []
    falls = 0
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_vels = []
        episode_dist = 0
        
        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            
            # Apply joint failures
            if failure_rate > 0:
                for i in range(len(action[0])):
                    if np.random.random() < failure_rate:
                        failure_type = np.random.choice(['lock', 'weak', 'noise'])
                        if failure_type == 'lock':
                            action[0][i] = 0.0  # Joint locked
                        elif failure_type == 'weak':
                            action[0][i] *= 0.3  # Weak joint (30% power)
                        elif failure_type == 'noise':
                            action[0][i] += np.random.normal(0, 0.5)  # Noisy joint
            
            obs, reward, done, info = env.step(action)
            
            # Extract velocity
            if info[0] and 'current_velocity' in info[0]:
                vel = info[0]['current_velocity']
            elif info[0] and 'speed' in info[0]:
                vel = info[0]['speed']
            else:
                vel = 0.0
            
            episode_vels.append(vel)
            episode_dist = info[0].get('distance_traveled', episode_dist) if info[0] else episode_dist
            
            if done[0]:
                if step < 400:  # Early termination = fall
                    falls += 1
                break
        
        velocities.extend(episode_vels)
        distances.append(episode_dist)
    
    avg_velocity = np.mean(velocities)
    avg_distance = np.mean(distances)
    fall_rate = falls / num_episodes
    
    return avg_velocity, avg_distance, fall_rate

def main():
    print("=" * 60)
    print("🤖 DR v2 MODEL COMPREHENSIVE EVALUATION")
    print("=" * 60)
    
    # Models to test
    models = {
        'Baseline': 'done/ppo_baseline_ueqbjf2x/best_model/best_model.zip',
        'DR v2': 'done/ppo_dr_gentle_v2_wptws01u/best_model/best_model.zip'
    }
    
    # Failure rates to test
    failure_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    results = {}
    
    for model_name, model_path in models.items():
        print(f"\n📊 Testing {model_name}: {model_path}")
        print("-" * 60)
        
        # Load model
        model = PPO.load(model_path)
        
        # Create environment
        def make_env():
            env = gym.make('RealAntMujoco-v0')
            env = SuccessRewardWrapper(env)
            return env
        
        env = DummyVecEnv([make_env])
        
        # Load VecNormalize if available
        vec_paths = [
            model_path.replace('best_model.zip', '../vec_normalize.pkl'),
            model_path.replace('best_model.zip', '../../vec_normalize.pkl'),
            os.path.dirname(model_path).replace('best_model', 'vec_normalize.pkl')
        ]
        
        for vec_path in vec_paths:
            if os.path.exists(vec_path):
                env = VecNormalize.load(vec_path, env)
                env.training = False
                env.norm_reward = False
                print(f"  ✅ Loaded VecNormalize: {vec_path}")
                break
        
        results[model_name] = {}
        
        # Test at different failure rates
        for failure_rate in failure_rates:
            print(f"\n  Testing with {failure_rate*100:.0f}% joint failure rate...")
            
            avg_vel, avg_dist, fall_rate = test_with_failures(
                model, env, failure_rate, num_episodes=3
            )
            
            results[model_name][failure_rate] = {
                'velocity': avg_vel,
                'distance': avg_dist,
                'fall_rate': fall_rate
            }
            
            print(f"    Velocity: {avg_vel:.3f} m/s")
            print(f"    Distance: {avg_dist:.2f} m")
            print(f"    Fall rate: {fall_rate*100:.0f}%")
    
    # Print comparison
    print("\n" + "=" * 60)
    print("📈 PERFORMANCE COMPARISON")
    print("=" * 60)
    
    print("\n🏃 Velocity (m/s):")
    print(f"{'Failure Rate':<15}", end='')
    for model_name in models.keys():
        print(f"{model_name:>12}", end='')
    print()
    
    for failure_rate in failure_rates:
        print(f"{failure_rate*100:>3.0f}%           ", end='')
        for model_name in models.keys():
            vel = results[model_name][failure_rate]['velocity']
            print(f"{vel:>12.3f}", end='')
        print()
    
    print("\n💀 Fall Rate (%):")
    print(f"{'Failure Rate':<15}", end='')
    for model_name in models.keys():
        print(f"{model_name:>12}", end='')
    print()
    
    for failure_rate in failure_rates:
        print(f"{failure_rate*100:>3.0f}%           ", end='')
        for model_name in models.keys():
            fall_pct = results[model_name][failure_rate]['fall_rate'] * 100
            print(f"{fall_pct:>12.0f}", end='')
        print()
    
    # Calculate robustness metrics
    print("\n" + "=" * 60)
    print("🛡️ ROBUSTNESS METRICS")
    print("=" * 60)
    
    for model_name in models.keys():
        baseline_vel = results[model_name][0.0]['velocity']
        high_failure_vel = results[model_name][0.3]['velocity']
        retention = (high_failure_vel / baseline_vel * 100) if baseline_vel > 0 else 0
        
        avg_fall_rate = np.mean([results[model_name][fr]['fall_rate'] 
                                 for fr in failure_rates[1:]])  # Exclude 0%
        
        print(f"\n{model_name}:")
        print(f"  Performance retention at 30% failure: {retention:.1f}%")
        print(f"  Average fall rate under failures: {avg_fall_rate*100:.1f}%")
        print(f"  Baseline velocity: {baseline_vel:.3f} m/s")
        print(f"  Velocity at 30% failure: {high_failure_vel:.3f} m/s")
    
    print("\n" + "=" * 60)
    print("🎯 CONCLUSION")
    print("=" * 60)
    
    # Determine winner
    dr_retention = (results['DR v2'][0.3]['velocity'] / results['DR v2'][0.0]['velocity'] * 100) if results['DR v2'][0.0]['velocity'] > 0 else 0
    baseline_retention = (results['Baseline'][0.3]['velocity'] / results['Baseline'][0.0]['velocity'] * 100) if results['Baseline'][0.0]['velocity'] > 0 else 0
    
    if dr_retention > baseline_retention:
        print("✅ DR v2 shows BETTER robustness to joint failures!")
        print(f"   DR v2 retains {dr_retention:.1f}% vs Baseline {baseline_retention:.1f}%")
    else:
        print("❌ DR v2 shows WORSE robustness than baseline")
        print(f"   DR v2 retains {dr_retention:.1f}% vs Baseline {baseline_retention:.1f}%")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()