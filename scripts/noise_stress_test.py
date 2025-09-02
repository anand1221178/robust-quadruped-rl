#!/usr/bin/env python3
"""
Noise Stress Test - Test model robustness under increasing sensor noise
Shows how performance degrades as noise increases
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import argparse
import os
import sys
from typing import List, Dict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import environments
import realant_sim
from envs.target_walking_wrapper import TargetWalkingWrapper

def test_with_noise(model_path: str, noise_level: float, episodes: int = 5, max_steps: int = 1000) -> Dict:
    """
    Test model performance with specific noise level
    UPDATED: Handles both baseline (no wrapper) and other models (with wrapper)
    
    Returns:
        Dictionary with performance metrics
    """
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    env = gym.make('RealAntMujoco-v0')
    env = TargetWalkingWrapper(env, target_distance=5.0)
    env = DummyVecEnv([lambda: env])
    
    # Load normalization if exists
    exp_dir = os.path.dirname(os.path.dirname(model_path))
    vec_norm_path = os.path.join(exp_dir, 'vec_normalize.pkl')
    if os.path.exists(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False
    
    # Metrics storage
    velocities = []
    distances = []
    rewards = []
    survived_steps = []
    falls = 0
    
    for episode in range(episodes):
        obs = env.reset()
        episode_velocity = []
        episode_reward = 0
        
        for step in range(max_steps):
            # Add sensor noise to observation
            if noise_level > 0:
                # Only add noise to joint sensors (dims 13-28)
                noise = np.zeros_like(obs)
                noise[:, 13:29] = np.random.normal(0, noise_level, (1, 16))
                obs_noisy = obs + noise
            else:
                obs_noisy = obs
            
            # Get action
            action, _ = model.predict(obs_noisy, deterministic=True)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            # Track metrics
            velocity = info[0].get('speed', 0) if info[0] else 0
            episode_velocity.append(velocity)
            episode_reward += reward[0]
            
            # Check if robot fell
            if done and step < max_steps - 1:
                falls += 1
                survived_steps.append(step)
                break
        else:
            survived_steps.append(max_steps)
        
        # Episode metrics
        avg_velocity = np.mean(episode_velocity)
        total_distance = sum([abs(v) * 0.05 for v in episode_velocity])
        
        velocities.append(avg_velocity)
        distances.append(total_distance)
        rewards.append(episode_reward)
    
    # Calculate statistics
    results = {
        'noise_level': noise_level,
        'avg_velocity': np.mean(velocities),
        'std_velocity': np.std(velocities),
        'avg_distance': np.mean(distances),
        'avg_reward': np.mean(rewards),
        'avg_survival_steps': np.mean(survived_steps),
        'fall_rate': falls / episodes,
        'success_rate': sum([s == max_steps for s in survived_steps]) / episodes
    }
    
    env.close()
    return results

def stress_test_models(model_paths: List[str], labels: List[str], 
                       noise_levels: List[float], episodes: int = 5):
    """
    Test multiple models under increasing noise levels
    """
    print("\n" + "="*70)
    print("NOISE STRESS TEST - Robustness Evaluation")
    print("="*70)
    
    # Test each model at each noise level
    all_results = {}
    
    for model_path, label in zip(model_paths, labels):
        print(f"\nTesting: {label}")
        print("-"*40)
        
        model_results = []
        
        for noise in noise_levels:
            print(f"  Noise level: {noise*100:.0f}%", end=" ")
            results = test_with_noise(model_path, noise, episodes)
            model_results.append(results)
            print(f"→ Vel: {results['avg_velocity']:.3f} m/s, "
                  f"Survival: {results['avg_survival_steps']:.0f} steps, "
                  f"Falls: {results['fall_rate']*100:.0f}%")
        
        all_results[label] = model_results
    
    # Print comparison table
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    # Velocity comparison
    print("\n📊 Average Velocity (m/s):")
    print(f"{'Noise %':<10}", end="")
    for label in labels:
        print(f"{label:<15}", end="")
    print()
    print("-"*40)
    
    for i, noise in enumerate(noise_levels):
        print(f"{noise*100:>6.0f}%   ", end="")
        for label in labels:
            vel = all_results[label][i]['avg_velocity']
            print(f"{vel:>6.3f}         ", end="")
        print()
    
    # Survival rate comparison
    print("\n⏱️ Average Survival Steps (out of 1000):")
    print(f"{'Noise %':<10}", end="")
    for label in labels:
        print(f"{label:<15}", end="")
    print()
    print("-"*40)
    
    for i, noise in enumerate(noise_levels):
        print(f"{noise*100:>6.0f}%   ", end="")
        for label in labels:
            steps = all_results[label][i]['avg_survival_steps']
            print(f"{steps:>6.0f}         ", end="")
        print()
    
    # Fall rate comparison
    print("\n💀 Fall Rate (%):")
    print(f"{'Noise %':<10}", end="")
    for label in labels:
        print(f"{label:<15}", end="")
    print()
    print("-"*40)
    
    for i, noise in enumerate(noise_levels):
        print(f"{noise*100:>6.0f}%   ", end="")
        for label in labels:
            falls = all_results[label][i]['fall_rate'] * 100
            print(f"{falls:>6.0f}         ", end="")
        print()
    
    # Create plots
    plot_results(all_results, labels, noise_levels)
    
    # Determine winner
    print("\n" + "="*70)
    print("ROBUSTNESS ANALYSIS")
    print("="*70)
    
    # Calculate robustness scores
    for label in labels:
        results = all_results[label]
        
        # Performance retention at high noise
        clean_vel = results[0]['avg_velocity']
        high_noise_vel = results[-1]['avg_velocity']
        retention = (high_noise_vel / clean_vel * 100) if clean_vel > 0 else 0
        
        # Average survival across all noise levels
        avg_survival = np.mean([r['avg_survival_steps'] for r in results])
        
        print(f"\n{label}:")
        print(f"  Clean performance: {clean_vel:.3f} m/s")
        print(f"  High noise performance: {high_noise_vel:.3f} m/s")
        print(f"  Performance retention: {retention:.1f}%")
        print(f"  Average survival: {avg_survival:.0f} steps")
    
    print("\n" + "="*70)

def plot_results(all_results: Dict, labels: List[str], noise_levels: List[float]):
    """Create visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Noise Stress Test - Robustness Comparison', fontsize=14)
    
    noise_percents = [n * 100 for n in noise_levels]
    
    # Velocity vs Noise
    ax = axes[0, 0]
    for label in labels:
        velocities = [r['avg_velocity'] for r in all_results[label]]
        errors = [r['std_velocity'] for r in all_results[label]]
        ax.errorbar(noise_percents, velocities, yerr=errors, 
                   marker='o', label=label, linewidth=2)
    ax.set_title('Velocity Degradation')
    ax.set_xlabel('Sensor Noise (%)')
    ax.set_ylabel('Average Velocity (m/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Survival Steps vs Noise
    ax = axes[0, 1]
    for label in labels:
        survival = [r['avg_survival_steps'] for r in all_results[label]]
        ax.plot(noise_percents, survival, marker='s', label=label, linewidth=2)
    ax.set_title('Survival Duration')
    ax.set_xlabel('Sensor Noise (%)')
    ax.set_ylabel('Steps Survived (out of 1000)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Fall Rate vs Noise
    ax = axes[1, 0]
    for label in labels:
        falls = [r['fall_rate'] * 100 for r in all_results[label]]
        ax.plot(noise_percents, falls, marker='^', label=label, linewidth=2)
    ax.set_title('Fall Rate')
    ax.set_xlabel('Sensor Noise (%)')
    ax.set_ylabel('Fall Rate (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Performance Retention
    ax = axes[1, 1]
    for label in labels:
        clean_vel = all_results[label][0]['avg_velocity']
        retention = [(r['avg_velocity'] / clean_vel * 100) if clean_vel > 0 else 0 
                    for r in all_results[label]]
        ax.plot(noise_percents, retention, marker='d', label=label, linewidth=2)
    ax.set_title('Performance Retention')
    ax.set_xlabel('Sensor Noise (%)')
    ax.set_ylabel('% of Clean Performance')
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Noise Stress Test for Robustness')
    parser.add_argument('--models', nargs='+', required=True, help='Model paths')
    parser.add_argument('--labels', nargs='+', help='Model labels')
    parser.add_argument('--noise-levels', type=str, default='0,0.05,0.1,0.15,0.2',
                       help='Comma-separated noise levels (0-1)')
    parser.add_argument('--episodes', type=int, default=5, help='Episodes per test')
    
    args = parser.parse_args()
    
    # Parse noise levels
    noise_levels = [float(n) for n in args.noise_levels.split(',')]
    
    # Set labels
    if args.labels:
        labels = args.labels
    else:
        labels = [f"Model {i+1}" for i in range(len(args.models))]
    
    if len(labels) != len(args.models):
        print("Error: Number of labels must match number of models")
        return
    
    stress_test_models(args.models, labels, noise_levels, args.episodes)

if __name__ == "__main__":
    main()