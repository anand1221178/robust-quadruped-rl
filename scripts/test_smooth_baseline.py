#!/usr/bin/env python3
"""
Test the new SmoothTargetWrapper baseline model
"""
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.smooth_target_wrapper import SmoothTargetWrapper
import realant_sim

def test_smooth_baseline():
    """Test the completed SmoothTargetWrapper baseline model"""
    
    print("🎯 TESTING SMOOTH BASELINE MODEL")
    print("=" * 60)
    
    # Model paths from cluster
    model_path = 'experiments/ppo_smooth_baseline_rohl32fn/final_model.zip'
    norm_path = 'experiments/ppo_smooth_baseline_rohl32fn/vec_normalize.pkl'
    
    # Create environment with SmoothTargetWrapper (same as training)
    def make_env():
        env = gym.make('RealAntMujoco-v0')
        env = SmoothTargetWrapper(env, target_distance=5.0)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Load VecNormalize (should work now - same obs space)
    try:
        env = VecNormalize.load(norm_path, env)
        env.training = False
        print("✅ VecNormalize loaded successfully (obs space matches!)")
    except Exception as e:
        print(f"⚠️  VecNormalize failed: {e}")
        print("   Will test without normalization")
    
    # Load model
    try:
        model = PPO.load(model_path)
        print(f"✅ Model loaded from: {model_path}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    print(f"\n🤖 TESTING GOAL-DIRECTED LOCOMOTION:")
    print("-" * 50)
    
    # Test for multiple episodes
    episode_results = []
    
    for episode in range(5):
        print(f"\n📍 EPISODE {episode + 1}:")
        
        obs = env.reset()
        episode_reward = 0
        targets_reached = 0
        distances_traveled = []
        velocities = []
        smoothness_scores = []
        positions = []
        
        for step in range(1000):  # Full episode
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0]
            
            # Extract position for distance calculation
            if hasattr(env.envs[0], 'unwrapped'):
                current_x = env.envs[0].unwrapped.data.qpos[0]
                positions.append(current_x)
            
            # Extract custom metrics from SmoothTargetWrapper
            if info[0]:
                if 'distance_to_target' in info[0]:
                    # Don't print every step - too verbose
                    pass
                if 'velocity' in info[0]:
                    velocities.append(info[0]['velocity'])
                if 'smoothness_score' in info[0]:
                    smoothness_scores.append(info[0]['smoothness_score'])
                if 'target_reached' in info[0] and info[0]['target_reached']:
                    targets_reached += 1
                    current_pos = positions[-1] if positions else 0
                    print(f"  🎯 Target {targets_reached} reached at position x={current_pos:.1f}m (step {step})")
            
            if done[0]:
                print(f"  Episode ended at step {step}")
                break
        
        # Calculate episode stats
        total_distance = positions[-1] - positions[0] if len(positions) >= 2 else 0
        avg_velocity = np.mean(velocities) if velocities else 0
        avg_smoothness = np.mean(smoothness_scores) if smoothness_scores else 0
        
        episode_results.append({
            'reward': episode_reward,
            'targets': targets_reached,
            'distance': total_distance,
            'velocity': avg_velocity,
            'smoothness': avg_smoothness,
            'steps': step + 1
        })
        
        print(f"  📊 Results: {episode_reward:.0f} reward, {targets_reached} targets, {total_distance:.1f}m, {avg_velocity:.3f} m/s")
    
    # Overall results
    print(f"\n📈 OVERALL RESULTS:")
    print("=" * 60)
    
    avg_reward = np.mean([r['reward'] for r in episode_results])
    avg_targets = np.mean([r['targets'] for r in episode_results])
    avg_distance = np.mean([r['distance'] for r in episode_results])
    avg_velocity = np.mean([r['velocity'] for r in episode_results])
    avg_smoothness = np.mean([r['smoothness'] for r in episode_results])
    
    print(f"✅ Average Episode Reward: {avg_reward:.0f}")
    print(f"✅ Average Targets Reached: {avg_targets:.1f} per episode")
    print(f"✅ Average Distance Traveled: {avg_distance:.1f} meters")
    print(f"✅ Average Velocity: {avg_velocity:.3f} m/s")
    print(f"✅ Average Smoothness Score: {avg_smoothness:.3f}")
    
    # Success assessment
    print(f"\n🎯 SUCCESS ASSESSMENT:")
    if avg_targets >= 1.0:
        print("🎉 EXCELLENT: Robot consistently reaches targets!")
    elif avg_targets >= 0.5:
        print("✅ GOOD: Robot reaches targets most episodes")
    else:
        print("⚠️  POOR: Robot struggling to reach targets")
    
    if avg_velocity > 0.2:
        print("🚀 GOOD: Healthy walking speed")
    else:
        print("🐌 SLOW: Walking speed could be better")
    
    if avg_distance > 10:
        print("🏃 EXCELLENT: Covers good distance per episode")
    elif avg_distance > 5:
        print("👍 DECENT: Covers reasonable distance")
    else:
        print("🚶 LIMITED: Not traveling very far")
    
    env.close()
    return episode_results

if __name__ == "__main__":
    results = test_smooth_baseline()
