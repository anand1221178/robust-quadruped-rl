#!/usr/bin/env python3
"""
Debug script to check actual velocities achieved by trained model
"""
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.target_walking_wrapper import TargetWalkingWrapper
import realant_sim

def test_model_velocity(model_path, norm_path=None):
    """Test what velocity a trained model actually achieves
    
    UPDATED: Now uses correct evaluation setup for baseline model
    """
    
    # Create environment - UPDATED FOR BASELINE COMPATIBILITY
    def make_env():
        env = gym.make('RealAntMujoco-v0')
        # NOTE: Baseline model (ppo_baseline_ueqbjf2x) doesn't use TargetWalkingWrapper
        # For other models that need it, add wrapper based on model path
        if "baseline_ueqbjf2x" not in model_path:
            env = TargetWalkingWrapper(env, target_distance=5.0)
        return env
    
    env = DummyVecEnv([make_env])
    
    if norm_path:
        env = VecNormalize.load(norm_path, env)
        env.training = False
    elif "baseline_ueqbjf2x" in model_path:
        # Auto-detect VecNormalize for baseline model
        import os
        auto_norm_path = os.path.join(os.path.dirname(os.path.dirname(model_path)), 'vec_normalize.pkl')
        if os.path.exists(auto_norm_path):
            env = VecNormalize.load(auto_norm_path, env)
            env.training = False
            print(f"Auto-loaded VecNormalize from: {auto_norm_path}")
    
    # Load model
    model = PPO.load(model_path)
    
    # Test for 5 episodes
    velocities = []
    distances = []
    
    for episode in range(5):
        obs = env.reset()
        episode_velocities = []
        total_distance = 0
        positions = []  # Track positions to calculate velocity
        
        for step in range(500):  # Max episode length
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Get current position directly from environment
            if hasattr(env, 'envs'):
                current_x = env.envs[0].unwrapped.data.qpos[0]
            else:
                current_x = env.unwrapped.data.qpos[0]
            
            positions.append(current_x)
            
            # Calculate velocity from position changes
            if len(positions) >= 2:
                vel = (positions[-1] - positions[-2]) / 0.05  # dt = 0.05
                episode_velocities.append(vel)
            
            # Get velocity from info if available (for TargetWalkingWrapper models)
            if info[0] and 'speed' in info[0]:
                vel = info[0]['speed']
                episode_velocities.append(vel)
            elif info[0] and 'progress' in info[0]:
                # Calculate velocity from progress
                vel = info[0]['progress'] / 0.05
                episode_velocities.append(vel)
            
            if done[0]:
                break
        
        avg_velocity = np.mean(episode_velocities) if episode_velocities else 0.0
        # Calculate total distance from positions
        if len(positions) >= 2:
            total_distance = positions[-1] - positions[0]
        else:
            total_distance = 0.0
            
        velocities.append(avg_velocity)
        distances.append(total_distance)
        
        print(f"Episode {episode+1}: Avg velocity = {avg_velocity:.4f} m/s, Distance = {total_distance:.4f} m")
    
    overall_avg = np.mean(velocities)
    overall_std = np.std(velocities)
    
    print(f"\nOverall Results:")
    print(f"Average velocity: {overall_avg:.4f} ± {overall_std:.4f} m/s")
    print(f"Average distance: {np.mean(distances):.4f} m")
    print(f"Target velocity: 2.0 m/s")
    print(f"Achievement: {(overall_avg/2.0)*100:.1f}% of target")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test model velocity - UPDATED with correct baseline')
    parser.add_argument('model_path', type=str, 
                       help='Path to model (use archive/experiments/ppo_baseline_ueqbjf2x/best_model/best_model.zip for baseline)')
    parser.add_argument('--norm_path', type=str, default=None, help='Path to vec_normalize (auto-detected for baseline)')
    args = parser.parse_args()
    
    print(f"Testing model: {args.model_path}")
    if "baseline_ueqbjf2x" in args.model_path:
        print("✅ Using CORRECT baseline model (smooth walking)")
    elif "target_walking_llsm451b" in args.model_path:
        print("⚠️  WARNING: Using old problematic baseline - expect erratic behavior")
    
    test_model_velocity(args.model_path, args.norm_path)