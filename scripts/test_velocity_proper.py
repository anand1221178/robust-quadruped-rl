#!/usr/bin/env python3
"""
Proper velocity testing script that uses the correct environment wrapper
"""
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from agents.ppo_sr2l import PPO_SR2L
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
from envs.target_walking_wrapper import TargetWalkingWrapper
import realant_sim
import argparse

def test_model_velocity(model_path, norm_path=None, use_target_walking=False, use_success_reward=False):
    """Test what velocity a trained model actually achieves with correct wrapper"""
    
    # Create environment with appropriate wrapper
    def make_env():
        env = gym.make('RealAntMujoco-v0')
        
        if use_target_walking:
            print("Using TargetWalkingWrapper (goal-directed navigation)")
            env = TargetWalkingWrapper(env, target_distance=5.0)
        elif use_success_reward:
            print("Using SuccessRewardWrapper (speed-focused)")
            env = SuccessRewardWrapper(env)
        else:
            print("Using base environment (no wrapper)")
        
        return env
    
    env = DummyVecEnv([make_env])
    
    if norm_path:
        env = VecNormalize.load(norm_path, env)
        env.training = False
        print(f"Loaded VecNormalize from {norm_path}")
    
    # Load model (try SR2L first, then regular PPO)
    try:
        model = PPO_SR2L.load(model_path)
        print(f"Loaded SR2L model from {model_path}")
    except:
        model = PPO.load(model_path)
        print(f"Loaded PPO model from {model_path}")
    
    # Test for 5 episodes
    velocities = []
    distances = []
    targets_reached = 0
    
    for episode in range(5):
        obs = env.reset()
        episode_velocities = []
        initial_x = 0
        
        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Extract velocity from info
            if info[0] and 'current_velocity' in info[0]:
                episode_velocities.append(info[0]['current_velocity'])
            elif info[0] and 'speed' in info[0]:  # TargetWalkingWrapper uses 'speed'
                episode_velocities.append(info[0]['speed'])
            
            # Check if target reached (for target walking)
            if info[0] and 'success_bonus' in info[0] and info[0]['success_bonus'] > 0:
                targets_reached += 1
            
            if done[0]:
                break
        
        # Calculate episode metrics
        if episode_velocities:
            avg_velocity = np.mean([v for v in episode_velocities if v > 0])  # Only forward velocities
            velocities.append(avg_velocity)
        
        # Get distance traveled
        if info[0] and 'distance_traveled' in info[0]:
            distances.append(info[0]['distance_traveled'])
        elif info[0] and 'distance_to_target' in info[0]:
            # For target walking, estimate distance from initial
            distances.append(5.0 - info[0]['distance_to_target'])  # Approximate
        
        if episode_velocities:
            print(f"Episode {episode+1}: Avg velocity = {avg_velocity:.4f} m/s")
    
    # Overall results
    overall_avg = np.mean(velocities) if velocities else 0
    overall_std = np.std(velocities) if velocities else 0
    
    print(f"\n{'='*50}")
    print("OVERALL RESULTS:")
    print(f"{'='*50}")
    print(f"Average velocity: {overall_avg:.4f} ± {overall_std:.4f} m/s")
    if distances:
        print(f"Average distance: {np.mean(distances):.4f} m")
    if use_target_walking and targets_reached > 0:
        print(f"Targets reached: {targets_reached}")
    print(f"Target velocity: 1.0 m/s (realistic for RealAnt)")
    print(f"Achievement: {(overall_avg/1.0)*100:.1f}% of target")
    print(f"{'='*50}")
    
    env.close()
    return overall_avg, overall_std

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='Path to model')
    parser.add_argument('--norm_path', type=str, default=None, help='Path to vec_normalize')
    parser.add_argument('--target-walking', action='store_true', help='Use TargetWalkingWrapper')
    parser.add_argument('--success-reward', action='store_true', help='Use SuccessRewardWrapper')
    args = parser.parse_args()
    
    # Auto-detect wrapper based on model path
    if 'target_walking' in args.model_path:
        args.target_walking = True
        print("Auto-detected: Using TargetWalkingWrapper based on model path")
    elif 'fast_walking' in args.model_path or 'success' in args.model_path:
        args.success_reward = True
        print("Auto-detected: Using SuccessRewardWrapper based on model path")
    elif 'sr2l' in args.model_path:
        # SR2L models trained with target walking in our case
        args.target_walking = True
        print("Auto-detected: Using TargetWalkingWrapper for SR2L model")
    
    test_model_velocity(
        args.model_path, 
        args.norm_path,
        args.target_walking,
        args.success_reward
    )