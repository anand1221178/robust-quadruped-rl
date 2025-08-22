#!/usr/bin/env python3
"""
Simple script to record video of trained models
"""

import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
from envs.target_walking_wrapper import TargetWalkingWrapper
import realant_sim
import imageio
import argparse
import os

def record_video(model_path, vec_normalize_path=None, output_path="walking_video.mp4", use_target_walking=False):
    """Record video of trained model"""
    
    # Load model
    model = PPO.load(model_path)
    print(f"Loaded PPO model from {model_path}")
    
    # Create environment with rgb_array for recording
    def make_env():
        env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')
        
        # Apply same wrapper as training
        if use_target_walking:
            env = TargetWalkingWrapper(env, target_distance=5.0)
            print("Using Target Walking Wrapper")
        else:
            env = SuccessRewardWrapper(env)
            print("Using Success Reward Wrapper")
        
        return env
    
    # Create vectorized environment
    env = DummyVecEnv([make_env])
    
    # Load normalization if provided
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False  # Don't update stats during recording
        print(f"Loaded VecNormalize from {vec_normalize_path}")
    
    # Record one episode
    obs = env.reset()
    frames = []
    episode_reward = 0
    velocities = []
    distances = []
    step_count = 0
    
    print("Recording video...")
    for step in range(1500):  # Max 1500 steps (~30 seconds at 50Hz)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # Get frame
        frame = env.render(mode='rgb_array')
        if frame is not None:
            frames.append(frame)
        
        episode_reward += reward[0]
        step_count += 1
        
        # Track metrics from info
        if len(info) > 0 and info[0] is not None:
            if 'current_velocity' in info[0]:
                velocities.append(info[0]['current_velocity'])
            if 'distance_traveled' in info[0]:
                distances.append(info[0]['distance_traveled'])
        
        if done[0]:
            print(f"Episode ended at step {step}")
            break
    
    # Save video
    if frames:
        print(f"Saving video to {output_path}...")
        imageio.mimsave(output_path, frames, fps=50)
        print(f"Video saved! ({len(frames)} frames)")
    else:
        print("No frames captured!")
        return
    
    # Print metrics
    print("\n" + "="*50)
    print("EPISODE METRICS:")
    print("="*50)
    print(f"Total Reward: {episode_reward:.2f}")
    print(f"Episode Length: {step_count} steps")
    if velocities:
        print(f"Average Velocity: {np.mean(velocities):.3f} m/s")
        print(f"Max Velocity: {np.max(velocities):.3f} m/s")
    if distances:
        print(f"Total Distance: {distances[-1]:.3f} m")
    print("="*50)
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--norm', type=str, help='Path to VecNormalize file')
    parser.add_argument('--output', type=str, default='walking_video.mp4', help='Output video path')
    parser.add_argument('--target-walking', action='store_true', help='Use target walking wrapper')
    
    args = parser.parse_args()
    
    record_video(
        args.model, 
        args.norm, 
        args.output,
        args.target_walking
    )