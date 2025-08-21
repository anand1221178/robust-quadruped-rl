#!/usr/bin/env python3
"""
Test trained models with action smoothing and record video
"""

import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from agents.ppo_sr2l import PPO_SR2L
from envs.success_reward_wrapper import SuccessRewardWrapper
from envs.action_smooth_wrapper import ActionSmoothWrapper
import realant_sim
import imageio
import argparse

def record_smooth_video(model_path, output_path="smooth_walking.mp4", use_smoothing=True, alpha=0.7):
    """Test model with optional action smoothing and record video"""
    
    # Load model
    try:
        model = PPO_SR2L.load(model_path)
        print("Loaded SR2L model")
    except:
        model = PPO.load(model_path)
        print("Loaded standard PPO model")
    
    # Create environment with rgb_array for recording
    env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')
    env = SuccessRewardWrapper(env)
    
    if use_smoothing:
        env = ActionSmoothWrapper(env, alpha=alpha)
        print(f"Action smoothing enabled (alpha={alpha})")
    
    # Record one episode
    obs, _ = env.reset()
    frames = []
    episode_reward = 0
    action_changes = []
    velocities = []
    prev_action = None
    
    print("Recording video...")
    for step in range(1000):  # Max 1000 steps
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Capture frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        episode_reward += reward
        
        # Track metrics
        if 'current_velocity' in info:
            velocities.append(info['current_velocity'])
        
        if prev_action is not None:
            change = np.linalg.norm(action - prev_action)
            action_changes.append(change)
        prev_action = action.copy()
        
        if terminated or truncated:
            break
    
    # Save video
    if frames:
        print(f"Saving video to {output_path}...")
        imageio.mimwrite(output_path, frames, fps=30)
        print(f"Video saved! ({len(frames)} frames)")
    
    # Print metrics
    print("\n" + "="*50)
    print("EPISODE METRICS:")
    print(f"Total Reward: {episode_reward:.2f}")
    print(f"Average Velocity: {np.mean(velocities) if velocities else 0:.3f} m/s")
    print(f"Max Velocity: {np.max(velocities) if velocities else 0:.3f} m/s")
    print(f"Average Action Change: {np.mean(action_changes) if action_changes else 0:.3f}")
    print(f"Max Action Change: {np.max(action_changes) if action_changes else 0:.3f}")
    print(f"Action Smoothness Score: {1.0 / (1.0 + np.mean(action_changes)) if action_changes else 0:.3f}")
    print("="*50)
    
    env.close()
    return episode_reward, np.mean(velocities) if velocities else 0

def compare_smoothing(model_path):
    """Compare with and without smoothing"""
    print("\n" + "="*60)
    print("TESTING WITHOUT SMOOTHING")
    print("="*60)
    reward_no_smooth, vel_no_smooth = record_smooth_video(
        model_path, 
        "no_smooth.mp4", 
        use_smoothing=False
    )
    
    print("\n" + "="*60)
    print("TESTING WITH SMOOTHING (alpha=0.8)")
    print("="*60)
    reward_smooth, vel_smooth = record_smooth_video(
        model_path, 
        "with_smooth.mp4", 
        use_smoothing=True, 
        alpha=0.8
    )
    
    print("\n" + "="*60)
    print("COMPARISON SUMMARY:")
    print("="*60)
    print(f"Without Smoothing: Reward={reward_no_smooth:.2f}, Velocity={vel_no_smooth:.3f} m/s")
    print(f"With Smoothing:    Reward={reward_smooth:.2f}, Velocity={vel_smooth:.3f} m/s")
    print(f"Improvement:       Reward={reward_smooth-reward_no_smooth:+.2f}, Velocity={vel_smooth-vel_no_smooth:+.3f} m/s")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--compare', action='store_true', help='Compare with and without smoothing')
    parser.add_argument('--output', type=str, default='smooth_walking.mp4', help='Output video path')
    parser.add_argument('--alpha', type=float, default=0.7, help='Smoothing factor (0-1)')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_smoothing(args.model)
    else:
        record_smooth_video(args.model, args.output, use_smoothing=True, alpha=args.alpha)