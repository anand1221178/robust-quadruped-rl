#!/usr/bin/env python3
"""
Test trained models with action smoothing wrapper
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
import argparse

def test_with_smoothing(model_path, use_smoothing=True, alpha=0.7):
    """Test model with optional action smoothing"""
    
    # Load model
    try:
        model = PPO_SR2L.load(model_path)
        print("Loaded SR2L model")
    except:
        model = PPO.load(model_path)
        print("Loaded standard PPO model")
    
    # Create environment (without render for now due to MuJoCo issue)
    env = gym.make('RealAntMujoco-v0')  # Remove render_mode to avoid error
    env = SuccessRewardWrapper(env)
    
    if use_smoothing:
        env = ActionSmoothWrapper(env, alpha=alpha)
        print(f"Action smoothing enabled (alpha={alpha})")
    
    # Test for multiple episodes
    num_episodes = 5
    episode_rewards = []
    action_changes = []
    velocities = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_actions = []
        episode_velocities = []
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_actions.append(action)
            
            if 'current_velocity' in info:
                episode_velocities.append(info['current_velocity'])
        
        # Calculate action smoothness
        if len(episode_actions) > 1:
            changes = []
            for i in range(1, len(episode_actions)):
                change = np.linalg.norm(episode_actions[i] - episode_actions[i-1])
                changes.append(change)
            action_changes.append(np.mean(changes))
        
        episode_rewards.append(episode_reward)
        velocities.append(np.mean(episode_velocities) if episode_velocities else 0)
        
        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
              f"Avg Velocity={velocities[-1]:.3f} m/s, "
              f"Max Velocity={np.max(episode_velocities) if episode_velocities else 0:.3f} m/s, "
              f"Avg Action Change={action_changes[-1] if action_changes else 0:.3f}")
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Average Velocity: {np.mean(velocities):.3f} m/s")
    print(f"Average Action Change: {np.mean(action_changes) if action_changes else 0:.3f}")
    print("="*50)
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--no-smooth', action='store_true', help='Disable action smoothing')
    parser.add_argument('--alpha', type=float, default=0.7, help='Smoothing factor (0-1)')
    
    args = parser.parse_args()
    
    test_with_smoothing(args.model, use_smoothing=not args.no_smooth, alpha=args.alpha)