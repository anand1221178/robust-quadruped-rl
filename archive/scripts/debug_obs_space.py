#!/usr/bin/env python3
"""
Debug script to understand RealAnt observation space
"""
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from envs.success_reward_wrapper import SuccessRewardWrapper
import realant_sim

def analyze_obs_space():
    """Analyze what's in RealAnt observation space"""
    
    # Create environment
    env = gym.make('RealAntMujoco-v0')
    env = SuccessRewardWrapper(env)
    
    print("RealAnt Observation Space Analysis")
    print("=" * 50)
    
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation space: {env.observation_space}")
    
    # Take a few steps and analyze
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"\nStep {step + 1}:")
        print(f"Observation (first 10): {obs[:10]}")
        print(f"Info keys: {list(info.keys())}")
        if 'current_velocity' in info:
            print(f"Wrapper velocity: {info['current_velocity']:.4f} m/s")
        
        # Check if velocity is in obs[:3]
        obs_velocity = np.linalg.norm(obs[:3])
        print(f"obs[:3] norm (what eval script uses): {obs_velocity:.4f}")
        
        # Check raw MuJoCo data
        if hasattr(env.unwrapped, 'data'):
            qvel = env.unwrapped.data.qvel
            body_vel = qvel[:3] if len(qvel) >= 3 else [0, 0, 0]
            actual_vel = np.linalg.norm(body_vel)
            print(f"MuJoCo body velocity: {actual_vel:.4f} m/s")
            print(f"MuJoCo qvel[:6]: {qvel[:6]}")
        
        if done or truncated:
            break
    
    env.close()

if __name__ == "__main__":
    analyze_obs_space()