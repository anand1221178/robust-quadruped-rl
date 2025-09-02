#!/usr/bin/env python3
"""
Simple test to see if baseline model works AT ALL
"""

import sys
import os
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import realant_sim

def test_baseline():
    """Minimal test - just load and run the baseline"""
    
    model_path = "experiments/ppo_target_walking_llsm451b/best_model/best_model.zip"
    
    print("Loading model...")
    model = PPO.load(model_path)
    
    # Create SIMPLEST possible environment - no wrappers
    print("Creating RAW environment (no wrappers)...")
    env = gym.make('RealAntMujoco-v0')
    env = DummyVecEnv([lambda: env])
    
    # Try with vecnorm
    vec_norm_path = "experiments/ppo_target_walking_llsm451b/vec_normalize.pkl"
    if os.path.exists(vec_norm_path):
        print(f"Loading VecNormalize from {vec_norm_path}")
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False
    
    print("\nRunning for 100 steps...")
    obs = env.reset()
    
    for i in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # Get robot position directly
        if hasattr(env, 'envs'):
            actual_env = env.envs[0]
        else:
            actual_env = env.env.envs[0]
        
        x_pos = actual_env.unwrapped.data.qpos[0]
        z_pos = actual_env.unwrapped.data.qpos[2]
        
        if i % 20 == 0:
            print(f"Step {i:3d}: X={x_pos:6.3f}, Z={z_pos:6.3f}, Reward={reward[0]:6.2f}, Action range=[{action.min():.2f}, {action.max():.2f}]")
        
        if done[0]:
            print(f"Episode ended at step {i}")
            break
    
    final_x = actual_env.unwrapped.data.qpos[0]
    print(f"\nFinal X position: {final_x:.3f}")
    print(f"Distance traveled: {final_x:.3f} meters")
    
    # Now test WITH TargetWalkingWrapper
    print("\n" + "="*60)
    print("Testing WITH TargetWalkingWrapper...")
    print("="*60)
    
    from envs.target_walking_wrapper import TargetWalkingWrapper
    
    env2 = gym.make('RealAntMujoco-v0')
    env2 = TargetWalkingWrapper(env2, target_distance=5.0)
    env2 = DummyVecEnv([lambda: env2])
    
    if os.path.exists(vec_norm_path):
        env2 = VecNormalize.load(vec_norm_path, env2)
        env2.training = False
        env2.norm_reward = False
    
    obs = env2.reset()
    
    for i in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env2.step(action)
        
        if i % 20 == 0:
            print(f"Step {i:3d}: Reward={reward[0]:6.2f}, Info keys: {list(info[0].keys()) if info[0] else 'None'}")
        
        if done[0]:
            print(f"Episode ended at step {i}")
            break
    
    env.close()
    env2.close()

if __name__ == "__main__":
    test_baseline()