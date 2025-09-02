#!/usr/bin/env python3
"""
Debug script to understand target placement issue
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import environments
import realant_sim
from envs.target_walking_wrapper import TargetWalkingWrapper

def debug_target_placement():
    """Debug target placement and robot initial position"""
    
    # Create environment with explicit target distance
    print("Creating environment with target_distance=5.0...")
    env = gym.make('RealAntMujoco-v0')
    
    # Check initial robot position BEFORE wrapper
    env.reset()
    initial_x_unwrapped = env.unwrapped.data.qpos[0]
    print(f"Initial X position (unwrapped): {initial_x_unwrapped:.4f}")
    
    # Now wrap it
    env = TargetWalkingWrapper(env, target_distance=5.0)
    
    # Reset and check
    obs, info = env.reset()
    
    # Check wrapper's understanding
    current_x = env.env.unwrapped.data.qpos[0]
    print(f"\nAfter wrapper reset:")
    print(f"  Current X position: {current_x:.4f}")
    print(f"  Target distance setting: {env.target_distance}")
    print(f"  Target X position: {env.target_x:.4f}")
    print(f"  Expected target (current + 5.0): {current_x + 5.0:.4f}")
    
    # Simulate reaching target
    print(f"\nSimulating target reach...")
    
    # Manually set robot position near target (for testing)
    old_target = env.target_x
    
    # Take a few steps to see what happens
    for i in range(5):
        action = np.zeros(8)  # No movement
        obs, reward, terminated, truncated, info = env.step(action)
        current_x = env.env.unwrapped.data.qpos[0]
        print(f"  Step {i}: X={current_x:.4f}, Target={env.target_x:.4f}, Distance={abs(env.target_x - current_x):.4f}")
        
        # Check if info has the right data
        if 'distance_to_target' in info:
            print(f"    Info distance: {info['distance_to_target']:.4f}")
    
    env.close()
    
    print("\n" + "="*60)
    print("DIAGNOSIS:")
    print("="*60)
    
    if abs(initial_x_unwrapped) > 1.0:
        print("❌ Robot starts at weird X position (should be near 0)")
    else:
        print("✅ Robot starts at reasonable X position")
    
    if env.target_x - current_x < 4.0:
        print("❌ Target is too close (should be 5m away)")
    else:
        print("✅ Target is properly placed")
    
    # Now test with a model to see runtime behavior
    print("\n" + "="*60)
    print("TESTING WITH ACTUAL MODEL:")
    print("="*60)
    
    model_path = "experiments/ppo_target_walking_llsm451b/best_model/best_model.zip"
    if os.path.exists(model_path):
        model = PPO.load(model_path)
        
        # Create fresh environment
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
            print("Loaded VecNormalize")
        
        obs = env.reset()
        
        # Get initial position through vecenv
        wrapped_env = env.envs[0] if hasattr(env, 'envs') else env
        if hasattr(wrapped_env, 'env'):  # Get through VecNormalize
            wrapped_env = wrapped_env.env.envs[0]
        
        print(f"\nInitial setup with model:")
        print(f"  Target distance: {wrapped_env.target_distance}")
        print(f"  Target X: {wrapped_env.target_x:.4f}")
        
        # Run for a bit
        for i in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            if i % 20 == 0:
                print(f"  Step {i}: Info keys: {info[0].keys() if info[0] else 'None'}")
        
        env.close()

if __name__ == "__main__":
    debug_target_placement()