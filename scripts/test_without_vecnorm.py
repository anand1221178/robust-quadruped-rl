#!/usr/bin/env python3
"""
Test model WITHOUT VecNormalize to isolate the issue
"""

import sys
import os
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import realant_sim
from envs.target_walking_wrapper import TargetWalkingWrapper

def test_without_normalization():
    """Test baseline model without VecNormalize"""
    
    model_path = "experiments/ppo_target_walking_llsm451b/best_model/best_model.zip"
    
    print("Loading model...")
    model = PPO.load(model_path)
    
    print("Creating environment WITHOUT VecNormalize...")
    
    def make_env():
        env = gym.make('RealAntMujoco-v0')
        env = TargetWalkingWrapper(env, target_distance=5.0)
        return env
    
    env = DummyVecEnv([make_env])
    # NO VecNormalize here!
    
    print("Running test...")
    obs = env.reset()
    
    positions = []
    velocities = []
    
    for i in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # Get position
        actual_env = env.envs[0]
        x_pos = actual_env.unwrapped.data.qpos[0]
        z_pos = actual_env.unwrapped.data.qpos[2]
        positions.append(x_pos)
        
        if i > 0:
            vel = (positions[-1] - positions[-2]) / 0.05
            velocities.append(vel)
        
        if i % 40 == 0:
            avg_vel = np.mean(velocities[-20:]) if velocities else 0
            print(f"Step {i:3d}: X={x_pos:6.3f}, Z={z_pos:6.3f}, "
                  f"Vel={avg_vel:6.3f} m/s, Reward={reward[0]:6.2f}")
            
            # Check if robot fell
            if z_pos < 0.1:
                print("  ⚠️  Robot may have fallen!")
        
        if done[0]:
            print(f"Episode ended at step {i}")
            break
    
    final_velocity = np.mean(velocities) if velocities else 0
    print(f"\nFinal average velocity: {final_velocity:.3f} m/s")
    
    if final_velocity > 0.5:
        print("✅ Model works well WITHOUT VecNormalize - VecNormalize is the problem!")
    else:
        print("❌ Model is still slow - problem is deeper than VecNormalize")
    
    env.close()

if __name__ == "__main__":
    test_without_normalization()