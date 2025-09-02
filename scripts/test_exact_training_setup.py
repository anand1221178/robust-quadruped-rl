#!/usr/bin/env python3
"""
Test model with EXACT training setup from config
"""

import sys
import os
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import yaml
import realant_sim
from envs.target_walking_wrapper import TargetWalkingWrapper

def test_with_exact_config(model_dir):
    """Test model using its exact training configuration"""
    
    # Load the config used for training
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Config loaded from {config_path}")
    print(f"Environment config: {config.get('env', {})}")
    
    # Extract env settings
    env_name = config.get('env', {}).get('name', 'RealAntMujoco-v0')
    use_target_walking = config.get('env', {}).get('use_target_walking', False)
    target_distance = config.get('env', {}).get('target_distance', 5.0)
    
    print(f"\nEnvironment: {env_name}")
    print(f"Use target walking: {use_target_walking}")
    print(f"Target distance: {target_distance}")
    
    # Create environment EXACTLY as in training
    def make_env():
        env = gym.make(env_name)
        if use_target_walking:
            print("Applying TargetWalkingWrapper...")
            env = TargetWalkingWrapper(env, target_distance=target_distance)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Load VecNormalize if it exists
    vec_norm_path = os.path.join(model_dir, "vec_normalize.pkl")
    if os.path.exists(vec_norm_path):
        print(f"Loading VecNormalize from {vec_norm_path}")
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False
    
    # Load the model
    model_path = os.path.join(model_dir, "best_model", "best_model.zip")
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    
    # Test run
    print("\n" + "="*60)
    print("RUNNING TEST")
    print("="*60)
    
    obs = env.reset()
    
    positions = []
    velocities = []
    rewards = []
    
    for i in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        rewards.append(reward[0])
        
        # Get actual environment
        if hasattr(env, 'envs'):
            actual_env = env.envs[0]
        else:
            actual_env = env.env.envs[0]
        
        x_pos = actual_env.unwrapped.data.qpos[0]
        z_pos = actual_env.unwrapped.data.qpos[2]
        positions.append(x_pos)
        
        # Calculate velocity
        if i > 0:
            vel = (positions[-1] - positions[-2]) / 0.05
            velocities.append(vel)
        
        if i % 40 == 0:
            avg_vel = np.mean(velocities[-20:]) if velocities else 0
            print(f"Step {i:3d}: X={x_pos:6.3f}, Z={z_pos:6.3f}, "
                  f"Vel={avg_vel:6.3f} m/s, Reward={reward[0]:6.2f}, "
                  f"Action std={action.std():.3f}")
        
        if done[0]:
            print(f"Episode ended at step {i}")
            break
    
    # Final stats
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    
    final_x = positions[-1] if positions else 0
    avg_velocity = np.mean(velocities) if velocities else 0
    total_reward = sum(rewards)
    
    print(f"Distance traveled: {final_x:.3f} meters")
    print(f"Average velocity: {avg_velocity:.3f} m/s")
    print(f"Total reward: {total_reward:.2f}")
    
    if avg_velocity < 0.5:
        print("\n⚠️  WARNING: Model is walking very slowly!")
        print("Possible issues:")
        print("  1. VecNormalize stats might be wrong")
        print("  2. Model might be corrupted")
        print("  3. Environment dynamics might have changed")
    
    env.close()

if __name__ == "__main__":
    # Test baseline
    print("TESTING BASELINE MODEL")
    print("="*80)
    test_with_exact_config("experiments/ppo_target_walking_llsm451b")
    
    print("\n\n")
    
    # Test SR2L Resume
    print("TESTING SR2L RESUME MODEL")
    print("="*80)
    test_with_exact_config("experiments/ppo_sr2l_fixed_v2_resume_5pc1hmrr")