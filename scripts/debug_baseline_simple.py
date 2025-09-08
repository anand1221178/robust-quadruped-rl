#!/usr/bin/env python3
"""
Simple debug - test baseline without our wrapper complications
"""
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import realant_sim

def test_baseline_raw():
    """Test baseline with just RealAnt environment"""
    
    print("🔧 DEBUG: Testing baseline with RAW RealAnt environment")
    print("=" * 60)
    
    model_path = 'experiments/ppo_smooth_baseline_rohl32fn/final_model.zip'
    norm_path = 'experiments/ppo_smooth_baseline_rohl32fn/vec_normalize.pkl'
    
    # Try WITHOUT any wrapper first
    def make_env():
        env = gym.make('RealAntMujoco-v0')
        return env
    
    env = DummyVecEnv([make_env])
    
    print("📏 Raw environment observation space:", env.observation_space)
    
    # Try loading model without VecNormalize
    print("\n🤖 Loading model WITHOUT VecNormalize...")
    try:
        model = PPO.load(model_path)
        print("✅ Model loaded")
        
        obs = env.reset()
        print(f"📊 Raw obs shape: {obs.shape}")
        
        # Test one step
        action, _ = model.predict(obs)
        print(f"📊 Action shape: {action.shape}")
        print(f"📊 Action values: {action[0]}")
        
        obs, reward, done, info = env.step(action)
        print(f"📊 Step result: reward={reward[0]:.3f}, done={done[0]}")
        
        # Test position tracking
        current_x = env.envs[0].unwrapped.data.qpos[0]
        print(f"📊 Robot X position: {current_x:.3f}")
        
    except Exception as e:
        print(f"❌ Failed without VecNormalize: {e}")
    
    env.close()
    
    # Try WITH VecNormalize but raw env
    print(f"\n🔧 Testing WITH VecNormalize...")
    env = DummyVecEnv([make_env])
    
    try:
        env = VecNormalize.load(norm_path, env)
        env.training = False
        print("✅ VecNormalize loaded with raw env")
        
        obs = env.reset()
        print(f"📊 Normalized obs shape: {obs.shape}")
        print(f"📊 Obs range: [{obs.min():.3f}, {obs.max():.3f}]")
        
        # Test 10 steps
        total_reward = 0
        positions = []
        
        for step in range(10):
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            
            current_x = env.envs[0].unwrapped.data.qpos[0]
            positions.append(current_x)
            
            print(f"Step {step}: reward={reward[0]:.3f}, x={current_x:.3f}")
        
        print(f"\n📊 10-step results:")
        print(f"  Total reward: {total_reward:.3f}")
        print(f"  Distance moved: {positions[-1] - positions[0]:.3f}m")
        print(f"  Average velocity: {(positions[-1] - positions[0]) / (10 * 0.05):.3f} m/s")
        
    except Exception as e:
        print(f"❌ Failed with VecNormalize: {e}")
    
    env.close()

if __name__ == "__main__":
    test_baseline_raw()