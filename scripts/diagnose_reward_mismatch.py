#!/usr/bin/env python3
"""
Diagnose the massive reward scale mismatch
W&B shows 4909, our test shows 10.1 - 500x difference!
"""
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.smooth_target_wrapper import SmoothTargetWrapper
import realant_sim

def diagnose_reward_scaling():
    """Investigate reward scaling issues"""
    
    print("🔍 DIAGNOSING REWARD SCALE MISMATCH")
    print("W&B Training: ~4909 reward")
    print("Our Test: ~10.1 reward") 
    print("=" * 60)
    
    model_path = 'experiments/ppo_smooth_baseline_rohl32fn/best_model/best_model.zip'
    norm_path = 'experiments/ppo_smooth_baseline_rohl32fn/vec_normalize.pkl'
    
    # Test 1: Full episode length (1000 steps like training)
    print("\n📏 TEST 1: FULL EPISODE LENGTH (1000 steps)")
    print("-" * 50)
    
    def make_env():
        env = gym.make('RealAntMujoco-v0')
        env = SmoothTargetWrapper(env, target_distance=5.0)
        return env
    
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(norm_path, env)
    env.training = False
    
    model = PPO.load(model_path)
    
    obs = env.reset()
    total_reward = 0
    raw_rewards = []
    
    for step in range(1000):  # Full episode like training
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        total_reward += reward[0]
        raw_rewards.append(reward[0])
        
        if step % 200 == 0:
            print(f"  Step {step}: cumulative_reward={total_reward:.1f}")
        
        if done[0]:
            print(f"  Episode ended early at step {step}")
            break
    
    print(f"📊 Full Episode Results:")
    print(f"  Total reward: {total_reward:.1f}")
    print(f"  Average reward per step: {np.mean(raw_rewards):.4f}")
    print(f"  Reward std: {np.std(raw_rewards):.4f}")
    
    env.close()
    
    # Test 2: Check VecNormalize reward scaling
    print(f"\n🔧 TEST 2: VECNORMALIZE INSPECTION")
    print("-" * 50)
    
    env = DummyVecEnv([make_env])
    vec_norm = VecNormalize.load(norm_path, env)
    
    print(f"  Reward normalization: {vec_norm.norm_reward}")
    print(f"  Training mode: {vec_norm.training}")
    
    if hasattr(vec_norm, 'ret_rms'):
        print(f"  Reward mean: {vec_norm.ret_rms.mean}")
        print(f"  Reward std: {np.sqrt(vec_norm.ret_rms.var)}")
    
    # Test with and without reward normalization
    print(f"\n🔧 TEST 3: WITH vs WITHOUT REWARD NORMALIZATION")
    print("-" * 50)
    
    # Test WITH normalization
    vec_norm.training = False
    obs = vec_norm.reset()
    total_norm = 0
    
    for step in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_norm.step(action)
        total_norm += reward[0]
    
    print(f"  WITH normalization (100 steps): {total_norm:.3f}")
    
    vec_norm.close()
    
    # Test WITHOUT normalization
    raw_env = DummyVecEnv([make_env])
    obs = raw_env.reset()
    total_raw = 0
    
    for step in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = raw_env.step(action)
        total_raw += reward[0]
    
    print(f"  WITHOUT normalization (100 steps): {total_raw:.3f}")
    print(f"  Ratio (raw/norm): {total_raw/total_norm if total_norm != 0 else 'inf':.1f}x")
    
    raw_env.close()
    
    # Test 4: Check individual reward components
    print(f"\n🔧 TEST 4: REWARD COMPONENT BREAKDOWN")
    print("-" * 50)
    
    env = DummyVecEnv([make_env])
    obs = env.reset()
    
    print("  Individual step rewards:")
    for step in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        if info[0]:
            progress = info[0].get('progress', 'N/A')
            velocity = info[0].get('velocity', 'N/A')
            smoothness = info[0].get('smoothness_score', 'N/A')
            custom_reward = info[0].get('custom_reward', 'N/A')
            
            print(f"    Step {step}: total_reward={reward[0]:.3f}, progress={progress}, velocity={velocity}, smoothness={smoothness}")
        else:
            print(f"    Step {step}: reward={reward[0]:.3f} (no info)")
    
    env.close()
    
    print(f"\n📊 DIAGNOSIS SUMMARY:")
    print(f"  If 1000-step episode ≈ 4909 → SmoothTargetWrapper working in training")
    print(f"  If reward normalization ratio is ~500x → VecNormalize issue")
    print(f"  If individual rewards are tiny → Wrapper reward scale wrong")

if __name__ == "__main__":
    diagnose_reward_scaling()