#!/usr/bin/env python3
"""
Test script for SmoothTargetWrapper - Does it actually work?
"""
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.smooth_target_wrapper import SmoothTargetWrapper
import realant_sim

def test_smooth_target_wrapper():
    """Test the SmoothTargetWrapper with baseline model"""
    
    print("🧪 TESTING SMOOTH TARGET WRAPPER")
    print("=" * 50)
    
    # Create environment with SmoothTargetWrapper
    def make_env():
        env = gym.make('RealAntMujoco-v0')
        env = SmoothTargetWrapper(env, target_distance=5.0)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Load baseline model and VecNormalize
    model_path = 'done/ppo_baseline_ueqbjf2x/best_model/best_model.zip'
    norm_path = 'done/ppo_baseline_ueqbjf2x/vec_normalize.pkl'
    
    # Note: VecNormalize expects original observation space, this might break!
    # We'll need to handle this properly for training
    try:
        env = VecNormalize.load(norm_path, env)
        env.training = False
        print("✅ VecNormalize loaded successfully")
    except Exception as e:
        print(f"⚠️  VecNormalize failed (expected - obs space mismatch): {e}")
        print("   This is expected - we need to retrain with new obs space")
    
    model = PPO.load(model_path)
    print(f"✅ Model loaded from: {model_path}")
    
    # Test run
    print("\n🤖 TESTING SMOOTH TARGET WALKING:")
    print("-" * 40)
    
    obs = env.reset()
    episode_rewards = []
    smoothness_scores = []
    targets_reached = 0
    
    for step in range(100):  # Short test
        try:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            episode_rewards.append(reward[0])
            
            # Extract custom metrics
            if info[0]:
                if 'smoothness_score' in info[0]:
                    smoothness_scores.append(info[0]['smoothness_score'])
                if 'target_reached' in info[0] and info[0]['target_reached']:
                    targets_reached += 1
                    print(f"🎯 Target reached at step {step}!")
                
                # Print progress every 20 steps
                if step % 20 == 0 and 'distance_to_target' in info[0]:
                    dist = info[0]['distance_to_target']
                    vel = info[0].get('velocity', 0)
                    smooth = info[0].get('smoothness_score', 0)
                    print(f"  Step {step:3d}: Distance={dist:.2f}m, Vel={vel:.3f}m/s, Smoothness={smooth:.2f}")
            
            if done[0]:
                print(f"Episode ended at step {step}")
                break
                
        except Exception as e:
            print(f"❌ Error at step {step}: {e}")
            print("   This might be due to observation space mismatch")
            break
    
    # Results
    print(f"\n📊 TEST RESULTS:")
    print(f"✅ Steps completed: {step + 1}")
    print(f"✅ Targets reached: {targets_reached}")
    print(f"✅ Average reward: {np.mean(episode_rewards):.3f}")
    if smoothness_scores:
        print(f"✅ Average smoothness: {np.mean(smoothness_scores):.3f}")
    
    print(f"\n🔍 WRAPPER ANALYSIS:")
    print(f"   Original obs space: {env.venv.envs[0].env.observation_space}")
    print(f"   Wrapper obs space: {env.observation_space}")
    
    env.close()

if __name__ == "__main__":
    test_smooth_target_wrapper()