#!/usr/bin/env python3
"""
⚡ QUICK VELOCITY TEST ⚡
Test current model velocity performance in under 30 seconds!
"""
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
import realant_sim
import os
import argparse

def quick_velocity_test(model_path, steps=200):
    """🏃 Quick velocity test - get results FAST!"""
    
    print("⚡" + "=" * 50 + "⚡")
    print("   🏃 QUICK VELOCITY TEST 🏃")
    print("⚡" + "=" * 50 + "⚡")
    print(f"📍 Model: {model_path}")
    print(f"🎯 Test steps: {steps}")
    
    # Determine model type from path
    if "sr2l" in model_path.lower():
        model_type = "SR2L"
    elif "curriculum" in model_path.lower():
        model_type = "Curriculum DR"
    else:
        model_type = "Standard PPO"
    
    print(f"🔬 Type: {model_type}")
    print("-" * 52)
    
    # Setup environment (simple, no fancy wrappers)
    try:
        env = gym.make('RealAntMujoco-v0')
        env = SuccessRewardWrapper(env)
        print(" Environment created")
        
        # Load VecNormalize if it exists
        norm_path = model_path.replace('.zip', '_vecnormalize.pkl').replace('final_model', 'vec_normalize').replace('best_model', 'vec_normalize')
        if not os.path.exists(norm_path):
            # Try alternative paths
            model_dir = os.path.dirname(model_path)
            norm_path = os.path.join(model_dir, 'vec_normalize.pkl')
        
        if os.path.exists(norm_path):
            from stable_baselines3.common.vec_env import DummyVecEnv
            env = DummyVecEnv([lambda: env])
            env = VecNormalize.load(norm_path, env)
            env.training = False
            env.norm_reward = False
            print(" VecNormalize loaded")
        else:
            print("⚠️  No VecNormalize found - using raw environment")
            from stable_baselines3.common.vec_env import DummyVecEnv
            env = DummyVecEnv([lambda: env])
        
        # Load model
        model = PPO.load(model_path)
        print(" Model loaded")
        
        # Quick test
        print("\n🏃 Running velocity test...")
        obs = env.reset()
        positions = []
        
        for step in range(steps):
            if hasattr(env, 'get_original_obs'):
                # VecEnv
                original_obs = env.get_original_obs()
                if len(original_obs) > 0:
                    pos_x = original_obs[0][0] if len(original_obs[0]) > 0 else 0
                else:
                    pos_x = 0
            else:
                # Regular env
                pos_x = obs[0][0] if hasattr(obs, '__len__') and len(obs) > 0 else 0
            
            positions.append(pos_x)
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Progress indicator
            if step % 50 == 0:
                print(f"  Step {step:3d}: x={pos_x:5.2f}m")
            
            if done.any() if hasattr(done, 'any') else done:
                break
        
        # Calculate velocity
        if len(positions) >= 2:
            distance = positions[-1] - positions[0]
            time_taken = len(positions) * 0.05  # 20 FPS = 0.05s per step
            velocity = distance / time_taken
            
            print(f"\n🎯 RESULTS:")
            print(f"📏 Distance: {distance:.3f}m")
            print(f"⏱️  Time: {time_taken:.2f}s")
            print(f"⚡ Velocity: {velocity:.3f} m/s")
            
            # Compare to baselines
            baseline_vel = 0.224  # Known baseline
            sr2l_vel = 0.181     # Known SR2L
            
            if velocity > 0.20:
                status = "🔥 EXCELLENT"
            elif velocity > 0.15:
                status = " GOOD"
            elif velocity > 0.10:
                status = "⚠️ MODERATE"
            else:
                status = "❌ POOR"
                
            print(f"📊 Status: {status}")
            print(f"📈 vs Baseline: {(velocity/baseline_vel*100):.1f}%")
            print(f"📈 vs SR2L: {(velocity/sr2l_vel*100):.1f}%")
            
            return velocity
        else:
            print("❌ Failed to collect position data")
            return 0.0
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 0.0

def main():
    parser = argparse.ArgumentParser(description='Quick velocity test for any model')
    parser.add_argument('model_path', help='Path to model (.zip file)')
    parser.add_argument('--steps', type=int, default=200, help='Number of test steps (default: 200)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        sys.exit(1)
    
    velocity = quick_velocity_test(args.model_path, args.steps)
    
    print("\n" + "⚡" * 52)
    print(f"🏁 FINAL RESULT: {velocity:.3f} m/s")
    print("⚡" * 52)

if __name__ == "__main__":
    main()