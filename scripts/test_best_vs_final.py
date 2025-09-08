#!/usr/bin/env python3
"""
Test BEST model vs FINAL model - find out which actually works
"""
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.smooth_target_wrapper import SmoothTargetWrapper
import realant_sim

def test_model(model_name, model_path):
    """Test a specific model"""
    
    print(f"\n🧪 TESTING {model_name}:")
    print("-" * 50)
    
    norm_path = 'experiments/ppo_smooth_baseline_rohl32fn/vec_normalize.pkl'
    
    # Create environment
    def make_env():
        env = gym.make('RealAntMujoco-v0')
        env = SmoothTargetWrapper(env, target_distance=5.0)
        return env
    
    env = DummyVecEnv([make_env])
    
    try:
        # Load VecNormalize and model
        env = VecNormalize.load(norm_path, env)
        env.training = False
        
        model = PPO.load(model_path)
        print(f"✅ {model_name} loaded successfully")
        
        # Quick test
        obs = env.reset()
        total_reward = 0
        targets_reached = 0
        positions = []
        
        for step in range(200):  # 200 steps = 10 seconds
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            total_reward += reward[0]
            
            # Track position
            current_x = env.envs[0].unwrapped.data.qpos[0]
            positions.append(current_x)
            
            # Check for targets
            if info[0] and 'target_reached' in info[0] and info[0]['target_reached']:
                targets_reached += 1
                print(f"  🎯 Target {targets_reached} reached at step {step}!")
            
            if step % 50 == 0:
                print(f"  Step {step}: reward={total_reward:.1f}, x={current_x:.2f}")
            
            if done[0]:
                break
        
        # Results
        distance = positions[-1] - positions[0] if len(positions) >= 2 else 0
        velocity = distance / (len(positions) * 0.05)
        
        print(f"\n📊 {model_name} RESULTS:")
        print(f"  Total reward: {total_reward:.1f}")
        print(f"  Targets reached: {targets_reached}")
        print(f"  Distance: {distance:.2f}m")
        print(f"  Velocity: {velocity:.3f} m/s")
        
        # Assessment
        if targets_reached > 0:
            print(f"  ✅ WORKING - Robot reaches targets!")
        elif velocity > 0.1:
            print(f"  ⚠️  PARTIAL - Robot moves but no targets")
        else:
            print(f"  ❌ BROKEN - Robot barely moves")
        
        env.close()
        return {
            'reward': total_reward,
            'targets': targets_reached,
            'velocity': velocity,
            'distance': distance
        }
        
    except Exception as e:
        print(f"❌ {model_name} failed: {e}")
        env.close()
        return None

def compare_models():
    """Compare best vs final model"""
    
    print("🔍 COMPARING BEST VS FINAL MODEL")
    print("=" * 60)
    
    models = [
        ("BEST MODEL", "experiments/ppo_smooth_baseline_rohl32fn/best_model/best_model.zip"),
        ("FINAL MODEL", "experiments/ppo_smooth_baseline_rohl32fn/final_model.zip")
    ]
    
    results = {}
    
    for name, path in models:
        results[name] = test_model(name, path)
    
    # Comparison
    print(f"\n🏆 COMPARISON:")
    print("=" * 60)
    
    if results["BEST MODEL"] and results["FINAL MODEL"]:
        best = results["BEST MODEL"]
        final = results["FINAL MODEL"]
        
        print(f"REWARD:    Best={best['reward']:.1f}  vs  Final={final['reward']:.1f}")
        print(f"TARGETS:   Best={best['targets']}      vs  Final={final['targets']}")
        print(f"VELOCITY:  Best={best['velocity']:.3f}   vs  Final={final['velocity']:.3f}")
        print(f"DISTANCE:  Best={best['distance']:.2f}    vs  Final={final['distance']:.2f}")
        
        if best['targets'] > final['targets']:
            print(f"\n🎯 WINNER: BEST MODEL is better!")
        elif final['targets'] > best['targets']:
            print(f"\n🎯 WINNER: FINAL MODEL is better!")
        else:
            print(f"\n🤷 TIE: Both models equally bad/good")
    
    else:
        print("❌ Could not compare - one or both models failed to load")

if __name__ == "__main__":
    compare_models()