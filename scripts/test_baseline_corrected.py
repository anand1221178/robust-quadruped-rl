#!/usr/bin/env python3
"""
Test baseline with CORRECTED VecNormalize settings
"""
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.smooth_target_wrapper import SmoothTargetWrapper
import realant_sim

def test_baseline_corrected():
    """Test baseline with proper VecNormalize settings"""
    
    print("🎯 TESTING BASELINE WITH CORRECTED SETTINGS")
    print("=" * 60)
    
    model_path = 'experiments/ppo_smooth_baseline_rohl32fn/best_model/best_model.zip'
    norm_path = 'experiments/ppo_smooth_baseline_rohl32fn/vec_normalize.pkl'
    
    # Create environment (same as training)
    def make_env():
        env = gym.make('RealAntMujoco-v0')
        env = SmoothTargetWrapper(env, target_distance=5.0)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Load VecNormalize with CORRECT settings
    env = VecNormalize.load(norm_path, env)
    env.training = False  # 🔧 CRITICAL FIX!
    env.norm_reward = False  # 🔧 Don't normalize rewards during evaluation
    
    print("✅ VecNormalize loaded with CORRECT settings:")
    print(f"  Training mode: {env.training}")
    print(f"  Reward normalization: {env.norm_reward}")
    
    # Load model
    model = PPO.load(model_path)
    print("✅ Model loaded")
    
    # Test multiple episodes
    print(f"\n🤖 TESTING GOAL-DIRECTED LOCOMOTION:")
    print("-" * 50)
    
    episode_results = []
    
    for episode in range(3):
        print(f"\n📍 EPISODE {episode + 1}:")
        
        obs = env.reset()
        episode_reward = 0
        targets_reached = 0
        positions = []
        velocities = []
        
        for step in range(1000):  # Full episode length
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0]
            
            # Track position
            if hasattr(env.envs[0], 'unwrapped'):
                current_x = env.envs[0].unwrapped.data.qpos[0]
                positions.append(current_x)
            
            # Track custom metrics
            if info[0]:
                if 'velocity' in info[0]:
                    velocities.append(info[0]['velocity'])
                if 'target_reached' in info[0] and info[0]['target_reached']:
                    targets_reached += 1
                    current_pos = positions[-1] if positions else 0
                    print(f"  🎯 TARGET {targets_reached} REACHED at x={current_pos:.1f}m (step {step})")
            
            # Progress updates
            if step % 200 == 0:
                current_pos = positions[-1] if positions else 0
                vel = velocities[-1] if velocities else 0
                print(f"  Step {step:3d}: reward={episode_reward:.0f}, x={current_pos:5.1f}m, vel={vel:.3f}m/s")
            
            if done[0]:
                print(f"  Episode ended at step {step}")
                break
        
        # Episode results
        total_distance = positions[-1] - positions[0] if len(positions) >= 2 else 0
        avg_velocity = np.mean(velocities) if velocities else 0
        
        episode_results.append({
            'reward': episode_reward,
            'targets': targets_reached,
            'distance': total_distance,
            'velocity': avg_velocity,
            'steps': step + 1
        })
        
        print(f"  📊 Episode {episode + 1}: {episode_reward:.0f} reward, {targets_reached} targets, {total_distance:.1f}m, {avg_velocity:.3f} m/s")
    
    # Overall results
    print(f"\n📈 CORRECTED RESULTS:")
    print("=" * 60)
    
    avg_reward = np.mean([r['reward'] for r in episode_results])
    avg_targets = np.mean([r['targets'] for r in episode_results])
    avg_distance = np.mean([r['distance'] for r in episode_results])
    avg_velocity = np.mean([r['velocity'] for r in episode_results])
    
    print(f"✅ Average Episode Reward: {avg_reward:.0f}")
    print(f"✅ Average Targets Reached: {avg_targets:.1f} per episode")
    print(f"✅ Average Distance Traveled: {avg_distance:.1f} meters")
    print(f"✅ Average Velocity: {avg_velocity:.3f} m/s")
    
    # Compare to W&B
    print(f"\n🔍 COMPARISON TO W&B:")
    print(f"  W&B training reward: ~4909")
    print(f"  Our corrected reward: ~{avg_reward:.0f}")
    if avg_reward > 3000:
        print(f"  🎉 MATCH! Rewards are in same ballpark!")
    elif avg_reward > 1000:
        print(f"  📈 CLOSE! Still some difference but much better")
    else:
        print(f"  ❌ STILL BROKEN: Rewards too low")
    
    # Success assessment
    print(f"\n🎯 PERFORMANCE ASSESSMENT:")
    if avg_targets >= 2.0:
        print("🎉 EXCELLENT: Robot consistently reaches multiple targets!")
        print("✅ SmoothTargetWrapper is WORKING!")
    elif avg_targets >= 1.0:
        print("✅ GOOD: Robot reaches targets regularly")
        print("✅ SmoothTargetWrapper is working")
    elif avg_targets >= 0.5:
        print("⚠️  PARTIAL: Robot reaches some targets")
    else:
        print("❌ POOR: Robot not reaching targets consistently")
    
    if avg_velocity > 0.2:
        print("🚀 GOOD: Healthy walking speed for goal-directed behavior")
    elif avg_velocity > 0.1:
        print("👍 DECENT: Reasonable walking speed")
    else:
        print("🐌 SLOW: Walking speed needs improvement")
    
    env.close()
    return episode_results

if __name__ == "__main__":
    results = test_baseline_corrected()
    
    if results and np.mean([r['targets'] for r in results]) >= 1.0:
        print(f"\n🎊 SUCCESS! SmoothTargetWrapper baseline is WORKING!")
        print(f"🎯 Ready for Phase 2: SR2L and DR training!")
    else:
        print(f"\n😞 Still need to debug further...")