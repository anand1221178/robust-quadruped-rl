#!/usr/bin/env python3
"""
Quick test to see robot movement patterns
"""

import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
import realant_sim

def quick_movement_test(model_path, model_name, steps=200):
    """Quick test of movement pattern"""
    
    print(f"\n🤖 Testing {model_name}")
    print("-" * 40)
    
    # Load model and environment
    model = PPO.load(model_path)
    
    def make_env():
        env = gym.make('RealAntMujoco-v0')
        env = SuccessRewardWrapper(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Load normalization
    norm_path = model_path.replace('/best_model.zip', '/../vec_normalize.pkl')
    try:
        env = VecNormalize.load(norm_path, env)
        env.training = False
        env.norm_reward = False
    except:
        pass
    
    # Run episode
    obs = env.reset()
    start_pos = None
    end_pos = None
    velocities = []
    
    for step in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # Track velocity from info
        vel = 0.0
        if info[0] and 'current_velocity' in info[0]:
            vel = info[0]['current_velocity']
        elif info[0] and 'speed' in info[0]:
            vel = info[0]['speed']
        velocities.append(vel)
        
        # Track position
        try:
            base_env = env.venv.envs[0]
            if hasattr(base_env, 'sim') and base_env.sim is not None:
                pos = base_env.sim.data.qpos[:3].copy()
                if start_pos is None:
                    start_pos = pos
                end_pos = pos
        except:
            pass
        
        if done[0]:
            break
    
    # Calculate metrics
    if start_pos is not None and end_pos is not None:
        displacement = end_pos - start_pos
        straight_distance = displacement[0]  # x-axis displacement
        lateral_distance = abs(displacement[1])  # y-axis displacement
        total_displacement = np.linalg.norm(displacement[:2])
        
        time_elapsed = len(velocities) * 0.05
        straight_velocity = straight_distance / time_elapsed
        info_velocity = np.mean(velocities) if velocities else 0
        
        print(f"Start position: ({start_pos[0]:.2f}, {start_pos[1]:.2f})")
        print(f"End position: ({end_pos[0]:.2f}, {end_pos[1]:.2f})")
        print(f"Forward displacement: {straight_distance:.2f}m")
        print(f"Lateral displacement: {lateral_distance:.2f}m")
        print(f"Straight velocity: {straight_velocity:.3f} m/s")
        print(f"Info velocity: {info_velocity:.3f} m/s")
        
        if lateral_distance > 1.0:
            print("⚠️ High lateral movement - possible wandering")
        else:
            print("✅ Low lateral movement - mostly straight")
            
        if abs(straight_velocity - info_velocity) > 0.05:
            print(f"⚠️ Velocity mismatch: {abs(straight_velocity - info_velocity):.3f} m/s difference")
        else:
            print("✅ Velocity measurements consistent")
    
    env.close()
    return straight_velocity if 'straight_velocity' in locals() else 0

# Test both models
print("🎯 QUICK MOVEMENT PATTERN TEST")
print("=" * 50)

models = {
    'Baseline': 'done/ppo_baseline_ueqbjf2x/best_model/best_model.zip',
    'DR v2': 'done/ppo_dr_gentle_v2_wptws01u/best_model/best_model.zip'
}

results = {}
for name, path in models.items():
    try:
        vel = quick_movement_test(path, name, steps=300)
        results[name] = vel
    except Exception as e:
        print(f"❌ Error testing {name}: {e}")

print("\n" + "=" * 50)
print("📊 SUMMARY")
print("=" * 50)
for name, vel in results.items():
    print(f"{name}: {vel:.3f} m/s straight-line velocity")

print("\n💡 VERDICT:")
if any(abs(v) < 0.1 for v in results.values()):
    print("❌ Very low velocities detected - investigate further")
elif any(v < 0 for v in results.values()):  
    print("❌ Negative velocities - robot walking backwards!")
else:
    print("✅ Reasonable forward velocities detected")
    print("   → Circling may not be the main issue")
    print("   → Proceed with permanent DR without straight-line constraint")