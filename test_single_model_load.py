#!/usr/bin/env python3
"""Quick test: Load one multiseed model and evaluate it"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import realant_sim

# Test M1 seed42
model_dir = Path("experiments/M1_baseline_seed42_rap72nn2")
model_path = model_dir / "models/model.zip"
vec_norm_path = model_dir / "vec_normalize.pkl"

print("="*80)
print("TESTING M1 BASELINE SEED42 MODEL LOADING")
print("="*80)
print(f"\nModel path: {model_path}")
print(f"VecNormalize path: {vec_norm_path}")
print(f"Model exists: {model_path.exists()}")
print(f"VecNormalize exists: {vec_norm_path.exists()}")

# Create environment
env = gym.make('RealAntMujoco-v0')
env = DummyVecEnv([lambda: env])

# Load VecNormalize
if vec_norm_path.exists():
    env = VecNormalize.load(str(vec_norm_path), env)
    env.training = False
    env.norm_reward = False
    print("\n✅ VecNormalize loaded successfully")
else:
    print("\n❌ VecNormalize not found!")

# Load model
model = PPO.load(str(model_path), env=env)
print("✅ Model loaded successfully")

# Run ONE episode
print("\n" + "="*80)
print("RUNNING 1 EVALUATION EPISODE")
print("="*80)

obs = env.reset()
done = False
episode_reward = 0
episode_length = 0
positions = []

# Get initial position
try:
    unwrapped = env.envs[0].unwrapped
    start_x = unwrapped.data.qpos[0]
    print(f"\nStart position (x): {start_x:.3f}")
except:
    start_x = 0.0

while not done:
    action, _ = model.predict(obs, deterministic=True)

    try:
        positions.append(unwrapped.data.qpos[0])
    except:
        pass

    obs, reward, done, info = env.step(action)
    episode_reward += reward[0]
    episode_length += 1

    # Print first few steps
    if episode_length <= 5:
        print(f"  Step {episode_length}: reward={reward[0]:.4f}, cumulative={episode_reward:.2f}")

    if done[0]:
        break

final_x = positions[-1] if positions else start_x
distance = abs(final_x - start_x)

print(f"\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Episode Length: {episode_length} steps")
print(f"Total Reward: {episode_reward:.2f}")
print(f"Distance Traveled: {distance:.3f} m")
print(f"Final x position: {final_x:.3f}")
print()

# Compare to expected values
print("="*80)
print("EXPECTED vs ACTUAL")
print("="*80)
print(f"Expected reward: ~10,000 - 300,000")
print(f"Actual reward:   {episode_reward:.2f}")
print()
print(f"Expected distance: ~6-8m")
print(f"Actual distance:   {distance:.3f}m")
print()

if episode_reward < 100:
    print("❌ ISSUE DETECTED: Reward way too low!")
    print("   Possible causes:")
    print("   - Model not loading correctly")
    print("   - VecNormalize stats incorrect")
    print("   - Episode terminating too early (robot falling)")
elif episode_length < 50:
    print("❌ ISSUE DETECTED: Episode too short!")
    print("   Robot is probably falling immediately")
else:
    print("✅ Results look reasonable!")
