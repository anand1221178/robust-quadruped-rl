#!/usr/bin/env python3
"""Quick debug script to see what's happening with baseline model"""
import sys
sys.path.append('../src')

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
import realant_sim

# Create environment
def make_env():
    base_env = gym.make('RealAntMujoco-v0', disable_env_checker=True)
    env = SuccessRewardWrapper(base_env)
    env = TimeLimit(env, max_episode_steps=1200)
    return env

env = DummyVecEnv([make_env])
env = VecNormalize.load('../done/ppo_baseline_ueqbjf2x/vec_normalize.pkl', env)
env.training = False
env.norm_reward = False

# Load model
model = PPO.load('../done/ppo_baseline_ueqbjf2x/best_model/best_model', env=env)

print("="*60)
print("DEBUGGING BASELINE MODEL")
print("="*60)

# Run 1 episode and track everything
obs = env.reset()
unwrapped = env.envs[0].unwrapped

print(f"\nInitial position: {unwrapped.data.qpos[:3]}")
print(f"Initial x: {unwrapped.data.qpos[0]:.6f}")

positions_x = [unwrapped.data.qpos[0]]
rewards = []

for step in range(1200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    rewards.append(reward[0])

    current_x = unwrapped.data.qpos[0]
    positions_x.append(current_x)

    if step % 200 == 0:
        print(f"Step {step:4d}: x={current_x:8.6f}, reward={reward[0]:8.2f}")

    if done:
        print(f"\nEpisode ended at step {step}")
        break

print(f"\nFinal position: {unwrapped.data.qpos[:3]}")
print(f"Final x: {unwrapped.data.qpos[0]:.6f}")
print(f"\nStart x: {positions_x[0]:.6f}")
print(f"End x:   {positions_x[-1]:.6f}")
print(f"Distance: {abs(positions_x[-1] - positions_x[0]):.6f} m")
print(f"Total reward: {sum(rewards):.2f}")
print(f"Steps completed: {len(positions_x)-1}")

env.close()
