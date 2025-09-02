#!/usr/bin/env python3
"""
Test baseline with absolutely minimal setup - no wrappers at all
"""

import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import realant_sim
import imageio

def test_minimal(model_path, output_path, use_vecnorm=True):
    """Test with minimal setup"""
    
    print(f"Testing minimal setup (VecNorm: {use_vecnorm})")
    model = PPO.load(model_path)
    
    if use_vecnorm:
        # Minimal with VecNormalize (how it was trained)
        def make_env():
            return gym.make('RealAntMujoco-v0', render_mode='rgb_array')
        
        env = DummyVecEnv([make_env])
        vec_norm_path = "experiments/ppo_target_walking_llsm451b/vec_normalize.pkl"
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False
        obs = env.reset()
        
        print("Raw RealAnt + VecNormalize only")
    else:
        # Absolutely minimal - just raw environment
        env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')
        obs, _ = env.reset()
        
        print("Pure raw RealAnt environment")
    
    frames = []
    positions = []
    rewards = []
    
    for step in range(500):  # Shorter test
        action, _ = model.predict(obs, deterministic=True)
        
        if use_vecnorm:
            obs, reward, done, info = env.step(action)
            reward = reward[0]
            done = done[0]
            # Get position
            x_pos = env.envs[0].unwrapped.data.qpos[0]
        else:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # Get position
            x_pos = env.unwrapped.data.qpos[0]
        
        positions.append(x_pos)
        rewards.append(reward)
        
        # Capture frame
        if use_vecnorm:
            frame = env.render(mode='rgb_array')
        else:
            frame = env.render()
        
        if frame is not None:
            frames.append(frame)
        
        if step % 50 == 0:
            distance = positions[-1] - positions[0] if len(positions) > 1 else 0
            velocity = (positions[-1] - positions[-10]) / (10 * 0.05) if len(positions) > 10 else 0
            print(f"Step {step:3d}: X={x_pos:.3f}, Dist={distance:.3f}, Vel={velocity:.3f}, Reward={reward:.2f}")
        
        if done:
            print(f"Episode ended at step {step}")
            break
    
    # Save video
    if frames:
        imageio.mimsave(output_path, frames, fps=30)
        print(f"Saved {output_path}")
    
    # Results
    final_distance = positions[-1] - positions[0] if len(positions) > 1 else 0
    avg_velocity = final_distance / (len(positions) * 0.05) if positions else 0
    total_reward = sum(rewards)
    
    print(f"Final: Distance={final_distance:.3f}m, AvgVel={avg_velocity:.3f}m/s, TotalReward={total_reward:.1f}")
    env.close()
    
    return avg_velocity, total_reward

if __name__ == "__main__":
    model_path = "experiments/ppo_target_walking_llsm451b/best_model/best_model.zip"
    
    print("="*60)
    print("TEST 1: Raw environment (no normalization)")
    print("="*60)
    vel1, rew1 = test_minimal(model_path, "minimal_raw.mp4", use_vecnorm=False)
    
    print("\n" + "="*60)
    print("TEST 2: With VecNormalize (training setup)")
    print("="*60)
    vel2, rew2 = test_minimal(model_path, "minimal_vecnorm.mp4", use_vecnorm=True)
    
    print("\n" + "="*60)
    print("COMPARISON:")
    print(f"Raw env:      Vel={vel1:.3f} m/s, Reward={rew1:.1f}")
    print(f"VecNormalize: Vel={vel2:.3f} m/s, Reward={rew2:.1f}")
    
    if max(vel1, vel2) > 0.5:
        print("✅ Found working setup!")
    else:
        print("❌ Both setups show poor performance")
        print("The model may be corrupted or there's an environment mismatch")
    print("="*60)