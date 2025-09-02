#!/usr/bin/env python3
"""
Recreate the original smooth video setup - CORRECTED VERSION
"""

import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
from envs.action_smooth_wrapper import ActionSmoothWrapper
import realant_sim
import imageio

def record_smooth_video(model_path, output_path="smooth_test.mp4", alpha=0.7, use_vecnorm=False):
    """Test model with action smoothing - FORCE PPO loading"""
    
    print(f"Loading REGULAR PPO model from {model_path}")
    model = PPO.load(model_path)  # Force regular PPO loading
    
    print("Creating environment setup...")
    
    def make_env():
        env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')
        env = SuccessRewardWrapper(env)
        env = ActionSmoothWrapper(env, alpha=alpha)
        return env
    
    if use_vecnorm:
        # Test with VecEnv + VecNormalize
        from stable_baselines3.common.vec_env import VecNormalize
        env = DummyVecEnv([make_env])
        vec_norm_path = "experiments/ppo_target_walking_llsm451b/vec_normalize.pkl"
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False
        print("Using VecNormalize")
        
        obs = env.reset()
    else:
        # Test without VecEnv (direct environment)
        env = make_env()
        print("Using direct environment (no VecNormalize)")
        obs, _ = env.reset()
    
    print(f"Action smoothing enabled (alpha={alpha})")
    
    # Record episode
    frames = []
    episode_reward = 0
    velocities = []
    
    print("Recording video...")
    for step in range(1000):
        if use_vecnorm:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            # Get velocity
            if info[0] and 'current_velocity' in info[0]:
                velocities.append(info[0]['current_velocity'])
        else:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            # Get velocity
            if 'current_velocity' in info:
                velocities.append(info['current_velocity'])
        
        # Capture frame
        if use_vecnorm:
            frame = env.render(mode='rgb_array')
        else:
            frame = env.render()
        
        if frame is not None:
            frames.append(frame)
        
        if step % 100 == 0:
            avg_vel = np.mean(velocities[-20:]) if len(velocities) >= 20 else (np.mean(velocities) if velocities else 0)
            print(f"Step {step}: Reward={episode_reward:.1f}, Recent Vel={avg_vel:.3f} m/s")
        
        if done:
            print(f"Episode ended at step {step}")
            break
    
    # Save video
    if frames:
        print(f"Saving video to {output_path}...")
        imageio.mimsave(output_path, frames, fps=30)
        print(f"Video saved! ({len(frames)} frames)")
    
    # Print metrics
    avg_vel = np.mean(velocities) if velocities else 0
    print("\n" + "="*50)
    print("EPISODE METRICS:")
    print(f"Total Reward: {episode_reward:.2f}")
    print(f"Average Velocity: {avg_vel:.3f} m/s")
    print(f"Max Velocity: {max(velocities) if velocities else 0:.3f} m/s")
    print(f"Steps completed: {step + 1}")
    print("="*50)
    
    env.close()
    return episode_reward, avg_vel

if __name__ == "__main__":
    model_path = "experiments/ppo_target_walking_llsm451b/best_model/best_model.zip"
    
    print("TESTING SETUP 1: Direct environment (like original)")
    reward1, vel1 = record_smooth_video(
        model_path, 
        "smooth_test_direct.mp4", 
        alpha=0.7, 
        use_vecnorm=False
    )
    
    print("\n" + "="*60)
    print("TESTING SETUP 2: With VecNormalize")
    reward2, vel2 = record_smooth_video(
        model_path, 
        "smooth_test_vecnorm.mp4", 
        alpha=0.7, 
        use_vecnorm=True
    )
    
    print("\n" + "="*60)
    print("COMPARISON:")
    print(f"Direct env:     Reward={reward1:7.2f}, Velocity={vel1:.3f} m/s")
    print(f"With VecNorm:   Reward={reward2:7.2f}, Velocity={vel2:.3f} m/s")
    print("="*60)