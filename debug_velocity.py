#!/usr/bin/env python3
"""
Debug script to check actual velocities achieved by trained model
"""
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
import realant_sim

def test_model_velocity(model_path, norm_path=None):
    """Test what velocity a trained model actually achieves"""
    
    # Create environment
    def make_env():
        env = gym.make('RealAntMujoco-v0')
        env = SuccessRewardWrapper(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    if norm_path:
        env = VecNormalize.load(norm_path, env)
        env.training = False
    
    # Load model
    model = PPO.load(model_path)
    
    # Test for 5 episodes
    velocities = []
    distances = []
    
    for episode in range(5):
        obs = env.reset()
        episode_velocities = []
        total_distance = 0
        
        for step in range(500):  # Max episode length
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Get velocity from info
            if 'current_velocity' in info[0]:
                vel = info[0]['current_velocity']
                episode_velocities.append(vel)
            
            if 'distance_traveled' in info[0]:
                total_distance = info[0]['distance_traveled']
            
            if done[0]:
                break
        
        avg_velocity = np.mean(episode_velocities)
        velocities.append(avg_velocity)
        distances.append(total_distance)
        
        print(f"Episode {episode+1}: Avg velocity = {avg_velocity:.4f} m/s, Distance = {total_distance:.4f} m")
    
    overall_avg = np.mean(velocities)
    overall_std = np.std(velocities)
    
    print(f"\nOverall Results:")
    print(f"Average velocity: {overall_avg:.4f} ± {overall_std:.4f} m/s")
    print(f"Average distance: {np.mean(distances):.4f} m")
    print(f"Target velocity: 2.0 m/s")
    print(f"Achievement: {(overall_avg/2.0)*100:.1f}% of target")

if __name__ == "__main__":
    model_path = "experiments/ppo_custom_reward_fzzp48df/best_model/best_model.zip"
    norm_path = "experiments/ppo_custom_reward_fzzp48df/vec_normalize.pkl"
    
    test_model_velocity(model_path, norm_path)