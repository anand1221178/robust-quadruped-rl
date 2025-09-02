#!/usr/bin/env python3
"""
Test baseline model with both wrappers to see the difference
"""

import sys
import os
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import realant_sim
from envs.target_walking_wrapper import TargetWalkingWrapper
from envs.success_reward_wrapper import SuccessRewardWrapper

def test_wrapper(wrapper_name, wrapper_class, use_vecnorm=True):
    """Test model with specific wrapper"""
    
    print(f"\n{'='*60}")
    print(f"TESTING WITH {wrapper_name}")
    if not use_vecnorm:
        print("(WITHOUT VecNormalize)")
    print(f"{'='*60}")
    
    model_path = "experiments/ppo_target_walking_llsm451b/best_model/best_model.zip"
    model = PPO.load(model_path)
    
    def make_env():
        env = gym.make('RealAntMujoco-v0')
        if wrapper_class == TargetWalkingWrapper:
            env = wrapper_class(env, target_distance=5.0)
        else:
            env = wrapper_class(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    if use_vecnorm:
        vec_norm_path = "experiments/ppo_target_walking_llsm451b/vec_normalize.pkl"
        if os.path.exists(vec_norm_path):
            env = VecNormalize.load(vec_norm_path, env)
            env.training = False
            env.norm_reward = False
    
    obs = env.reset()
    
    positions = []
    velocities = []
    rewards = []
    
    for i in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        rewards.append(reward[0])
        
        # Get position
        actual_env = env.envs[0] if hasattr(env, 'envs') else env.env.envs[0]
        x_pos = actual_env.unwrapped.data.qpos[0]
        positions.append(x_pos)
        
        if i > 0:
            vel = (positions[-1] - positions[-2]) / 0.05
            velocities.append(vel)
        
        if i % 25 == 0 and i > 0:
            avg_vel = np.mean(velocities[-10:])
            print(f"Step {i:3d}: X={x_pos:6.3f}, Vel={avg_vel:6.3f} m/s, "
                  f"Reward={reward[0]:8.2f}, Action std={action.std():.3f}")
        
        if done[0]:
            print(f"Episode ended at step {i}")
            break
    
    avg_velocity = np.mean(velocities) if velocities else 0
    total_reward = sum(rewards)
    final_distance = positions[-1] - positions[0] if positions else 0
    
    print(f"\nFINAL RESULTS:")
    print(f"  Average velocity: {avg_velocity:.3f} m/s")
    print(f"  Total distance: {final_distance:.3f} m")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Action variability: {np.std([action.std() for _ in range(10)]):.3f}")
    
    env.close()
    return avg_velocity, total_reward, final_distance

if __name__ == "__main__":
    # Test all combinations
    results = {}
    
    # Test SuccessRewardWrapper (what might have been used originally)
    vel1, rew1, dist1 = test_wrapper("SuccessRewardWrapper", SuccessRewardWrapper, use_vecnorm=True)
    results["Success+VecNorm"] = (vel1, rew1, dist1)
    
    # Test TargetWalkingWrapper (what config says)
    vel2, rew2, dist2 = test_wrapper("TargetWalkingWrapper", TargetWalkingWrapper, use_vecnorm=True)
    results["Target+VecNorm"] = (vel2, rew2, dist2)
    
    # Test SuccessRewardWrapper without VecNorm
    vel3, rew3, dist3 = test_wrapper("SuccessRewardWrapper", SuccessRewardWrapper, use_vecnorm=False)
    results["Success+NoNorm"] = (vel3, rew3, dist3)
    
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    for name, (vel, rew, dist) in results.items():
        print(f"{name:20s}: Vel={vel:6.3f} m/s, Reward={rew:8.1f}, Dist={dist:6.3f} m")
    
    # Find the best one
    best_setup = max(results.items(), key=lambda x: x[1][0])
    print(f"\n🏆 BEST PERFORMANCE: {best_setup[0]} with {best_setup[1][0]:.3f} m/s")
    
    if "Success" in best_setup[0]:
        print("\n💡 CONCLUSION: Model was likely trained with SuccessRewardWrapper!")
        print("   The config might be wrong, or there was a mismatch during training.")
    else:
        print("\n💡 CONCLUSION: Model was trained with TargetWalkingWrapper as expected.")