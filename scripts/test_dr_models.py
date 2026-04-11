#!/usr/bin/env python3
"""Quick test of DR models"""
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
import realant_sim

def test_model(model_path, vec_norm_path, name):
    """Test a model quickly"""
    print(f"\n{'='*50}")
    print(f"Testing: {name}")
    print(f"{'='*50}")
    
    def make_env():
        env = gym.make('RealAntMujoco-v0')
        env = SuccessRewardWrapper(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    try:
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False
        print("✅ VecNormalize loaded")
    except:
        print("⚠️ No VecNormalize - using raw env")
    
    model = PPO.load(model_path)
    print("✅ Model loaded")
    
    # Quick test
    obs = env.reset()
    total_reward = 0
    positions = []
    
    for step in range(500):  # 25 seconds
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        
        if done[0]:
            print(f"Episode ended at step {step}")
            break
            
        # Track position
        x_pos = env.envs[0].unwrapped.data.qpos[0]
        positions.append(x_pos)
        
        if step % 100 == 0:
            vel = (positions[-1] - positions[0]) / ((step+1) * 0.05) if positions else 0
            print(f"  Step {step}: x={x_pos:.2f}m, vel={vel:.3f}m/s, reward={reward[0]:.0f}")
    
    # Final stats
    if len(positions) > 1:
        distance = positions[-1] - positions[0]
        time = len(positions) * 0.05
        velocity = distance / time
        print(f"\n📊 Results:")
        print(f"  Distance: {distance:.2f}m")
        print(f"  Velocity: {velocity:.3f}m/s")
        print(f"  Total Reward: {total_reward:.0f}")
    
    env.close()

if __name__ == "__main__":
    print("🚀 Testing Completed Models")
    
    # Test SR2L (COMPLETE - 20M steps)
    test_model(
        "experiments/ppo_sr2l_forward_m7gtjtpa/final_model.zip",
        "experiments/ppo_sr2l_forward_m7gtjtpa/vec_normalize.pkl",
        "SR2L with Tanh (COMPLETE - 20M steps)"
    )
    
    # Test Persistent DR
    test_model(
        "experiments/ppo_persistent_dr_forward_qv9kub6b/final_model.zip",
        "experiments/ppo_persistent_dr_forward_qv9kub6b/vec_normalize.pkl",
        "Persistent DR (10M/25M steps - 40% complete)"
    )
    
    # Test Permanent DR if it has a final model
    import os
    if os.path.exists("experiments/ppo_permanent_dr_forward_030el90h/final_model.zip"):
        test_model(
            "experiments/ppo_permanent_dr_forward_030el90h/final_model.zip",
            "experiments/ppo_permanent_dr_forward_030el90h/vec_normalize.pkl",
            "Permanent DR (10M/30M steps)"
        )
    else:
        # Try a checkpoint
        checkpoint = "experiments/ppo_permanent_dr_forward_030el90h/checkpoints/checkpoint_9950000_steps.zip"
        if os.path.exists(checkpoint):
            print("\n⚠️ No final model for Permanent DR, testing checkpoint at 9.95M steps")
            test_model(
                checkpoint,
                "experiments/ppo_permanent_dr_forward_030el90h/vec_normalize.pkl",
                "Permanent DR Checkpoint (9.95M/30M steps)"
            )