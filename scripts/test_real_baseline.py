#!/usr/bin/env python3
"""
Test the ACTUAL baseline model that created the smooth video
"""

import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import realant_sim
import imageio

def test_real_baseline():
    """Test the model that actually created the smooth walking video"""
    
    model_path = "archive/experiments/ppo_baseline_ueqbjf2x/best_model/best_model.zip"
    vec_norm_path = "archive/experiments/ppo_baseline_ueqbjf2x/vec_normalize.pkl"
    
    print("🎯 TESTING THE REAL BASELINE MODEL")
    print(f"Model: {model_path}")
    print(f"VecNorm: {vec_norm_path}")
    print("="*60)
    
    model = PPO.load(model_path)
    
    # Test with VecNormalize (likely setup)
    def make_env():
        return gym.make('RealAntMujoco-v0', render_mode='rgb_array')
    
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(vec_norm_path, env)
    env.training = False
    env.norm_reward = False
    
    obs = env.reset()
    frames = []
    positions = []
    rewards = []
    
    print("Recording smooth walking test...")
    
    for step in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # Get position
        x_pos = env.envs[0].unwrapped.data.qpos[0]
        z_pos = env.envs[0].unwrapped.data.qpos[2]
        positions.append(x_pos)
        rewards.append(reward[0])
        
        # Capture frame
        frame = env.render(mode='rgb_array')
        if frame is not None:
            frames.append(frame)
        
        if step % 50 == 0:
            distance = positions[-1] - positions[0] if len(positions) > 1 else 0
            velocity = (positions[-1] - positions[-10]) / (10 * 0.05) if len(positions) > 10 else 0
            print(f"Step {step:3d}: X={x_pos:.3f}, Z={z_pos:.3f}, Dist={distance:.3f}, Vel={velocity:.3f}, Reward={reward[0]:.2f}")
        
        if done[0]:
            print(f"Episode ended at step {step}")
            break
    
    # Save video
    if frames:
        imageio.mimsave('REAL_baseline_test.mp4', frames, fps=30)
        print("✅ Saved REAL_baseline_test.mp4")
    
    # Results
    final_distance = positions[-1] - positions[0] if len(positions) > 1 else 0
    avg_velocity = final_distance / (len(positions) * 0.05) if positions else 0
    total_reward = sum(rewards)
    avg_reward = np.mean(rewards)
    
    print("\n" + "="*60)
    print("REAL BASELINE RESULTS:")
    print(f"Final distance: {final_distance:.3f} m")
    print(f"Average velocity: {avg_velocity:.3f} m/s") 
    print(f"Total reward: {total_reward:.1f}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Steps completed: {len(positions)}")
    
    # Check if stable (no falling)
    z_positions = [env.envs[0].unwrapped.data.qpos[2] for _ in range(10)]  # Get final Z
    final_z = env.envs[0].unwrapped.data.qpos[2]
    
    if final_z > 0.15:  # Still standing
        print("✅ Robot remained stable (no falling)")
    else:
        print("❌ Robot may have fallen")
    
    if avg_velocity > 0.03 and avg_reward > -1:
        print("🎉 THIS LOOKS PROMISING FOR SMOOTH WALKING!")
    else:
        print("😞 Still not working well")
    
    print("="*60)
    
    env.close()
    return avg_velocity > 0.03

if __name__ == "__main__":
    works = test_real_baseline()
    if works:
        print("\n🏆 SUCCESS! Found the working baseline model!")
        print("Now we can use this for all our evaluations!")
    else:
        print("\n😞 Even the 'real' baseline doesn't work...")
        print("The mystery continues...")