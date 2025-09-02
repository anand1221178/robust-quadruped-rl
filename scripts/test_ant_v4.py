#!/usr/bin/env python3
"""
Test if the baseline model works with standard Ant-v4 instead of RealAnt
"""

import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import imageio

def test_ant_v4():
    """Test baseline model with standard Ant-v4"""
    
    print("Testing baseline model with Ant-v4...")
    model = PPO.load('experiments/ppo_target_walking_llsm451b/best_model/best_model.zip')
    
    env = gym.make('Ant-v4', render_mode='rgb_array')
    obs, _ = env.reset()
    frames = []
    
    positions = []
    
    for i in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Get position
        x_pos = env.unwrapped.data.qpos[0]
        positions.append(x_pos)
        
        frame = env.render()
        frames.append(frame)
        
        if i % 50 == 0:
            distance = positions[-1] - positions[0] if len(positions) > 1 else 0
            velocity = (positions[-1] - positions[-10]) / (10 * 0.05) if len(positions) > 10 else 0
            print(f'Step {i:3d}: X={x_pos:.3f}, Dist={distance:.3f}, Vel={velocity:.3f}, Reward={reward:.2f}')
        
        if terminated or truncated:
            print(f'Episode ended at step {i}')
            break
    
    # Save video
    imageio.mimsave('test_ant_v4.mp4', frames, fps=30)
    print('Saved test_ant_v4.mp4')
    
    # Results
    final_distance = positions[-1] - positions[0] if len(positions) > 1 else 0
    avg_velocity = final_distance / (len(positions) * 0.05) if positions else 0
    
    print(f'\nFINAL RESULTS:')
    print(f'Distance: {final_distance:.3f} m')
    print(f'Velocity: {avg_velocity:.3f} m/s')
    print(f'Steps: {len(positions)}')
    
    env.close()
    return avg_velocity > 0.1  # Return True if it works well

if __name__ == "__main__":
    works = test_ant_v4()
    if works:
        print("✅ Ant-v4 might be the solution!")
    else:
        print("❌ Ant-v4 doesn't work either")