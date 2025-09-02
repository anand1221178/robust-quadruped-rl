#!/usr/bin/env python3
"""
Simple Live Performance Monitor - Shows real-time metrics while robot walks
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import argparse
import os
import sys
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import environments
import realant_sim
from envs.target_walking_wrapper import TargetWalkingWrapper

def test_with_live_plots(model_path, max_steps=1000):
    """Run model with live performance plots"""
    
    print(f"\n{'='*60}")
    print(f"LIVE PERFORMANCE MONITOR")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Press Ctrl+C to stop\n")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    env = gym.make('RealAntMujoco-v0')
    env = TargetWalkingWrapper(env, target_distance=5.0)
    env = DummyVecEnv([lambda: env])
    
    # Load normalization if exists
    exp_dir = os.path.dirname(os.path.dirname(model_path))
    vec_norm_path = os.path.join(exp_dir, 'vec_normalize.pkl')
    if os.path.exists(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False
        print("✅ Loaded normalization stats")
    
    # Setup live plotting
    plt.ion()  # Interactive mode
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Live Robot Performance', fontsize=14)
    
    # Configure subplots
    ax_velocity = axes[0, 0]
    ax_reward = axes[0, 1]
    ax_distance = axes[1, 0]
    ax_actions = axes[1, 1]
    
    # Data storage - NO LIMIT, keep all history!
    velocities = []
    rewards = []
    distances = []
    action_history = []
    timesteps = []
    
    # Initialize plots
    line_vel, = ax_velocity.plot([], [], 'g-', linewidth=2)
    line_reward, = ax_reward.plot([], [], 'b-', linewidth=2)
    line_dist, = ax_distance.plot([], [], 'r-', linewidth=2)
    
    # Action plot (bar chart)
    action_bars = None
    
    # Configure axes (auto-scaling)
    ax_velocity.set_title('Velocity (m/s)')
    ax_velocity.grid(True, alpha=0.3)
    ax_velocity.set_xlabel('Steps')
    
    ax_reward.set_title('Instant Reward')
    ax_reward.grid(True, alpha=0.3)
    ax_reward.set_xlabel('Steps')
    
    ax_distance.set_title('Distance Traveled (m)')
    ax_distance.grid(True, alpha=0.3)
    ax_distance.set_xlabel('Steps')
    
    ax_actions.set_title('Current Actions (Motor Commands)')
    ax_actions.set_ylim(-1.2, 1.2)
    ax_actions.grid(True, alpha=0.3)
    ax_actions.set_xlabel('Joint Index')
    
    # Run episode
    obs = env.reset()
    total_distance = 0
    total_reward = 0
    step = 0
    
    try:
        for step in range(max_steps):
            # Get action
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            # Extract metrics - TargetWalkingWrapper provides 'speed'
            velocity = 0
            if 'speed' in info[0]:
                velocity = info[0]['speed']
            elif 'x_velocity' in info[0]:
                velocity = info[0]['x_velocity']
            elif 'progress' in info[0]:
                # Calculate from progress
                velocity = info[0]['progress'] / 0.05
            
            # Update metrics
            total_distance += abs(velocity) * 0.05
            total_reward += reward[0]
            
            # Store data
            timesteps.append(step)
            velocities.append(velocity)
            rewards.append(reward[0])
            distances.append(total_distance)
            action_history.append(action[0])
            
            # Update plots every 5 steps (for performance)
            if step % 5 == 0:
                # Update line plots
                if len(timesteps) > 1:
                    line_vel.set_data(timesteps, velocities)
                    line_reward.set_data(timesteps, rewards)
                    line_dist.set_data(timesteps, distances)
                    
                    # Adjust limits - show all history with sliding window
                    window_size = 200  # Show last 200 points but keep all data
                    for ax in [ax_velocity, ax_reward, ax_distance]:
                        if step > window_size:
                            ax.set_xlim(step - window_size, step + 10)
                        else:
                            ax.set_xlim(0, max(window_size, step + 10))
                        ax.relim()  # Recalculate limits
                        ax.autoscale_view(scaley=True)  # Auto-scale y-axis
                    
                    # Update action bar chart
                    ax_actions.clear()
                    ax_actions.bar(range(8), action[0], color='cyan', alpha=0.7)
                    ax_actions.set_title('Current Actions (Motor Commands)')
                    ax_actions.set_ylim(-1.2, 1.2)
                    ax_actions.grid(True, alpha=0.3)
                    ax_actions.set_xlabel('Joint Index')
                    ax_actions.set_xticks(range(8))
                    ax_actions.set_xticklabels(['FL_H', 'FL_K', 'FR_H', 'FR_K', 
                                               'BL_H', 'BL_K', 'BR_H', 'BR_K'])
                
                # Update display
                plt.draw()
                plt.pause(0.001)
            
            # Print live stats every 50 steps
            if step % 50 == 0:
                print(f"Step {step:4d} | Vel: {velocity:6.3f} m/s | "
                      f"Dist: {total_distance:6.2f} m | "
                      f"Reward: {total_reward:7.2f}")
            
            if done:
                print("\n🏁 Episode terminated!")
                break
                
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    
    # Final stats
    print(f"\n{'='*60}")
    print("FINAL STATISTICS")
    print(f"{'='*60}")
    print(f"Steps completed: {step}")
    print(f"Total distance: {total_distance:.2f} m")
    print(f"Average velocity: {np.mean(list(velocities)):.3f} m/s")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final velocity: {velocities[-1] if velocities else 0:.3f} m/s")
    print(f"{'='*60}\n")
    
    plt.ioff()
    plt.show()
    
    env.close()

def main():
    parser = argparse.ArgumentParser(description='Live Performance Monitor')
    parser.add_argument('model_path', type=str, help='Path to trained model')
    parser.add_argument('--steps', type=int, default=1000, help='Max steps to run')
    
    args = parser.parse_args()
    
    test_with_live_plots(args.model_path, args.steps)

if __name__ == "__main__":
    main()