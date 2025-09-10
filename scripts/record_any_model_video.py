#!/usr/bin/env python3
"""
Record video of ANY model with SuccessRewardWrapper (no target walking)
Shows actual forward locomotion performance
"""
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
import realant_sim
import cv2
import os
from datetime import datetime
import argparse

def record_model_video(model_folder, steps=1000):
    """Two-pass video of any model with SuccessRewardWrapper"""
    
    model_name = os.path.basename(model_folder)
    print(f"🎬 TWO-PASS VIDEO: {model_name.upper()}")
    print(f"✅ Model: {model_folder}")  
    print(f"✅ Wrapper: SuccessRewardWrapper (forward speed rewards)")
    print(f"🎯 Steps: {steps}")
    print("=" * 70)
    
    # Model paths
    model_path = os.path.join(model_folder, 'final_model.zip')
    if not os.path.exists(model_path):
        model_path = os.path.join(model_folder, 'best_model', 'best_model.zip')
    
    vec_normalize_path = os.path.join(model_folder, 'vec_normalize.pkl')
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    if not os.path.exists(vec_normalize_path):
        print(f"❌ VecNormalize not found: {vec_normalize_path}")
        return
    
    # PASS 1: Collect trajectory
    print("📁 PASS 1: Collecting trajectory (TRUE performance)")
    print("-" * 50)
    
    # Create environment
    env = gym.make('RealAntMujoco-v0')
    env = SuccessRewardWrapper(env)
    print(f"📏 Observation space: {env.observation_space}")
    
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(vec_normalize_path, env)
    env.training = False
    env.norm_reward = False
    print("✅ VecNormalize loaded (spaces matched!)")
    
    # Load model
    model = PPO.load(model_path)
    print("✅ Model loaded")
    
    print(f"\n🤖 Recording robot performance...")
    
    # Collect trajectory data
    trajectory = {'obs': [], 'actions': [], 'rewards': []}
    obs = env.reset()
    trajectory['obs'].append(obs)
    
    total_reward = 0
    positions = []
    velocities = []
    
    # Get initial position
    if hasattr(env, 'get_original_obs'):
        original_obs = env.get_original_obs()[0]
        start_pos = original_obs[0]
    else:
        start_pos = 0.0
    
    print(f"🏃 Starting position: {start_pos:.3f}m")
    positions.append(start_pos)
    
    for step in range(steps):
        # Get position for velocity calculation
        if hasattr(env, 'get_original_obs'):
            original_obs = env.get_original_obs()[0]
            pos_x = original_obs[0]
        else:
            pos_x = start_pos
            
        positions.append(pos_x)
        
        # Calculate current velocity (over last 10 steps)
        if len(positions) >= 10:
            recent_positions = positions[-10:]
            distance = recent_positions[-1] - recent_positions[0]
            time_taken = 9 * 0.05  # 9 steps * 0.05s/step
            current_velocity = distance / time_taken if time_taken > 0 else 0
        else:
            current_velocity = 0
            
        velocities.append(current_velocity)
        
        # Progress output
        if step % 100 == 0:
            print(f"  Step {step:3d}: x={pos_x:5.1f}m, vel={current_velocity:.3f}m/s")
        
        # Predict and step
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        trajectory['actions'].append(action)
        trajectory['rewards'].append(reward)
        trajectory['obs'].append(obs)
        
        total_reward += reward[0] if hasattr(reward, '__len__') else reward
        
        if done.any() if hasattr(done, 'any') else done:
            print(f"Episode ended at step {step}")
            break
    
    # Calculate final metrics
    final_distance = positions[-1] - positions[0]
    total_time = len(positions) * 0.05
    avg_velocity = final_distance / total_time if total_time > 0 else 0
    
    print(f"\n📊 TRAJECTORY PERFORMANCE:")
    print(f"  Steps: {len(trajectory['obs'])}")
    print(f"  Total reward: {total_reward:.0f}")
    print(f"  Distance traveled: {final_distance:.2f}m")
    print(f"  Average velocity: {avg_velocity:.3f} m/s")
    
    if avg_velocity < 0:
        print("⚠️  NEGATIVE VELOCITY - Robot walking backwards!")
    elif avg_velocity < 0.05:
        print("⚠️  Very low velocity - robot barely moving")
    elif avg_velocity > 0.15:
        print("✅ Good forward locomotion")
    
    # PASS 2: Render video
    print(f"\n🎥 PASS 2: Rendering video from trajectory")
    print("-" * 50)
    
    # Create new environment for rendering
    render_env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')
    render_env = SuccessRewardWrapper(render_env)
    
    print("🎬 Rendering video...")
    frames = []
    obs = render_env.reset()[0] if hasattr(render_env.reset(), '__len__') else render_env.reset()
    
    for i, action in enumerate(trajectory['actions']):
        if i % 100 == 0:
            print(f"  Rendered {i+1}/{len(trajectory['actions'])} frames...")
        
        # Step and render
        step_result = render_env.step(action[0] if hasattr(action, '__len__') else action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result
        frame = render_env.render()
        
        # Add text overlay
        if frame is not None:
            frame_with_text = frame.copy()
            
            # Current step velocity
            step_vel = velocities[min(i, len(velocities)-1)]
            
            # Add performance text
            cv2.putText(frame_with_text, f'Step: {i+1}/{len(trajectory["actions"])}', 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_with_text, f'Velocity: {step_vel:.3f} m/s', 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_with_text, f'Model: {model_name}', 
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            frames.append(frame_with_text)
        
        if done:
            break
    
    render_env.close()
    
    if frames:
        # Save video
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{model_name}_video_{timestamp}.mp4'
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
        
        for frame in frames:
            out.write(frame)
        out.release()
        
        print(f"\n✅ VIDEO COMPLETED!")
        print(f"📁 Saved as: {filename}")
        print(f"📊 Performance: {avg_velocity:.3f} m/s")
        
        # Status summary
        if avg_velocity < 0:
            print("🚨 ALERT: Robot walks backwards!")
        elif avg_velocity < 0.05:
            print("⚠️  WARNING: Robot barely moves")  
        elif avg_velocity > 0.15:
            print("✅ SUCCESS: Good locomotion")
        else:
            print("📊 INFO: Moderate locomotion")
            
        return filename
    else:
        print("❌ No frames captured!")
        return None

def main():
    parser = argparse.ArgumentParser(description='Record video of any model')
    parser.add_argument('model_folder', help='Path to model folder (containing final_model.zip and vec_normalize.pkl)')
    parser.add_argument('--steps', type=int, default=1000, help='Number of steps to record (default: 1000)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_folder):
        print(f"❌ Model folder not found: {args.model_folder}")
        sys.exit(1)
    
    filename = record_model_video(args.model_folder, args.steps)
    
    if filename:
        print(f"\n🎬 Video ready: {filename}")
    else:
        print(f"\n❌ Video creation failed")

if __name__ == "__main__":
    main()