#!/usr/bin/env python3
"""
Create video of SR2L Tanh model showing sensor noise robustness
Gradually increases sensor noise to show what the model was trained for
"""

import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
import realant_sim
import imageio
import os
from PIL import Image, ImageDraw, ImageFont

def collect_trajectory_with_noise(model_path, noise_levels, episode_length=200):
    """Collect trajectory with gradually increasing sensor noise"""
    
    print("=" * 60)
    print("COLLECTING SR2L TRAJECTORY WITH INCREASING SENSOR NOISE")
    print("=" * 60)
    
    # Load SR2L model
    model = PPO.load(model_path)
    print(f"Loaded SR2L model from {model_path}")
    
    # Create environment WITHOUT rendering (for accurate trajectory collection)
    def make_env():
        env = gym.make('RealAntMujoco-v0')  # No render_mode!
        env = SuccessRewardWrapper(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Load VecNormalize
    vec_path = model_path.replace('final_model.zip', 'vec_normalize.pkl')
    if os.path.exists(vec_path):
        env = VecNormalize.load(vec_path, env)
        env.training = False
        env.norm_reward = False
        print(f"Loaded VecNormalize from: {vec_path}")
    
    # Collect trajectory with varying noise levels
    obs = env.reset()
    trajectory = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'noise_levels': [],
        'velocities': []
    }
    
    print(f"Collecting {episode_length} steps with noise levels: {noise_levels}")
    
    for step in range(episode_length):
        # Determine current noise level (gradually increase, then hold at max)
        if step < len(noise_levels) * 30:  # During progression phase
            progress = step / (len(noise_levels) * 30)
            noise_idx = min(int(progress * len(noise_levels)), len(noise_levels) - 1)
            current_noise = noise_levels[noise_idx]
        else:  # Extended phase - hold at maximum noise
            current_noise = noise_levels[-1]  # Maximum noise (0.200)
        
        # Store original observation
        clean_obs = obs.copy()
        
        # Apply sensor noise to joint observations (dims 13-28 for RealAnt)
        # This is exactly what SR2L was trained to handle!
        noisy_obs = obs.copy()
        if current_noise > 0:
            # Add noise to joint positions (dims 13-20) and velocities (dims 21-28)
            joint_noise = np.random.normal(0, current_noise, 16)  # 8 pos + 8 vel = 16 joint sensors
            noisy_obs[0, 13:29] += joint_noise
        
        # Store data
        trajectory['observations'].append(clean_obs.copy())
        trajectory['noise_levels'].append(current_noise)
        
        # Get action from model using NOISY observations (what SR2L trained for)
        action, _ = model.predict(noisy_obs, deterministic=True)
        trajectory['actions'].append(action.copy())
        
        # Step environment with clean observations (environment uses true state)
        obs, reward, done, info = env.step(action)
        
        trajectory['rewards'].append(reward[0])
        
        # Track velocity - try multiple methods to get accurate measurement
        velocity = 0.0
        if info[0] is not None:
            if 'speed' in info[0]:
                velocity = info[0]['speed']
            elif 'current_velocity' in info[0]:
                velocity = info[0]['current_velocity']
            elif 'velocity' in info[0]:
                velocity = info[0]['velocity']
        
        # If no velocity in info, calculate from position change
        if velocity == 0.0 and step > 0:
            # Try to extract velocity from observation change (rough estimate)
            if len(obs[0]) > 2:  # Position should be in obs[0:3]
                prev_pos = trajectory['observations'][-1][0][:3] if trajectory['observations'] else np.zeros(3)
                curr_pos = obs[0][:3]
                pos_change = np.linalg.norm(curr_pos - prev_pos)
                velocity = pos_change * 20  # 20 Hz (assuming 50ms timesteps)
        
        trajectory['velocities'].append(velocity)
        
        if step % 40 == 0:  # Print every 2 seconds
            print(f"  Step {step:3d}: Noise={current_noise:.3f}, Velocity={velocity:.3f} m/s")
        
        if done[0]:
            print(f"Episode ended early at step {step}")
            break
    
    env.close()
    
    # Print performance summary
    print(f"\nSR2L SENSOR NOISE ROBUSTNESS TEST:")
    print(f"Episode Length: {len(trajectory['actions'])} steps")
    if trajectory['velocities']:
        print(f"Average Velocity: {np.mean(trajectory['velocities']):.3f} m/s")
        
        # Performance at different noise levels
        for i, noise_level in enumerate(noise_levels):
            start_idx = int(i * episode_length / len(noise_levels))
            end_idx = int((i + 1) * episode_length / len(noise_levels))
            segment_vels = trajectory['velocities'][start_idx:end_idx]
            if segment_vels:
                print(f"  Noise {noise_level:.3f}: {np.mean(segment_vels):.3f} m/s")
    
    return trajectory

def replay_with_noise_visualization(trajectory, output_path="sr2l_sensor_noise_test.mp4"):
    """Replay trajectory with sensor noise visualization at high quality"""
    
    print(f"\nReplaying trajectory with noise visualization at 1920x1080...")
    
    # Create environment WITH rendering
    def make_env():
        env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')
        env = SuccessRewardWrapper(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Reset environment
    obs = env.reset()
    frames = []
    
    # Try to load fonts - larger for high resolution
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 48)
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 36)
        small_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 28)
    except:
        title_font = ImageFont.load_default()
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    print(f"Replaying {len(trajectory['actions'])} steps...")
    
    for step, action in enumerate(trajectory['actions']):
        # Step environment
        obs, reward, done, info = env.step(action)
        
        # Capture frame
        frame = env.render(mode='rgb_array')
        if frame is not None:
            # Create overlay and resize to 1920x1080
            img = Image.fromarray(frame)
            img = img.resize((1920, 1080), Image.LANCZOS)
            draw = ImageDraw.Draw(img)
            
            # Get current data
            noise_level = trajectory['noise_levels'][step]
            velocity = trajectory['velocities'][step] if step < len(trajectory['velocities']) else 0.0
            
            # Main title - scaled for 1920x1080
            draw.rectangle([(20, 20), (800, 90)], fill=(20, 20, 80, 200))
            draw.text((30, 30), "SR2L: Sensor Noise Robustness Test", fill=(255, 255, 255), font=title_font)
            
            # Current sensor noise level with better color coding
            noise_intensity = min(noise_level / 0.05, 1.0)  # Normalize to training max
            noise_color = (
                min(255, int(255 * noise_intensity)), 
                max(100, int(255 * (1 - noise_intensity))), 
                int(100 * (1 - noise_intensity))
            )
            
            draw.rectangle([(20, 110), (800, 170)], fill=(40, 40, 40, 220))
            noise_text = f"Sensor Noise: {noise_level:.3f} std"
            if noise_level > 0.05:
                noise_text += f" ({noise_level/0.01:.1f}x training)"
            draw.text((30, 125), noise_text, fill=noise_color, font=font)
            
            # Enhanced noise intensity bar with zones - scaled up
            bar_width = 700
            bar_height = 30
            bar_x, bar_y = 30, 180
            max_noise_display = 0.20  # Show up to 0.20 noise
            
            # Background bar
            draw.rectangle([(bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height)], fill=(60, 60, 60))
            
            # Training zone (0.01) marker
            training_pos = int(bar_width * 0.01 / max_noise_display)
            draw.rectangle([(bar_x + training_pos - 2, bar_y), (bar_x + training_pos + 2, bar_y + bar_height)], fill=(0, 255, 0))
            
            # Max training zone (0.05) marker  
            max_training_pos = int(bar_width * 0.05 / max_noise_display)
            draw.rectangle([(bar_x + max_training_pos - 2, bar_y), (bar_x + max_training_pos + 2, bar_y + bar_height)], fill=(255, 255, 0))
            
            # Current noise level bar
            noise_width = int(bar_width * min(noise_level / max_noise_display, 1.0))
            draw.rectangle([(bar_x, bar_y), (bar_x + noise_width, bar_y + bar_height)], fill=noise_color)
            
            # Zone labels
            draw.text((30, 220), "Green=Training | Yellow=Max Training | Red=Beyond Training", fill=(200, 200, 200), font=small_font)
            
            # Step counter and progress
            draw.rectangle([(20, 260), (600, 320)], fill=(40, 40, 40, 180))
            progress = (step + 1) / len(trajectory['actions'])
            draw.text((30, 275), f"Step: {step+1} / {len(trajectory['actions'])} ({progress*100:.1f}%)", fill=(255, 255, 255), font=small_font)
            
            # Smoothness indicator (if we have previous action)
            if step > 0:
                prev_action = trajectory['actions'][step-1][0]
                curr_action = action[0]
                action_change = np.mean(np.abs(curr_action - prev_action))
                
                smoothness_score = 1.0 / (1.0 + action_change * 10)  # Similar to SR2L metric
                smoothness_color = (int(255 * (1 - smoothness_score)), int(255 * smoothness_score), 0)
                
                draw.rectangle([(20, 340), (640, 400)], fill=(40, 40, 40, 180))
                draw.text((30, 355), f"Action Smoothness: {smoothness_score:.3f}", fill=smoothness_color, font=small_font)
            
            frames.append(np.array(img))
        
        if done[0]:
            break
    
    env.close()
    
    # Save video
    if frames:
        print(f"Saving video to {output_path}...")
        imageio.mimsave(output_path, frames, fps=30)
        print(f"✅ Video saved! ({len(frames)} frames)")
        print(f"This video shows SR2L's response to increasing sensor noise")
    else:
        print("❌ No frames captured!")
    
    return len(frames)

def create_sr2l_sensor_noise_video():
    """Create SR2L sensor noise robustness demonstration video"""
    
    print("🎬 CREATING SR2L SENSOR NOISE ROBUSTNESS VIDEO")
    print("=" * 70)
    
    # SR2L model path
    model_path = 'experiments/ppo_sr2l_tanh_resume_hibwjaia/final_model.zip'
    
    # Comprehensive noise levels to test (much more detailed progression)
    noise_levels = [
        # Clean baseline
        0.000, 0.000, 0.000,  # 3 segments of clean performance
        
        # Very light noise
        0.002, 0.004, 0.006, 0.008,
        
        # Light noise (around training level)
        0.010, 0.012, 0.014, 0.016, 0.018,
        
        # Moderate noise
        0.020, 0.023, 0.026, 0.029, 
        
        # Heavy noise
        0.032, 0.035, 0.038, 0.041, 0.044,
        
        # Extreme noise (above training)
        0.047, 0.050, 0.055, 0.060, 0.065,
        
        # Stress test levels
        0.070, 0.075, 0.080, 0.090, 0.100,
        
        # Extreme stress test
        0.120, 0.150, 0.200
    ]
    
    print("Testing comprehensive noise levels:", len(noise_levels), "levels")
    print("Range: 0.000 → 0.200 (20x training noise!)")
    print("Training reference: perturbation_std=0.01, max_perturbation=0.05")
    
    # Much longer episode with extended ending
    episode_length = len(noise_levels) * 30 + 150  # 30 steps per noise level + 150 extra steps = ~1170 steps total
    
    # Pass 1: Collect trajectory without rendering
    trajectory = collect_trajectory_with_noise(model_path, noise_levels, episode_length)
    
    # Pass 2: Replay with visualization
    frames_rendered = replay_with_noise_visualization(trajectory, "sr2l_sensor_noise_robustness.mp4")
    
    print("\n" + "=" * 70)
    print("🎉 SR2L SENSOR NOISE VIDEO CREATED!")
    print("=" * 70)
    print("✅ File: sr2l_sensor_noise_robustness.mp4")
    print(f"✅ Frames: {frames_rendered}")
    print("✅ Shows gradual increase in joint sensor noise")
    print("✅ Demonstrates what SR2L was actually trained for")
    print("=" * 70)
    
    return trajectory

if __name__ == "__main__":
    trajectory = create_sr2l_sensor_noise_video()