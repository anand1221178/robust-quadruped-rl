#!/usr/bin/env python3
"""
Create side-by-side comparison video of Persistent DR vs Permanent DR using two-pass approach
Shows joint health and failure indicators
"""

import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
from envs.persistent_dr_wrapper import PersistentDRWrapper
from envs.permanent_dr_wrapper import PermanentDRWrapper
import realant_sim
import imageio
import os
from PIL import Image, ImageDraw, ImageFont
import argparse

def collect_trajectory_with_dr(model_path, dr_type='none', episode_length=300):
    """First pass: Collect trajectory with DR applied"""
    
    print(f"=" * 60)
    print(f"PASS 1: Collecting trajectory for {dr_type.upper()} DR")
    print(f"=" * 60)
    
    # Load model
    model = PPO.load(model_path)
    print(f"Loaded model from {model_path}")
    
    # Create environment WITHOUT rendering
    def make_env():
        env = gym.make('RealAntMujoco-v0')  # No render_mode!
        env = SuccessRewardWrapper(env)
        
        # Apply DR wrapper
        if dr_type == 'persistent':
            print("Applying Persistent DR Wrapper")
            persistent_config = {
                'failure_prob': 0.15,
                'max_failed_joints': 2,
                'duration_probs': [0.4, 0.4, 0.2],
                'short_duration': [50, 200], 
                'medium_duration': [200, 1000],
                'failure_types': ['lock', 'weak', 'erratic'],
                'failure_type_probs': [0.5, 0.3, 0.2],
                'use_curriculum': False,  # No curriculum for testing
                'warmup_steps': 0,
                'curriculum_steps': 0
            }
            env = PersistentDRWrapper(env, persistent_config)
        elif dr_type == 'permanent':
            print("Applying Permanent DR Wrapper")
            env = PermanentDRWrapper(env,
                failure_rate=0.001,
                max_failed_joints=4,
                warmup_steps=0,
                curriculum_steps=0,
                verbose=True
            )
        
        return env
    
    env = DummyVecEnv([make_env])
    
    # Load VecNormalize
    vec_path = model_path.replace('final_model.zip', 'vec_normalize.pkl')
    if os.path.exists(vec_path):
        env = VecNormalize.load(vec_path, env)
        env.training = False
        env.norm_reward = False
        print(f"Loaded VecNormalize from: {vec_path}")
    
    # Collect trajectory
    obs = env.reset()
    trajectory = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'infos': [],
        'dones': [],
        'joint_health': []  # Track joint failures
    }
    
    velocities = []
    episode_reward = 0
    
    print(f"Collecting {episode_length} steps...")
    for step in range(episode_length):
        # Store observation
        trajectory['observations'].append(obs.copy())
        
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        trajectory['actions'].append(action.copy())
        
        # Step environment
        obs, reward, done, info = env.step(action)
        
        # Store results
        trajectory['rewards'].append(reward[0])
        trajectory['infos'].append(info[0])
        trajectory['dones'].append(done[0])
        
        # Track joint health (extract from wrapper if available)
        joint_health = 1.0  # Default: all joints healthy
        if hasattr(env.envs[0], 'get_joint_health'):
            joint_health = env.envs[0].get_joint_health()
        elif hasattr(env.envs[0], 'failed_joints'):
            # For permanent DR, calculate health based on failed joints
            total_joints = 8
            failed_count = len(env.envs[0].failed_joints)
            joint_health = 1.0 - (failed_count / total_joints)
        
        trajectory['joint_health'].append(joint_health)
        
        episode_reward += reward[0]
        
        # Track velocity if available
        if info[0] is not None and 'speed' in info[0]:
            velocities.append(info[0]['speed'])
        
        if done[0]:
            print(f"Episode ended early at step {step}")
            break
    
    env.close()
    
    # Print performance metrics
    print(f"\nActual Performance ({dr_type.upper()} DR):")
    print(f"Total Reward: {episode_reward:.2f}")
    print(f"Episode Length: {len(trajectory['actions'])} steps")
    if velocities:
        print(f"Average Velocity: {np.mean(velocities):.3f} m/s")
    print(f"Average Joint Health: {np.mean(trajectory['joint_health']):.1%}")
    
    return trajectory, np.mean(velocities) if velocities else 0.0

def replay_with_rendering_and_overlay(trajectory, dr_type, output_path):
    """Second pass: Replay trajectory with rendering and joint health overlay"""
    
    print(f"\nPASS 2: Replaying {dr_type.upper()} DR with rendering...")
    
    # Create environment WITH rendering
    def make_env():
        env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')
        env = SuccessRewardWrapper(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Reset environment
    obs = env.reset()
    frames = []
    
    print(f"Replaying {len(trajectory['actions'])} steps...")
    
    # Try to load a font for text overlay
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
        small_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Replay the exact trajectory
    for step, action in enumerate(trajectory['actions']):
        obs, reward, done, info = env.step(action)
        
        # Capture frame
        frame = env.render(mode='rgb_array')
        if frame is not None:
            # Add overlay information
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            
            # Model name label
            model_name = f"{dr_type.upper()} DR"
            draw.rectangle([(10, 10), (200, 40)], fill=(0, 0, 0, 200))
            draw.text((15, 15), model_name, fill=(255, 255, 255), font=font)
            
            # Joint health indicator
            joint_health = trajectory['joint_health'][step]
            health_color = (int(255 * (1 - joint_health)), int(255 * joint_health), 0)  # Red to green
            
            draw.rectangle([(10, 50), (200, 75)], fill=(0, 0, 0, 200))
            draw.text((15, 53), f"Joint Health: {joint_health:.1%}", fill=health_color, font=small_font)
            
            # Health bar
            bar_width = 150
            bar_height = 8
            bar_x, bar_y = 15, 80
            
            # Background bar (red)
            draw.rectangle([(bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height)], fill=(80, 80, 80))
            # Health bar (color based on health)
            health_width = int(bar_width * joint_health)
            draw.rectangle([(bar_x, bar_y), (bar_x + health_width, bar_y + bar_height)], fill=health_color)
            
            # Step counter
            draw.rectangle([(10, 100), (150, 125)], fill=(0, 0, 0, 180))
            draw.text((15, 103), f"Step: {step+1}", fill=(255, 255, 255), font=small_font)
            
            # Velocity if available
            if step < len(trajectory['infos']) and trajectory['infos'][step] is not None:
                if 'speed' in trajectory['infos'][step]:
                    velocity = trajectory['infos'][step]['speed']
                    draw.rectangle([(10, 130), (200, 155)], fill=(0, 0, 0, 180))
                    draw.text((15, 133), f"Velocity: {velocity:.3f} m/s", fill=(255, 255, 255), font=small_font)
            
            frames.append(np.array(img))
        
        if done[0]:
            break
    
    env.close()
    
    # Save individual video
    if frames:
        print(f"Saving {dr_type} video to {output_path}...")
        imageio.mimsave(output_path, frames, fps=30)
        print(f"Video saved! ({len(frames)} frames)")
    
    return frames

def create_side_by_side_comparison():
    """Create side-by-side comparison of Persistent vs Permanent DR"""
    
    print("🎬 CREATING PERSISTENT DR vs PERMANENT DR COMPARISON VIDEO")
    print("=" * 80)
    
    # Model paths
    models = {
        'persistent': 'experiments/ppo_persistent_dr_resume_h96y2uqe/final_model.zip',
        'permanent': 'experiments/ppo_permanent_dr_resume_pv9vmhor/final_model.zip'
    }
    
    # Collect trajectories
    trajectories = {}
    velocities = {}
    all_frames = {}
    
    for dr_type, model_path in models.items():
        print(f"\n{'='*60}")
        print(f"PROCESSING {dr_type.upper()} DR MODEL")
        print(f"{'='*60}")
        
        # Pass 1: Collect trajectory
        trajectory, velocity = collect_trajectory_with_dr(model_path, dr_type)
        trajectories[dr_type] = trajectory
        velocities[dr_type] = velocity
        
        # Pass 2: Render with overlays
        frames = replay_with_rendering_and_overlay(
            trajectory, 
            dr_type, 
            f"{dr_type}_dr_video.mp4"
        )
        all_frames[dr_type] = frames
    
    # Create side-by-side video
    print(f"\n{'='*60}")
    print("CREATING SIDE-BY-SIDE COMPARISON")
    print(f"{'='*60}")
    
    # Get minimum frame count
    min_frames = min(len(all_frames['persistent']), len(all_frames['permanent']))
    
    combined_frames = []
    for i in range(min_frames):
        frame1 = all_frames['persistent'][i]
        frame2 = all_frames['permanent'][i]
        
        # Resize frames to same height if needed
        h = min(frame1.shape[0], frame2.shape[0])
        frame1_resized = frame1[:h, :]
        frame2_resized = frame2[:h, :]
        
        # Combine horizontally
        combined = np.hstack([frame1_resized, frame2_resized])
        
        # Add comparison title at the top
        img = Image.fromarray(combined)
        draw = ImageDraw.Draw(img)
        
        # Title bar
        title_height = 40
        title_img = Image.new('RGB', (combined.shape[1], title_height), color=(40, 40, 40))
        title_draw = ImageDraw.Draw(title_img)
        
        try:
            title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
        except:
            title_font = ImageFont.load_default()
        
        title_text = "Persistent DR vs Permanent DR Comparison"
        title_draw.text((20, 10), title_text, fill=(255, 255, 255), font=title_font)
        
        # Performance metrics
        perf_text = f"Persistent: {velocities['persistent']:.3f} m/s | Permanent: {velocities['permanent']:.3f} m/s"
        try:
            perf_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
        except:
            perf_font = ImageFont.load_default()
        title_draw.text((20, 25), perf_text, fill=(200, 200, 200), font=perf_font)
        
        # Combine title with video frame
        final_frame = np.vstack([np.array(title_img), combined])
        combined_frames.append(final_frame)
    
    # Save comparison video
    output_path = "persistent_vs_permanent_dr_comparison.mp4"
    imageio.mimsave(output_path, combined_frames, fps=30)
    
    print(f"\n🎉 COMPARISON VIDEO CREATED!")
    print(f"✅ File: {output_path}")
    print(f"✅ Frames: {len(combined_frames)}")
    print(f"✅ Persistent DR Performance: {velocities['persistent']:.3f} m/s")
    print(f"✅ Permanent DR Performance: {velocities['permanent']:.3f} m/s")
    print("=" * 80)

if __name__ == "__main__":
    create_side_by_side_comparison()