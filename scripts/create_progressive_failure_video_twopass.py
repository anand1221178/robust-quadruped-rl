#!/usr/bin/env python3
"""
Create a side-by-side comparison video using TWO-PASS approach:
Pass 1: Collect trajectories without rendering (true performance)
Pass 2: Replay trajectories with rendering (accurate video)
"""

import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
import realant_sim
import os
from datetime import datetime

def apply_joint_failures(action, failure_rate):
    """Apply joint failures to actions"""
    action_copy = action.copy()
    if failure_rate > 0:
        for i in range(len(action_copy[0])):
            if np.random.random() < failure_rate:
                failure_type = np.random.choice(['lock', 'weak', 'noise'])
                if failure_type == 'lock':
                    action_copy[0][i] = 0.0  # Joint locked
                elif failure_type == 'weak':
                    action_copy[0][i] *= 0.3  # Weak joint (30% power)
                elif failure_type == 'noise':
                    action_copy[0][i] += np.random.normal(0, 0.5)  # Noisy joint
    return action_copy

def collect_trajectory(model, env, failure_rate=0.0, steps=150):
    """Pass 1: Collect trajectory WITHOUT rendering for true performance"""
    trajectory = []
    velocities = []
    
    obs = env.reset()
    
    for step in range(steps):
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Apply joint failures
        action_with_failures = apply_joint_failures(action, failure_rate)
        
        # Store trajectory data
        trajectory.append({
            'obs': obs.copy(),
            'action': action.copy(),
            'action_with_failures': action_with_failures.copy()
        })
        
        # Step environment (no rendering)
        obs, reward, done, info = env.step(action_with_failures)
        
        # Track velocity
        vel = 0.0
        if info[0]:
            if 'current_velocity' in info[0]:
                vel = info[0]['current_velocity']
            elif 'speed' in info[0]:
                vel = info[0]['speed']
            # Also check for x velocity directly
            try:
                base_env = env.venv.envs[0]
                if hasattr(base_env, 'sim') and base_env.sim is not None:
                    vel = base_env.sim.data.qvel[0]  # Direct x velocity
            except:
                pass
        velocities.append(vel)
        
        if done[0]:
            break
    
    avg_velocity = np.mean(velocities) if velocities else 0.0
    return trajectory, avg_velocity

def replay_trajectory_with_render(trajectory, env):
    """Pass 2: Replay trajectory WITH rendering to create video"""
    frames = []
    
    # Reset environment with rendering
    env.reset()
    
    for step_data in trajectory:
        # Step with recorded action (with failures already applied)
        obs, reward, done, info = env.step(step_data['action_with_failures'])
        
        # Render frame
        frame = env.render(mode='rgb_array')
        if frame is not None:
            frames.append(frame)
        
        if done[0]:
            break
    
    return frames

def create_progressive_video_twopass():
    """Create side-by-side comparison video using two-pass approach"""
    
    print("=" * 60)
    print("🎬 CREATING PROGRESSIVE FAILURE VIDEO (TWO-PASS METHOD)")
    print("=" * 60)
    
    # Model paths
    baseline_path = 'done/ppo_baseline_ueqbjf2x/best_model/best_model.zip'
    dr_path = 'done/ppo_dr_gentle_v2_wptws01u/best_model/best_model.zip'
    
    # Load models
    print("\n📦 Loading models...")
    baseline_model = PPO.load(baseline_path)
    dr_model = PPO.load(dr_path)
    
    # Video settings
    fps = 30
    output_path = f'videos/progressive_failure_twopass_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
    os.makedirs('videos', exist_ok=True)
    
    # Failure rate progression
    failure_stages = [
        (0.00, 150, "0% Joint Failures (Normal)"),
        (0.05, 150, "5% Joint Failures"),
        (0.10, 150, "10% Joint Failures"),
        (0.15, 150, "15% Joint Failures"),
        (0.20, 150, "20% Joint Failures"),
        (0.25, 150, "25% Joint Failures"),
        (0.30, 150, "30% Joint Failures (Extreme)")
    ]
    
    # Store all collected data
    all_data = []
    
    print("\n" + "="*60)
    print("PASS 1: Collecting trajectories WITHOUT rendering...")
    print("="*60)
    
    for failure_rate, steps, label in failure_stages:
        print(f"\n🎯 Collecting trajectories for {label}...")
        
        # Create environments WITHOUT rendering for Pass 1
        def make_env_no_render():
            env = gym.make('RealAntMujoco-v0')  # No render_mode
            env = SuccessRewardWrapper(env)
            return env
        
        baseline_env = DummyVecEnv([make_env_no_render])
        dr_env = DummyVecEnv([make_env_no_render])
        
        # Load VecNormalize if available
        baseline_norm_path = os.path.join(os.path.dirname(baseline_path), '..', 'vec_normalize.pkl')
        dr_norm_path = os.path.join(os.path.dirname(dr_path), '..', 'vec_normalize.pkl')
        
        if os.path.exists(baseline_norm_path):
            baseline_env = VecNormalize.load(baseline_norm_path, baseline_env)
            baseline_env.training = False
            baseline_env.norm_reward = False
            
        if os.path.exists(dr_norm_path):
            dr_env = VecNormalize.load(dr_norm_path, dr_env)
            dr_env.training = False
            dr_env.norm_reward = False
        
        # Collect trajectories (Pass 1)
        baseline_traj, baseline_vel = collect_trajectory(
            baseline_model, baseline_env, failure_rate, steps
        )
        dr_traj, dr_vel = collect_trajectory(
            dr_model, dr_env, failure_rate, steps
        )
        
        print(f"  ✅ Baseline TRUE velocity: {baseline_vel:.3f} m/s")
        print(f"  ✅ DR model TRUE velocity: {dr_vel:.3f} m/s")
        
        all_data.append({
            'label': label,
            'failure_rate': failure_rate,
            'baseline_traj': baseline_traj,
            'baseline_vel': baseline_vel,
            'dr_traj': dr_traj,
            'dr_vel': dr_vel
        })
        
        # Clean up environments
        baseline_env.close()
        dr_env.close()
    
    print("\n" + "="*60)
    print("PASS 2: Replaying trajectories WITH rendering...")
    print("="*60)
    
    all_frames = []
    
    for stage_data in all_data:
        print(f"\n🎥 Rendering {stage_data['label']}...")
        
        # Create environments WITH rendering for Pass 2
        def make_env_with_render():
            env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')
            env = SuccessRewardWrapper(env)
            return env
        
        baseline_env = DummyVecEnv([make_env_with_render])
        dr_env = DummyVecEnv([make_env_with_render])
        
        # Load VecNormalize if needed
        if os.path.exists(baseline_norm_path):
            baseline_env = VecNormalize.load(baseline_norm_path, baseline_env)
            baseline_env.training = False
            baseline_env.norm_reward = False
            
        if os.path.exists(dr_norm_path):
            dr_env = VecNormalize.load(dr_norm_path, dr_env)
            dr_env.training = False
            dr_env.norm_reward = False
        
        # Replay trajectories with rendering (Pass 2)
        baseline_frames = replay_trajectory_with_render(stage_data['baseline_traj'], baseline_env)
        dr_frames = replay_trajectory_with_render(stage_data['dr_traj'], dr_env)
        
        print(f"  Baseline frames: {len(baseline_frames)}")
        print(f"  DR model frames: {len(dr_frames)}")
        
        # Ensure same number of frames
        min_frames = min(len(baseline_frames), len(dr_frames))
        baseline_frames = baseline_frames[:min_frames]
        dr_frames = dr_frames[:min_frames]
        
        # Combine frames side by side with overlays
        for i, (bf, df) in enumerate(zip(baseline_frames, dr_frames)):
            # Resize frames if needed
            height = max(bf.shape[0], df.shape[0])
            
            if bf.shape[0] != height:
                bf = cv2.resize(bf, (bf.shape[1], height))
            if df.shape[0] != height:
                df = cv2.resize(df, (df.shape[1], height))
            
            # Create side-by-side frame
            combined = np.hstack([bf, df])
            
            # Add text overlays
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Main title
            cv2.putText(combined, stage_data['label'], 
                       (combined.shape[1]//2 - 200, 40),
                       font, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
            
            # Model labels
            cv2.putText(combined, "BASELINE PPO", 
                       (50, 80),
                       font, 0.8, (100, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(combined, "DR MODEL (30M steps)", 
                       (bf.shape[1] + 50, 80),
                       font, 0.8, (100, 255, 100), 2, cv2.LINE_AA)
            
            # True velocities from Pass 1
            cv2.putText(combined, f"Velocity: {stage_data['baseline_vel']:.3f} m/s", 
                       (50, 120),
                       font, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.putText(combined, f"Velocity: {stage_data['dr_vel']:.3f} m/s", 
                       (bf.shape[1] + 50, 120),
                       font, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            
            # Performance comparison
            if stage_data['dr_vel'] > 0 and stage_data['baseline_vel'] > 0:
                comparison = (stage_data['dr_vel'] / stage_data['baseline_vel']) * 100
                color = (100, 255, 100) if comparison > 100 else (255, 200, 100)
                cv2.putText(combined, f"DR Performance: {comparison:.1f}% of baseline", 
                           (combined.shape[1]//2 - 150, combined.shape[0] - 80),
                           font, 0.6, color, 2, cv2.LINE_AA)
            
            # Add progress bar showing failure rate
            bar_y = combined.shape[0] - 50
            bar_width = 300
            bar_x_start = combined.shape[1]//2 - bar_width//2
            
            # Draw progress bar background
            cv2.rectangle(combined, 
                         (bar_x_start, bar_y), 
                         (bar_x_start + bar_width, bar_y + 20),
                         (50, 50, 50), -1)
            
            # Draw progress bar fill
            fill_width = int(bar_width * stage_data['failure_rate'] / 0.30)
            # Color gradient from green to red
            green = int(255 * (1 - stage_data['failure_rate'] / 0.30))
            red = int(255 * stage_data['failure_rate'] / 0.30)
            color = (0, green, red)
            cv2.rectangle(combined, 
                         (bar_x_start, bar_y), 
                         (bar_x_start + fill_width, bar_y + 20),
                         color, -1)
            
            # Add failure rate text
            cv2.putText(combined, f"Joint Failure Rate: {stage_data['failure_rate']*100:.0f}%", 
                       (combined.shape[1]//2 - 100, bar_y - 10),
                       font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
            all_frames.append(combined)
        
        # Clean up
        baseline_env.close()
        dr_env.close()
    
    # Write video
    if all_frames:
        print(f"\n💾 Writing video to {output_path}...")
        height, width = all_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in all_frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"✅ Video saved: {output_path}")
        print(f"   Total frames: {len(all_frames)}")
        print(f"   Duration: {len(all_frames)/fps:.1f} seconds")
    else:
        print("❌ No frames captured!")
    
    print("\n" + "=" * 60)
    print("🎬 TWO-PASS VIDEO CREATION COMPLETE!")
    print("=" * 60)
    print(f"\nThis video shows TRUE performance metrics from Pass 1")
    print(f"combined with accurate visual rendering from Pass 2")
    print(f"Notice the realistic velocities now!")

if __name__ == "__main__":
    create_progressive_video_twopass()