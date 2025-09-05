#!/usr/bin/env python3
"""
Video showing PERMANENT joint failures (joints stay disabled once failed)
Note: Models were NOT trained for this - this shows how they handle 
permanent failures vs the temporary ones they were trained on
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

class PermanentJointFailureTracker:
    """Track PERMANENT joint failures - once failed, stays failed"""
    def __init__(self, num_joints=8):
        self.num_joints = num_joints
        self.joint_status = np.ones(num_joints)  # 1.0 = healthy, 0.0 = permanently failed
        self.joint_names = [f"J{i}" for i in range(num_joints)]
        self.failure_step = {}  # Track when each joint failed
        
    def apply_permanent_failures(self, action, failure_rate, step):
        """Apply PERMANENT joint failures - once failed, always failed"""
        action_copy = action.copy()
        
        # First, check for new failures
        if failure_rate > 0:
            for i in range(self.num_joints):
                # Only check healthy joints for new failures
                if self.joint_status[i] == 1.0:
                    if np.random.random() < failure_rate:
                        # PERMANENT FAILURE - joint is now broken forever
                        self.joint_status[i] = 0.0
                        self.failure_step[i] = step
        
        # Apply all permanent failures (both new and existing)
        for i in range(self.num_joints):
            if self.joint_status[i] == 0.0:
                # Joint is permanently disabled - zero torque
                action_copy[0][i] = 0.0
        
        return action_copy
    
    def get_status_bar(self, width=300, height=60):
        """Create a visual status bar showing permanent failures"""
        img = np.ones((height, width, 3), dtype=np.uint8) * 30  # Dark background
        
        bar_width = width // (self.num_joints + 1)
        bar_height = int(height * 0.6)
        y_start = int(height * 0.2)
        
        for i, status in enumerate(self.joint_status):
            x_start = int((i + 0.5) * bar_width)
            
            # Draw background
            cv2.rectangle(img, 
                         (x_start, y_start),
                         (x_start + int(bar_width * 0.8), y_start + bar_height),
                         (60, 60, 60), -1)
            
            # Draw status (green = working, red = permanently failed)
            if status > 0.5:
                color = (0, 255, 0)  # Green - working
                text = "OK"
            else:
                color = (0, 0, 255)  # Red - FAILED
                text = "FAIL"
            
            cv2.rectangle(img,
                         (x_start + 2, y_start + 2),
                         (x_start + int(bar_width * 0.8) - 2, y_start + bar_height - 2),
                         color, -1)
            
            # Add joint label
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, self.joint_names[i],
                       (x_start + 2, height - 5),
                       font, 0.3, (200, 200, 200), 1, cv2.LINE_AA)
            
            # Add status text
            cv2.putText(img, text,
                       (x_start + 2, y_start + bar_height - 5),
                       font, 0.25, (255, 255, 255), 1, cv2.LINE_AA)
        
        return img

def collect_trajectory_permanent_failures(model, env, failure_rate=0.0, steps=300):
    """Collect trajectory with PERMANENT joint failures"""
    trajectory = []
    velocities = []
    joint_tracker = PermanentJointFailureTracker()
    
    obs = env.reset()
    
    for step in range(steps):
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Apply PERMANENT joint failures
        action_with_failures = joint_tracker.apply_permanent_failures(action, failure_rate, step)
        
        # Store trajectory data
        trajectory.append({
            'obs': obs.copy(),
            'action': action.copy(),
            'action_with_failures': action_with_failures.copy(),
            'joint_status': joint_tracker.joint_status.copy()
        })
        
        # Step environment
        obs, reward, done, info = env.step(action_with_failures)
        
        # Track velocity
        vel = 0.0
        try:
            base_env = env.venv.envs[0]
            if hasattr(base_env, 'sim') and base_env.sim is not None:
                vel = base_env.sim.data.qvel[0]
        except:
            if info[0]:
                vel = info[0].get('current_velocity', info[0].get('speed', 0.0))
        
        velocities.append(vel)
        
        if done[0]:
            break
    
    avg_velocity = np.mean(velocities) if velocities else 0.0
    num_failed = (joint_tracker.joint_status == 0.0).sum()
    
    return trajectory, avg_velocity, num_failed, joint_tracker

def replay_trajectory_permanent(trajectory, env):
    """Replay trajectory with permanent failure visualization"""
    frames = []
    env.reset()
    
    for i, step_data in enumerate(trajectory):
        # Step with recorded action (with permanent failures applied)
        obs, reward, done, info = env.step(step_data['action_with_failures'])
        
        # Render frame
        frame = env.render(mode='rgb_array')
        
        if frame is not None:
            # Add joint status visualization
            joint_tracker = PermanentJointFailureTracker()
            joint_tracker.joint_status = step_data['joint_status']
            status_bar = joint_tracker.get_status_bar(300, 60)
            
            # Add status bar to frame (top of frame)
            if frame.shape[0] >= 60 and frame.shape[1] >= 300:
                # Center the status bar
                x_start = (frame.shape[1] - 300) // 2
                frame[10:70, x_start:x_start+300] = status_bar
            
            frames.append(frame)
        
        if done[0]:
            break
    
    return frames

def create_permanent_failure_video():
    """Create video showing PERMANENT joint failures"""
    
    print("=" * 60)
    print("🎬 PERMANENT JOINT FAILURE COMPARISON")
    print("=" * 60)
    print("\n⚠️  NOTE: Models were trained with TEMPORARY failures")
    print("This video shows how they handle PERMANENT failures")
    print("(different from their training!)\n")
    
    # Model paths
    baseline_path = 'done/ppo_baseline_ueqbjf2x/best_model/best_model.zip'
    dr_path = 'done/ppo_dr_gentle_v2_wptws01u/best_model/best_model.zip'
    
    # Load models
    print("📦 Loading models...")
    baseline_model = PPO.load(baseline_path)
    dr_model = PPO.load(dr_path)
    
    # Video settings
    fps = 30
    output_path = f'videos/permanent_failures_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
    os.makedirs('videos', exist_ok=True)
    
    # Test scenarios with PERMANENT failures
    scenarios = [
        (0.000, 300, "No Failures - Normal Operation"),
        (0.003, 300, "1% Chance Per Step - Gradual Permanent Failures"),
        (0.007, 300, "2% Chance Per Step - Moderate Permanent Failures"),
        (0.010, 300, "3% Chance Per Step - Severe Permanent Failures")
    ]
    
    all_data = []
    
    print("\n" + "="*60)
    print("PASS 1: Testing with PERMANENT joint failures...")
    print("="*60)
    
    for failure_rate, steps, label in scenarios:
        print(f"\n🔴 Testing: {label}")
        print(f"   (Once a joint fails, it stays broken forever)")
        
        # Create environments WITHOUT rendering
        def make_env():
            env = gym.make('RealAntMujoco-v0')
            env = SuccessRewardWrapper(env)
            return env
        
        baseline_env = DummyVecEnv([make_env])
        dr_env = DummyVecEnv([make_env])
        
        # Load VecNormalize
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
        
        # Collect trajectories with PERMANENT failures
        b_traj, b_vel, b_failed, b_tracker = collect_trajectory_permanent_failures(
            baseline_model, baseline_env, failure_rate, steps
        )
        d_traj, d_vel, d_failed, d_tracker = collect_trajectory_permanent_failures(
            dr_model, dr_env, failure_rate, steps
        )
        
        print(f"  Baseline: {b_vel:.3f} m/s, {b_failed} joints permanently failed")
        print(f"  DR model: {d_vel:.3f} m/s, {d_failed} joints permanently failed")
        
        all_data.append({
            'label': label,
            'failure_rate': failure_rate,
            'baseline_traj': b_traj,
            'baseline_vel': b_vel,
            'baseline_failed': b_failed,
            'dr_traj': d_traj,
            'dr_vel': d_vel,
            'dr_failed': d_failed
        })
        
        baseline_env.close()
        dr_env.close()
    
    print("\n" + "="*60)
    print("PASS 2: Creating video with permanent failure visualization...")
    print("="*60)
    
    all_frames = []
    
    for stage_data in all_data:
        print(f"\n🎥 Rendering: {stage_data['label']}")
        
        # Create environments WITH rendering
        def make_env_render():
            env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')
            env = SuccessRewardWrapper(env)
            return env
        
        baseline_env = DummyVecEnv([make_env_render])
        dr_env = DummyVecEnv([make_env_render])
        
        # Load VecNormalize
        if os.path.exists(baseline_norm_path):
            baseline_env = VecNormalize.load(baseline_norm_path, baseline_env)
            baseline_env.training = False
            baseline_env.norm_reward = False
            
        if os.path.exists(dr_norm_path):
            dr_env = VecNormalize.load(dr_norm_path, dr_env)
            dr_env.training = False
            dr_env.norm_reward = False
        
        # Replay with visualization
        baseline_frames = replay_trajectory_permanent(stage_data['baseline_traj'], baseline_env)
        dr_frames = replay_trajectory_permanent(stage_data['dr_traj'], dr_env)
        
        # Combine frames
        min_frames = min(len(baseline_frames), len(dr_frames))
        baseline_frames = baseline_frames[:min_frames]
        dr_frames = dr_frames[:min_frames]
        
        for i, (bf, df) in enumerate(zip(baseline_frames, dr_frames)):
            # Resize if needed
            height = max(bf.shape[0], df.shape[0])
            if bf.shape[0] != height:
                bf = cv2.resize(bf, (bf.shape[1], height))
            if df.shape[0] != height:
                df = cv2.resize(df, (df.shape[1], height))
            
            # Create side-by-side
            combined = np.hstack([bf, df])
            
            # Add overlays
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Title
            cv2.putText(combined, stage_data['label'], 
                       (combined.shape[1]//2 - 250, 40),
                       font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Warning that this is different from training
            if stage_data['failure_rate'] > 0:
                cv2.putText(combined, "⚠️ PERMANENT FAILURES (Models trained on TEMPORARY failures)", 
                           (combined.shape[1]//2 - 300, 70),
                           font, 0.5, (100, 200, 255), 1, cv2.LINE_AA)
            
            # Model labels with performance
            cv2.putText(combined, f"BASELINE: {stage_data['baseline_vel']:.2f} m/s | {stage_data['baseline_failed']} failed joints", 
                       (30, combined.shape[0] - 30),
                       font, 0.6, (100, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(combined, f"DR MODEL: {stage_data['dr_vel']:.2f} m/s | {stage_data['dr_failed']} failed joints", 
                       (bf.shape[1] + 30, combined.shape[0] - 30),
                       font, 0.6, (100, 255, 100), 2, cv2.LINE_AA)
            
            # Progress bar
            progress = (i + 1) / len(baseline_frames)
            bar_width = 400
            bar_x = combined.shape[1]//2 - 200
            bar_y = combined.shape[0] - 10
            cv2.rectangle(combined, (bar_x, bar_y), (bar_x + bar_width, bar_y + 5), (50, 50, 50), -1)
            cv2.rectangle(combined, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + 5), (0, 255, 100), -1)
            
            all_frames.append(combined)
        
        baseline_env.close()
        dr_env.close()
    
    # Write video
    if all_frames:
        print(f"\n💾 Writing video to {output_path}...")
        height, width = all_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in all_frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"✅ Video saved: {output_path}")
        print(f"   Duration: {len(all_frames)/fps:.1f} seconds")
        
        print("\n" + "="*60)
        print("⚠️  IMPORTANT NOTE:")
        print("="*60)
        print("This video shows PERMANENT joint failures")
        print("The models were trained with TEMPORARY failures")
        print("For proper PERMANENT failure handling, we would need to:")
        print("1. Retrain with permanent failure scenarios")
        print("2. Use curriculum learning (start with 1 failure, increase gradually)")
        print("3. Train for adaptive gaits with missing joints")
    
    print("\n🎉 VIDEO COMPLETE!")

if __name__ == "__main__":
    create_permanent_failure_video()