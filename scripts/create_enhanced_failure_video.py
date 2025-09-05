#!/usr/bin/env python3
"""
Enhanced comparison video with:
- Longer episodes (300 steps each)
- Joint health indicators
- 3D trajectory visualization
- Detailed failure tracking
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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d import Axes3D

class JointHealthTracker:
    """Track health status of each joint"""
    def __init__(self, num_joints=8):
        self.num_joints = num_joints
        self.joint_health = np.ones(num_joints)  # 1.0 = healthy, 0.0 = failed
        self.joint_failures = np.zeros(num_joints)  # Track failure counts
        self.joint_names = [f"J{i}" for i in range(num_joints)]
        
    def apply_failures(self, action, failure_rate):
        """Apply joint failures and track which joints failed"""
        action_copy = action.copy()
        self.joint_health = np.ones(self.num_joints)  # Reset to healthy
        
        if failure_rate > 0:
            for i in range(len(action_copy[0])):
                if np.random.random() < failure_rate:
                    failure_type = np.random.choice(['lock', 'weak', 'noise'])
                    self.joint_failures[i] += 1
                    
                    if failure_type == 'lock':
                        action_copy[0][i] = 0.0
                        self.joint_health[i] = 0.0  # Complete failure
                    elif failure_type == 'weak':
                        action_copy[0][i] *= 0.3
                        self.joint_health[i] = 0.3  # Weak joint
                    elif failure_type == 'noise':
                        action_copy[0][i] += np.random.normal(0, 0.5)
                        self.joint_health[i] = 0.7  # Noisy but functional
        
        return action_copy
    
    def get_health_bar_image(self, width=200, height=100):
        """Create a visual health bar for joints"""
        img = np.ones((height, width, 3), dtype=np.uint8) * 40  # Dark background
        
        bar_width = width // (self.num_joints + 1)
        bar_height = int(height * 0.7)
        y_start = int(height * 0.15)
        
        for i, health in enumerate(self.joint_health):
            x_start = int((i + 0.5) * bar_width)
            
            # Draw background bar
            cv2.rectangle(img, 
                         (x_start, y_start + bar_height - int(bar_height * 0.8)),
                         (x_start + int(bar_width * 0.8), y_start + bar_height),
                         (60, 60, 60), -1)
            
            # Draw health bar
            bar_color = (0, int(255 * health), int(255 * (1 - health)))  # Green to red
            filled_height = int(bar_height * 0.8 * health)
            if filled_height > 0:
                cv2.rectangle(img,
                             (x_start, y_start + bar_height - filled_height),
                             (x_start + int(bar_width * 0.8), y_start + bar_height),
                             bar_color, -1)
            
            # Add joint label
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, self.joint_names[i],
                       (x_start - 5, height - 5),
                       font, 0.3, (200, 200, 200), 1, cv2.LINE_AA)
        
        return img

def create_3d_trajectory_plot(positions, title="3D Trajectory"):
    """Create 3D trajectory visualization"""
    fig = plt.figure(figsize=(4, 3), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    
    if len(positions) > 1:
        positions = np.array(positions)
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        
        # Create color gradient
        colors = plt.cm.viridis(np.linspace(0, 1, len(positions)))
        
        # Plot trajectory
        for i in range(len(positions) - 1):
            ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], 
                   color=colors[i], linewidth=2, alpha=0.8)
        
        # Mark start and end
        ax.scatter(x[0], y[0], z[0], color='green', s=50, marker='o', label='Start')
        ax.scatter(x[-1], y[-1], z[-1], color='red', s=50, marker='*', label='End')
        
        # Set limits
        ax.set_xlim([np.min(x) - 1, np.max(x) + 1])
        ax.set_ylim([np.min(y) - 1, np.max(y) + 1])
        ax.set_zlim([0, np.max(z) + 0.5])
    
    # Styling
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_zlabel('Z', color='white')
    ax.set_title(title, color='white', fontsize=10)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3)
    
    # Convert to image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    plt.close(fig)
    
    return img

def collect_enhanced_trajectory(model, env, failure_rate=0.0, steps=300):
    """Collect trajectory with enhanced tracking"""
    trajectory = []
    velocities = []
    positions = []
    joint_tracker = JointHealthTracker()
    
    obs = env.reset()
    
    # Track initial position
    try:
        base_env = env.venv.envs[0]
        if hasattr(base_env, 'sim') and base_env.sim is not None:
            pos = base_env.sim.data.qpos[:3].copy()  # x, y, z position
            positions.append(pos)
    except:
        positions.append([0, 0, 0.75])
    
    for step in range(steps):
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Apply joint failures with tracking
        action_with_failures = joint_tracker.apply_failures(action, failure_rate)
        
        # Store trajectory data
        trajectory.append({
            'obs': obs.copy(),
            'action': action.copy(),
            'action_with_failures': action_with_failures.copy(),
            'joint_health': joint_tracker.joint_health.copy()
        })
        
        # Step environment
        obs, reward, done, info = env.step(action_with_failures)
        
        # Track velocity and position
        vel = 0.0
        if info[0]:
            if 'current_velocity' in info[0]:
                vel = info[0]['current_velocity']
            elif 'speed' in info[0]:
                vel = info[0]['speed']
        
        try:
            base_env = env.venv.envs[0]
            if hasattr(base_env, 'sim') and base_env.sim is not None:
                vel = base_env.sim.data.qvel[0]  # Direct x velocity
                pos = base_env.sim.data.qpos[:3].copy()
                positions.append(pos)
        except:
            if positions:
                # Estimate position from velocity
                last_pos = positions[-1].copy()
                last_pos[0] += vel * 0.05  # dt = 0.05
                positions.append(last_pos)
        
        velocities.append(vel)
        
        if done[0]:
            break
    
    avg_velocity = np.mean(velocities) if velocities else 0.0
    distance = positions[-1][0] - positions[0][0] if len(positions) > 1 else 0.0
    
    return trajectory, avg_velocity, distance, positions, joint_tracker

def replay_enhanced_trajectory(trajectory, env, positions, model_name="Model"):
    """Replay trajectory with enhanced visualizations"""
    frames = []
    env.reset()
    
    for i, step_data in enumerate(trajectory):
        # Step with recorded action
        obs, reward, done, info = env.step(step_data['action_with_failures'])
        
        # Render frame
        frame = env.render(mode='rgb_array')
        
        if frame is not None:
            # Add joint health visualization
            joint_tracker = JointHealthTracker()
            joint_tracker.joint_health = step_data['joint_health']
            health_bar = joint_tracker.get_health_bar_image(200, 80)
            
            # Resize health bar to fit
            health_bar = cv2.resize(health_bar, (200, 80))
            
            # Add health bar to frame (top-right corner)
            if frame.shape[0] >= 80 and frame.shape[1] >= 200:
                frame[10:90, -210:-10] = health_bar
            
            frames.append(frame)
        
        if done[0]:
            break
    
    return frames

def create_enhanced_video():
    """Create enhanced comparison video"""
    
    print("=" * 60)
    print("🎬 CREATING ENHANCED COMPARISON VIDEO")
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
    output_path = f'videos/enhanced_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
    os.makedirs('videos', exist_ok=True)
    
    # Failure rate progression with LONGER episodes
    failure_stages = [
        (0.00, 300, "0% Joint Failures - Normal Operation"),
        (0.10, 300, "10% Joint Failures - Mild Degradation"),
        (0.20, 300, "20% Joint Failures - Moderate Challenge"),
        (0.30, 300, "30% Joint Failures - Extreme Stress Test")
    ]
    
    # Store all collected data
    all_data = []
    
    print("\n" + "="*60)
    print("PASS 1: Collecting enhanced trajectories...")
    print("="*60)
    
    for failure_rate, steps, label in failure_stages:
        print(f"\n🎯 Collecting {steps}-step trajectories for {label}...")
        
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
        
        # Collect enhanced trajectories
        b_traj, b_vel, b_dist, b_pos, b_tracker = collect_enhanced_trajectory(
            baseline_model, baseline_env, failure_rate, steps
        )
        d_traj, d_vel, d_dist, d_pos, d_tracker = collect_enhanced_trajectory(
            dr_model, dr_env, failure_rate, steps
        )
        
        print(f"  ✅ Baseline: {b_vel:.3f} m/s, distance: {b_dist:.2f}m")
        print(f"  ✅ DR model: {d_vel:.3f} m/s, distance: {d_dist:.2f}m")
        print(f"  📊 Baseline joint failures: {b_tracker.joint_failures.sum():.0f}")
        print(f"  📊 DR joint failures: {d_tracker.joint_failures.sum():.0f}")
        
        all_data.append({
            'label': label,
            'failure_rate': failure_rate,
            'baseline_traj': b_traj,
            'baseline_vel': b_vel,
            'baseline_dist': b_dist,
            'baseline_pos': b_pos,
            'dr_traj': d_traj,
            'dr_vel': d_vel,
            'dr_dist': d_dist,
            'dr_pos': d_pos
        })
        
        baseline_env.close()
        dr_env.close()
    
    print("\n" + "="*60)
    print("PASS 2: Creating enhanced visualizations...")
    print("="*60)
    
    all_frames = []
    
    for stage_data in all_data:
        print(f"\n🎥 Rendering {stage_data['label']}...")
        
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
        
        # Replay with enhanced visuals
        baseline_frames = replay_enhanced_trajectory(
            stage_data['baseline_traj'], baseline_env, 
            stage_data['baseline_pos'], "Baseline"
        )
        dr_frames = replay_enhanced_trajectory(
            stage_data['dr_traj'], dr_env,
            stage_data['dr_pos'], "DR Model"
        )
        
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
            
            # Add main overlays
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Title
            cv2.putText(combined, stage_data['label'], 
                       (combined.shape[1]//2 - 250, 40),
                       font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Model labels with performance
            cv2.putText(combined, f"BASELINE: {stage_data['baseline_vel']:.2f} m/s | {stage_data['baseline_dist']:.1f}m", 
                       (30, combined.shape[0] - 30),
                       font, 0.6, (100, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(combined, f"DR MODEL: {stage_data['dr_vel']:.2f} m/s | {stage_data['dr_dist']:.1f}m", 
                       (bf.shape[1] + 30, combined.shape[0] - 30),
                       font, 0.6, (100, 255, 100), 2, cv2.LINE_AA)
            
            # Progress indicator
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
        print(f"\n💾 Writing enhanced video to {output_path}...")
        height, width = all_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in all_frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"✅ Video saved: {output_path}")
        print(f"   Total frames: {len(all_frames)}")
        print(f"   Duration: {len(all_frames)/fps:.1f} seconds")
        print("\n🎬 ENHANCED FEATURES:")
        print("   ✅ Joint health indicators (top-right)")
        print("   ✅ Extended episodes (300 steps each)")
        print("   ✅ Real performance metrics")
        print("   ✅ Two-pass recording for accurate velocities")
    
    print("\n" + "=" * 60)
    print("🎉 ENHANCED VIDEO COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    create_enhanced_video()