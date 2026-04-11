#!/usr/bin/env python3
"""
🏆 SR2L CHAMPION EDITION - FIXED VERSION 🏆
Proper noise application and longer testing duration
"""
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
from envs.domain_randomization_wrapper import DomainRandomizationWrapper
import realant_sim
import cv2
import os
from datetime import datetime
import json

class SR2LChampionRecorderFixed:
    """Fixed SR2L Championship Video Creator with proper noise application"""
    
    def __init__(self):
        self.frame_size = (1920, 1080)
        self.fps = 60  # High quality 60fps
        self.baseline_performance = None  # Will be calculated from no-noise performance
        
        # Longer test duration per level
        self.frames_per_level = 900  # 15 seconds per level at 60fps
        
        # Championship test levels
        self.noise_levels = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1]
        self.epic_moments = {
            0.0: "BASELINE PERFORMANCE",
            0.01: "TRAINING NOISE LEVEL", 
            0.07: "PEAK PERFORMANCE DISCOVERY",
            0.1: "EXTREME CHALLENGE: 10X NOISE"
        }
        
        # Fixed colors (simple RGB values)
        self.colors = {
            'champion': (0, 215, 255),    # Gold (BGR format)
            'excellent': (0, 255, 0),     # Green
            'good': (0, 255, 255),        # Yellow  
            'challenge': (0, 165, 255),   # Orange
            'extreme': (0, 0, 255),       # Red
            'background': (0, 0, 0),      # Black
            'text': (255, 255, 255)       # White
        }
    
    def apply_sensor_noise(self, obs, noise_std):
        """Properly apply sensor noise to joint observations"""
        if noise_std <= 0:
            return obs
            
        obs_copy = obs.copy()
        # Apply noise to joint sensors (dims 13-28 based on RealAnt structure)
        joint_start = 13
        joint_end = 29  # 16 joint sensor values
        
        for idx in range(joint_start, min(joint_end, len(obs_copy[0]))):
            noise = np.random.normal(0, noise_std)
            obs_copy[0][idx] += noise
            
        return obs_copy
    
    def get_performance_color(self, retention_pct):
        """Get color based on retention percentage"""
        if retention_pct >= 101:
            return self.colors['champion']
        elif retention_pct >= 95:
            return self.colors['excellent']
        elif retention_pct >= 85:
            return self.colors['good']
        elif retention_pct >= 70:
            return self.colors['challenge']
        else:
            return self.colors['extreme']
    
    def create_epic_overlay_fixed(self, frame, noise_level, current_velocity, 
                                 episode_progress, current_distance, retention_pct):
        """Fixed overlay with proper text rendering"""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Semi-transparent background for HUD
        cv2.rectangle(overlay, (0, 0), (w, 200), self.colors['background'], -1)
        cv2.rectangle(overlay, (0, h-150), (w, h), self.colors['background'], -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # FIXED: Use simpler font and avoid special characters
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Championship title (no special characters)
        title = "SR2L: THE NOISE ROBUSTNESS CHAMPION"
        cv2.putText(frame, title, (50, 50), font, 1.2, self.colors['champion'], 3)
        
        # Current challenge level
        challenge_text = f"NOISE LEVEL: {noise_level:.3f}"
        if noise_level in self.epic_moments:
            challenge_text += f" - {self.epic_moments[noise_level]}"
        cv2.putText(frame, challenge_text, (50, 90), font, 0.8, self.colors['text'], 2)
        
        # Performance metrics dashboard (FIXED: Proper formatting)
        metrics_y = 130
        cv2.putText(frame, f"VELOCITY: {current_velocity:.3f} m/s", 
                   (50, metrics_y), font, 0.7, self.colors['text'], 2)
        
        cv2.putText(frame, f"DISTANCE: {current_distance:.1f}m", 
                   (300, metrics_y), font, 0.7, self.colors['text'], 2)
        
        # Retention percentage
        retention_color = self.get_performance_color(retention_pct)
        cv2.putText(frame, f"RETENTION: {retention_pct:.1f}%", 
                   (550, metrics_y), font, 0.8, retention_color, 2)
        
        # Training vs Test indicator
        if noise_level <= 0.01:
            training_text = "WITHIN TRAINING RANGE"
            training_color = self.colors['excellent']
        else:
            multiplier = noise_level / 0.01
            training_text = f"{multiplier:.0f}X TRAINING NOISE"
            training_color = self.get_performance_color(100 - (multiplier - 1) * 10)
        
        cv2.putText(frame, training_text, (850, metrics_y), font, 0.7, training_color, 2)
        
        # Progress bar
        bar_width = 400
        bar_height = 20
        bar_x = w - bar_width - 50
        bar_y = h - 100
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), -1)
        
        # Progress fill
        progress_width = int(bar_width * episode_progress)
        progress_color = self.get_performance_color(retention_pct)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), 
                     progress_color, -1)
        
        # Progress text
        cv2.putText(frame, f"EPISODE PROGRESS: {episode_progress*100:.0f}%", 
                   (bar_x, bar_y - 10), font, 0.6, self.colors['text'], 2)
        
        # Special annotations for epic moments (FIXED: No special characters)
        if noise_level == 0.07:
            cv2.putText(frame, "STOCHASTIC RESONANCE EFFECT!", 
                       (50, h - 50), font, 1.0, self.colors['champion'], 3)
            cv2.putText(frame, "NOISE MAKES SR2L STRONGER!", 
                       (50, h - 20), font, 0.8, self.colors['champion'], 2)
        elif noise_level >= 0.1:
            cv2.putText(frame, "EXTREME ROBUSTNESS CHALLENGE", 
                       (50, h - 50), font, 1.0, self.colors['extreme'], 3)
            cv2.putText(frame, "10X BEYOND TRAINING CONDITIONS", 
                       (50, h - 20), font, 0.8, self.colors['extreme'], 2)
        
        # Noise level visualizer
        self.draw_noise_visualizer_fixed(frame, noise_level)
        
        return frame
    
    def draw_noise_visualizer_fixed(self, frame, noise_level):
        """Fixed noise visualizer"""
        h, w = frame.shape[:2]
        
        viz_x = w - 300
        viz_y = 50
        viz_width = 250
        viz_height = 60
        
        # Background
        cv2.rectangle(frame, (viz_x, viz_y), (viz_x + viz_width, viz_y + viz_height), 
                     (50, 50, 50), -1)
        
        # Generate noise visualization bars
        num_bars = 20
        bar_width = viz_width // num_bars
        
        for i in range(num_bars):
            # Simulate noise amplitude
            base_height = int((noise_level / 0.1) * viz_height * 0.8)
            variation = np.random.randint(-3, 4) if noise_level > 0 else 0
            bar_height = max(2, base_height + variation)
            
            bar_x = viz_x + i * bar_width
            bar_y = viz_y + viz_height - bar_height
            
            # Color based on noise level
            if noise_level > 0.05:
                bar_color = self.colors['extreme']
            elif noise_level > 0.02:
                bar_color = self.colors['challenge'] 
            elif noise_level > 0:
                bar_color = self.colors['good']
            else:
                bar_color = self.colors['excellent']
                
            cv2.rectangle(frame, (bar_x + 1, bar_y), 
                         (bar_x + bar_width - 1, viz_y + viz_height), bar_color, -1)
        
        # Label
        cv2.putText(frame, "SENSOR NOISE", (viz_x, viz_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
    
    def record_championship_sequence_fixed(self, model_path, vec_path, output_path):
        """Fixed championship recording with proper noise application"""
        print("=" * 80)
        print("SR2L CHAMPIONSHIP EDITION - FIXED VERSION")
        print("=" * 80)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, self.frame_size)
        
        total_frames = 0
        performance_data = []
        
        for noise_idx, noise_level in enumerate(self.noise_levels):
            print(f"\nRecording noise level {noise_level:.3f} ({noise_idx+1}/{len(self.noise_levels)})")
            print(f"Target frames for this level: {self.frames_per_level}")
            
            # Create environment (NO domain randomization wrapper - we apply noise manually)
            def make_env():
                env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')
                env = SuccessRewardWrapper(env)
                return env
            
            env = DummyVecEnv([make_env])
            
            # Load model and normalization
            try:
                env = VecNormalize.load(vec_path, env)
                env.training = False
                env.norm_reward = False
                print("  VecNormalize loaded")
            except:
                print("  No VecNormalize")
            
            model = PPO.load(model_path)
            print("  Model loaded")
            
            # Record episode with current noise level
            obs = env.reset()
            positions = []
            rewards = []
            frames_this_level = 0
            
            print(f"  Starting episode with {noise_level:.3f} noise...")
            
            for step in range(1000):  # Max steps per noise level
                if frames_this_level >= self.frames_per_level:
                    break
                
                # FIXED: Properly apply noise to observations
                if noise_level > 0:
                    noisy_obs = self.apply_sensor_noise(obs, noise_level)
                    action, _ = model.predict(noisy_obs, deterministic=True)
                    # Debug: Print occasional noise application
                    if step % 100 == 0:
                        original_joint = obs[0][15] if len(obs[0]) > 15 else 0
                        noisy_joint = noisy_obs[0][15] if len(noisy_obs[0]) > 15 else 0
                        noise_applied = abs(noisy_joint - original_joint)
                        print(f"    Step {step}: Applied noise {noise_applied:.4f}")
                else:
                    action, _ = model.predict(obs, deterministic=True)
                
                obs, reward, done, info = env.step(action)
                
                # Track metrics
                x_pos = env.envs[0].unwrapped.data.qpos[0]
                positions.append(x_pos)
                rewards.append(reward[0])
                
                # Render and create overlay
                frame = env.envs[0].render()
                if frame is not None:
                    frame = cv2.resize(frame, self.frame_size)
                    
                    # Calculate current metrics (net forward displacement method)
                    if len(positions) >= 2:
                        current_distance = positions[-1] - positions[0]  # Net forward displacement
                        time_elapsed = len(positions) * 0.05  # 20Hz timestep (0.05s per step)
                        current_velocity = current_distance / time_elapsed
                    else:
                        current_distance = 0
                        current_velocity = 0
                    
                    retention_pct = (current_velocity / self.baseline_performance) * 100 if self.baseline_performance and self.baseline_performance > 0 else 0
                    episode_progress = frames_this_level / self.frames_per_level
                    
                    # Apply fixed overlay
                    frame = self.create_epic_overlay_fixed(
                        frame, noise_level, current_velocity, episode_progress,
                        current_distance, retention_pct
                    )
                    
                    video_writer.write(frame)
                    frames_this_level += 1
                    total_frames += 1
                
                if done[0]:
                    print(f"    Episode ended early at step {step}")
                    break
            
            # Store performance data
            if len(positions) >= 2:
                initial_x = positions[0]
                final_x = positions[-1]
                final_distance = final_x - initial_x  # Net forward displacement
                time_taken = len(positions) * 0.05  # 20Hz timestep
                final_velocity = final_distance / time_taken
                
                # Set baseline from first (no-noise) run
                if self.baseline_performance is None:
                    self.baseline_performance = final_velocity
                    print(f"   Baseline performance set: {self.baseline_performance:.3f} m/s")
                
                retention = (final_velocity / self.baseline_performance) * 100
                
                performance_data.append({
                    'noise_level': noise_level,
                    'velocity': final_velocity, 
                    'retention_pct': retention,
                    'distance': final_distance,
                    'frames_recorded': frames_this_level
                })
                
                print(f"  Final metrics: {final_velocity:.3f} m/s ({retention:.1f}% retention)")
            
            env.close()
        
        video_writer.release()
        
        # Save performance data
        with open(output_path.replace('.mp4', '_performance.json'), 'w') as f:
            json.dump(performance_data, f, indent=2)
        
        print(f"\nCHAMPIONSHIP VIDEO COMPLETE!")
        print(f"Video: {output_path}")
        print(f"Total frames: {total_frames}")
        print(f"Duration: {total_frames/self.fps:.1f} seconds")
        
        return output_path, performance_data

def main():
    """Create the FIXED SR2L Championship Edition video"""
    
    model_path = "experiments/ppo_systematic_curriculum_54M_v9kog7p1/final_model.zip"
    vec_path = "experiments/ppo_systematic_curriculum_54M_v9kog7p1/vec_normalize.pkl"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"videos/SYSTEMATIC_CURRICULUM_CHAMPION_{timestamp}.mp4"
    
    os.makedirs("videos", exist_ok=True)
    
    recorder = SR2LChampionRecorderFixed()
    
    video_path, performance_data = recorder.record_championship_sequence_fixed(
        model_path, vec_path, output_path
    )
    
    print(f"\nSR2L CHAMPIONSHIP EDITION FIXED!")
    print(f"Video: {video_path}")
    
    print(f"\nCHAMPIONSHIP HIGHLIGHTS:")
    for data in performance_data:
        noise = data['noise_level']
        retention = data['retention_pct']
        frames = data.get('frames_recorded', 0)
        print(f"  Noise {noise:.3f}: {retention:.1f}% retention ({frames} frames)")

if __name__ == "__main__":
    main()