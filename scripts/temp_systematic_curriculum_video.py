#!/usr/bin/env python3
"""
🏆 SYSTEMATIC CURRICULUM CHAMPION - TEMP VIDEO 🏆
Based on working SR2L Championship script pattern
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

class SystematicCurriculumChampionRecorder:
    """Systematic Curriculum Championship Video Creator"""
    
    def __init__(self):
        self.frame_size = (1920, 1080)
        self.fps = 60  # High quality 60fps
        self.baseline_performance = None
        
        # Test duration per scenario
        self.frames_per_level = 900  # 15 seconds per scenario at 60fps
        
        # Joint failure scenarios
        self.failure_scenarios = [
            {'name': 'BASELINE', 'config': None, 'epic': 'NO FAILURES'},
            {'name': 'HIP_FAILURE', 'config': {'joint_dropout_prob': 1.0, 'max_dropped_joints': 1, 'min_dropped_joints': 1, 'sensor_noise_std': 0.0}, 'epic': 'SINGLE HIP JOINT FAILURE'},
            {'name': 'ANKLE_FAILURE', 'config': {'joint_dropout_prob': 1.0, 'max_dropped_joints': 1, 'min_dropped_joints': 1, 'sensor_noise_std': 0.0}, 'epic': 'SINGLE ANKLE JOINT FAILURE'},
            {'name': 'DUAL_FAILURE', 'config': {'joint_dropout_prob': 1.0, 'max_dropped_joints': 2, 'min_dropped_joints': 2, 'sensor_noise_std': 0.0}, 'epic': 'DUAL JOINT FAILURES'}
        ]
        
        # Colors (BGR format)
        self.colors = {
            'champion': (0, 215, 255),    # Gold
            'excellent': (0, 255, 0),     # Green
            'good': (0, 255, 255),        # Yellow  
            'challenge': (0, 165, 255),   # Orange
            'extreme': (0, 0, 255),       # Red
            'background': (0, 0, 0),      # Black
            'text': (255, 255, 255)       # White
        }
    
    def get_performance_color(self, distance):
        """Get color based on distance performance"""
        if distance >= 8:
            return self.colors['champion']
        elif distance >= 5:
            return self.colors['excellent']
        elif distance >= 2:
            return self.colors['good']
        elif distance >= 1:
            return self.colors['challenge']
        else:
            return self.colors['extreme']
    
    def create_epic_overlay_fixed(self, frame, scenario_name, epic_moment, 
                                 current_velocity, current_distance, progress):
        """Create overlay with systematic curriculum info"""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Semi-transparent background for HUD
        cv2.rectangle(overlay, (0, 0), (w, 200), self.colors['background'], -1)
        cv2.rectangle(overlay, (0, h-150), (w, h), self.colors['background'], -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Championship title
        title = "SYSTEMATIC CURRICULUM: JOINT FAILURE CHAMPION"
        cv2.putText(frame, title, (50, 50), font, 1.2, self.colors['champion'], 3)
        
        # Current test scenario
        challenge_text = f"TEST: {scenario_name}"
        cv2.putText(frame, challenge_text, (50, 90), font, 0.8, self.colors['text'], 2)
        
        # Epic moment description
        cv2.putText(frame, epic_moment, (50, 130), font, 0.7, self.colors['challenge'], 2)
        
        # Performance metrics
        metrics_y = 170
        cv2.putText(frame, f"VELOCITY: {current_velocity:.3f} m/s", 
                   (50, metrics_y), font, 0.7, self.colors['text'], 2)
        
        cv2.putText(frame, f"DISTANCE: {current_distance:.1f}m", 
                   (300, metrics_y), font, 0.7, self.colors['text'], 2)
        
        # Performance assessment
        perf_color = self.get_performance_color(current_distance)
        if current_distance >= 5:
            assessment = "EXCELLENT"
        elif current_distance >= 2:
            assessment = "GOOD"
        elif current_distance >= 1:
            assessment = "POOR"
        else:
            assessment = "FAILED"
        
        cv2.putText(frame, assessment, (550, metrics_y), font, 0.8, perf_color, 2)
        
        # Progress bar
        bar_width = 400
        bar_height = 20
        bar_x = w - bar_width - 50
        bar_y = h - 100
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), -1)
        
        # Progress fill
        progress_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), 
                     self.colors['champion'], -1)
        
        # Progress text
        cv2.putText(frame, f"PROGRESS: {int(progress * 100)}%", 
                   (bar_x, bar_y - 10), font, 0.6, self.colors['text'], 2)
        
        return frame
    
    def record_championship_sequence_fixed(self, model_path, vec_path, output_path):
        """Record systematic curriculum championship sequence"""
        print("=" * 80)
        print("SYSTEMATIC CURRICULUM CHAMPIONSHIP EDITION")
        print("=" * 80)
        
        # Try different codecs that work better on macOS
        codecs_to_try = [
            cv2.VideoWriter_fourcc(*'mp4v'),
            cv2.VideoWriter_fourcc(*'XVID'),
            cv2.VideoWriter_fourcc(*'MJPG'),
            cv2.VideoWriter_fourcc('M','J','P','G')
        ]
        
        video_writer = None
        for codec in codecs_to_try:
            video_writer = cv2.VideoWriter(output_path, codec, self.fps, self.frame_size)
            if video_writer.isOpened():
                print(f"  ✅ Video codec working")
                break
            video_writer.release()
        
        if not video_writer or not video_writer.isOpened():
            print("❌ All video codecs failed, trying fallback...")
            # Fallback to simple AVI
            output_path = output_path.replace('.mp4', '.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, self.frame_size)
        
        total_frames = 0
        performance_data = []
        
        for scenario_idx, scenario in enumerate(self.failure_scenarios):
            print(f"\nRecording scenario {scenario_idx+1}: {scenario['name']}")
            print(f"Epic moment: {scenario['epic']}")
            
            # Create environment
            def make_env():
                env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')
                env = SuccessRewardWrapper(env)
                if scenario['config']:
                    env = DomainRandomizationWrapper(env, scenario['config'])
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
            
            # Record scenario
            obs = env.reset()
            positions = []
            frames_this_scenario = 0
            
            print(f"  Starting scenario recording...")
            
            for step in range(1000):  # Max steps per scenario
                if frames_this_scenario >= self.frames_per_level:
                    break
                
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                
                # Track metrics
                x_pos = env.envs[0].unwrapped.data.qpos[0]
                positions.append(x_pos)
                
                # Calculate current performance
                if len(positions) >= 2:
                    current_distance = sum(abs(positions[i] - positions[i-1]) 
                                         for i in range(1, len(positions)))
                    current_velocity = current_distance / (len(positions) * 0.05) if len(positions) > 0 else 0.0
                else:
                    current_distance = 0.0
                    current_velocity = 0.0
                
                # Render and create overlay
                frame = env.envs[0].render()
                if frame is not None:
                    progress = frames_this_scenario / self.frames_per_level
                    overlay_frame = self.create_epic_overlay_fixed(
                        frame, scenario['name'], scenario['epic'], 
                        current_velocity, current_distance, progress
                    )
                    
                    success = video_writer.write(overlay_frame)
                    if success:
                        frames_this_scenario += 1
                        total_frames += 1
                    else:
                        print(f"    ⚠️  Frame write failed at {frames_this_scenario}")
                
                if done[0]:
                    obs = env.reset()
                    # Keep going until we hit frame limit
            
            env.close()
            
            # Final performance for this scenario
            if len(positions) >= 2:
                total_distance = sum(abs(positions[i] - positions[i-1]) 
                                   for i in range(1, len(positions)))
                avg_velocity = total_distance / (len(positions) * 0.05) if len(positions) > 0 else 0.0
            else:
                total_distance = 0.0
                avg_velocity = 0.0
            
            performance_data.append({
                'scenario': scenario['name'],
                'distance': total_distance,
                'velocity': avg_velocity,
                'assessment': 'EXCELLENT' if total_distance >= 5 else 
                             'GOOD' if total_distance >= 2 else 
                             'POOR' if total_distance >= 1 else 'FAILED'
            })
            
            print(f"  Performance: {total_distance:.1f}m ({avg_velocity:.3f} m/s)")
        
        video_writer.release()
        
        # Save performance data
        perf_output = output_path.replace('.mp4', '_performance.json')
        with open(perf_output, 'w') as f:
            json.dump(performance_data, f, indent=2)
        
        print(f"\n✅ Video created: {output_path}")
        print(f"📊 Performance data: {perf_output}")
        print(f"🎬 Total frames: {total_frames}")
        
        return output_path

def main():
    # Systematic curriculum model paths
    model_path = 'experiments/ppo_systematic_curriculum_54M_v9kog7p1/final_model.zip'
    vec_path = 'experiments/ppo_systematic_curriculum_54M_v9kog7p1/vec_normalize.pkl'
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'videos/SYSTEMATIC_CURRICULUM_CHAMPION_{timestamp}.mp4'
    
    recorder = SystematicCurriculumChampionRecorder()
    recorder.record_championship_sequence_fixed(model_path, vec_path, output_path)

if __name__ == "__main__":
    main()