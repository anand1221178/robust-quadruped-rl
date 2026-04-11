#!/usr/bin/env python3
"""
🏆 JUDGMENT DAY DEMONSTRATION VIDEO 🏆

Professional simulation demonstration video for thesis defense
Shows your trained PPO policy performing in MuJoCo with:
- Championship-style overlays
- Performance metrics
- Professional presentation quality

Physical robot will be demonstrated LIVE during presentation.
This video shows the simulation baseline performance.
"""
import sys
sys.path.append('src')

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
import realant_sim
import cv2
import os
from datetime import datetime
import json

class JudgmentDayRecorder:
    """Professional simulation demonstration video creator"""

    def __init__(self):
        self.frame_size = (1920, 1080)
        self.fps = 60

        # Video structure
        self.intro_frames = 180  # 3 seconds
        self.main_demo_frames = 3000  # 50 seconds of walking
        self.outro_frames = 180  # 3 seconds

        # Colors (BGR format)
        self.colors = {
            'champion': (0, 215, 255),    # Gold
            'primary': (255, 128, 0),     # Blue
            'success': (0, 255, 0),       # Green
            'highlight': (0, 255, 255),   # Yellow
            'background': (0, 0, 0),      # Black
            'text': (255, 255, 255),      # White
        }

    def create_title_slide(self, title, subtitle, duration_frames):
        """Create professional title slide"""
        frames = []
        for i in range(duration_frames):
            frame = np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)

            # Gradient background
            for y in range(self.frame_size[1]):
                intensity = int((y / self.frame_size[1]) * 50)
                frame[y, :] = [intensity, intensity, intensity]

            font = cv2.FONT_HERSHEY_SIMPLEX

            # Main title
            title_size = cv2.getTextSize(title, font, 2.5, 4)[0]
            title_x = (self.frame_size[0] - title_size[0]) // 2
            title_y = (self.frame_size[1] - title_size[1]) // 2 - 100

            cv2.putText(frame, title, (title_x, title_y), font, 2.5,
                       self.colors['champion'], 4, cv2.LINE_AA)

            # Subtitle
            if subtitle:
                sub_size = cv2.getTextSize(subtitle, font, 1.2, 2)[0]
                sub_x = (self.frame_size[0] - sub_size[0]) // 2
                sub_y = title_y + 120

                cv2.putText(frame, subtitle, (sub_x, sub_y), font, 1.2,
                           self.colors['text'], 2, cv2.LINE_AA)

            # Fade effects
            if i < 30:
                alpha = i / 30.0
                frame = (frame * alpha).astype(np.uint8)
            elif i > duration_frames - 30:
                alpha = (duration_frames - i) / 30.0
                frame = (frame * alpha).astype(np.uint8)

            frames.append(frame)

        return frames

    def create_professional_overlay(self, frame, velocity, distance, time_elapsed,
                                   progress, model_info):
        """Create professional overlay with metrics"""
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Semi-transparent HUD backgrounds
        cv2.rectangle(overlay, (0, 0), (w, 200), self.colors['background'], -1)
        cv2.rectangle(overlay, (0, h-140), (w, h), self.colors['background'], -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Title section
        cv2.putText(frame, "QUADRUPED LOCOMOTION - TRAINED POLICY",
                   (50, 50), font, 1.3, self.colors['champion'], 3, cv2.LINE_AA)

        cv2.putText(frame, model_info,
                   (50, 95), font, 0.8, self.colors['text'], 2, cv2.LINE_AA)

        cv2.putText(frame, "MuJoCo Physics Simulation",
                   (50, 135), font, 0.7, self.colors['primary'], 2, cv2.LINE_AA)

        # Performance metrics panel
        metrics_y = 175
        cv2.putText(frame, f"Velocity: {velocity:.3f} m/s",
                   (50, metrics_y), font, 0.9, self.colors['highlight'], 2, cv2.LINE_AA)

        cv2.putText(frame, f"Distance: {distance:.2f}m",
                   (400, metrics_y), font, 0.9, self.colors['highlight'], 2, cv2.LINE_AA)

        cv2.putText(frame, f"Time: {time_elapsed:.1f}s",
                   (700, metrics_y), font, 0.9, self.colors['highlight'], 2, cv2.LINE_AA)

        # Status indicator
        status_color = self.colors['success'] if velocity > 0.1 else self.colors['text']
        status_text = "WALKING" if velocity > 0.1 else "INITIALIZING"
        cv2.putText(frame, f"Status: {status_text}",
                   (1000, metrics_y), font, 0.9, status_color, 2, cv2.LINE_AA)

        # Progress bar
        bar_width = 800
        bar_height = 35
        bar_x = (w - bar_width) // 2
        bar_y = h - 85

        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (100, 100, 100), -1)

        # Progress fill
        progress_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height),
                     self.colors['primary'], -1)

        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     self.colors['champion'], 2)

        # Progress text
        cv2.putText(frame, f"Demonstration Progress: {progress*100:.0f}%",
                   (bar_x, bar_y - 15), font, 0.8, self.colors['text'], 2, cv2.LINE_AA)

        # Bottom info bar
        bottom_y = h - 40
        cv2.putText(frame, "PPO + Domain Randomization | RealANT Environment",
                   (50, bottom_y), font, 0.7, self.colors['text'], 2, cv2.LINE_AA)

        # Performance indicator
        if velocity > 0.2:
            cv2.putText(frame, "⭐ EXCELLENT PERFORMANCE",
                       (w - 450, bottom_y), font, 0.8, self.colors['champion'], 2, cv2.LINE_AA)
        elif velocity > 0.15:
            cv2.putText(frame, "✓ GOOD PERFORMANCE",
                       (w - 380, bottom_y), font, 0.8, self.colors['success'], 2, cv2.LINE_AA)

        return frame

    def record_demonstration(self, model_path, vec_path, output_path):
        """Record complete demonstration video"""
        print("\n" + "="*80)
        print("🏆 JUDGMENT DAY DEMONSTRATION VIDEO")
        print("="*80)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, self.frame_size)

        total_frames = 0

        # 1. INTRO
        print("\n1. Creating intro...")
        intro = self.create_title_slide(
            "QUADRUPED LOCOMOTION",
            "PPO Policy with Domain Randomization",
            self.intro_frames
        )
        for frame in intro:
            video_writer.write(frame)
            total_frames += 1

        # 2. MAIN DEMONSTRATION
        print("\n2. Recording simulation demonstration...")

        # Create environment
        def make_env():
            base_env = gym.make('RealAntMujoco-v0', render_mode='rgb_array',
                               disable_env_checker=True)
            while isinstance(base_env, TimeLimit):
                base_env = base_env.env
            env = TimeLimit(base_env, max_episode_steps=3000)
            env = SuccessRewardWrapper(env)
            return env

        env = DummyVecEnv([make_env])

        # Load model
        try:
            env = VecNormalize.load(vec_path, env)
            env.training = False
            env.norm_reward = False
            print("  ✓ VecNormalize loaded")
        except:
            print("  No VecNormalize")

        model = PPO.load(model_path)
        print("  ✓ Model loaded")

        # Model info for overlay
        model_info = f"Model: {os.path.basename(model_path).replace('.zip', '')}"

        # Record episode
        obs = env.reset()
        positions = []
        velocities = []
        frames_recorded = 0

        print(f"  Recording {self.main_demo_frames} frames...")

        for step in range(5000):  # Extra buffer
            if frames_recorded >= self.main_demo_frames:
                break

            # Get action
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            # Track position
            x_pos = env.envs[0].unwrapped.data.qpos[0]
            positions.append(x_pos)

            # Render
            frame = env.envs[0].render()
            if frame is not None:
                frame = cv2.resize(frame, self.frame_size)

                # Calculate metrics
                if len(positions) >= 2:
                    distance = positions[-1] - positions[0]
                    time_elapsed = len(positions) * 0.05
                    velocity = distance / time_elapsed if time_elapsed > 0 else 0
                    velocities.append(velocity)
                else:
                    distance = 0
                    velocity = 0
                    time_elapsed = 0

                progress = frames_recorded / self.main_demo_frames

                # Add professional overlay
                frame = self.create_professional_overlay(
                    frame, velocity, distance, time_elapsed, progress, model_info
                )

                video_writer.write(frame)
                frames_recorded += 1
                total_frames += 1

            if done[0]:
                # Reset but keep recording
                if len(positions) > 0:
                    positions.append(positions[-1])  # Continuous distance tracking
                obs = env.reset()

        env.close()

        # Calculate final metrics
        final_velocity = np.mean(velocities[-100:]) if len(velocities) > 100 else 0
        final_distance = positions[-1] - positions[0] if len(positions) > 1 else 0

        print(f"  ✓ Recorded {frames_recorded} frames")
        print(f"  Average velocity: {final_velocity:.3f} m/s")
        print(f"  Total distance: {final_distance:.2f}m")

        # 3. OUTRO
        print("\n3. Creating outro...")
        results_text = "DEMONSTRATION COMPLETE"
        results_subtitle = f"Avg Velocity: {final_velocity:.3f} m/s | Distance: {final_distance:.1f}m"
        outro = self.create_title_slide(results_text, results_subtitle, self.outro_frames)
        for frame in outro:
            video_writer.write(frame)
            total_frames += 1

        video_writer.release()

        # Save performance data
        performance_data = {
            'model': model_info,
            'duration_seconds': total_frames / self.fps,
            'average_velocity': float(final_velocity),
            'total_distance': float(final_distance),
            'frames': total_frames
        }

        with open(output_path.replace('.mp4', '_performance.json'), 'w') as f:
            json.dump(performance_data, f, indent=2)

        print("\n" + "="*80)
        print("✅ DEMONSTRATION VIDEO COMPLETE!")
        print("="*80)
        print(f"📁 Output: {output_path}")
        print(f"⏱️  Duration: {total_frames/self.fps:.1f} seconds")
        print(f"🎯 Performance: {final_velocity:.3f} m/s average")
        print(f"\n🏆 READY FOR JUDGMENT DAY!")

        return output_path, performance_data


def main():
    """Create judgment day demonstration video"""

    # Your best trained model
    model_name = "v7_7e_ultra_speed_jtfwl2qf"
    model_path = f"done/{model_name}/final_model.zip"
    vec_path = f"done/{model_name}/vec_normalize.pkl"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"videos/JUDGMENT_DAY_DEMO_{timestamp}.mp4"

    os.makedirs("videos", exist_ok=True)

    recorder = JudgmentDayRecorder()

    video_path, performance = recorder.record_demonstration(
        model_path, vec_path, output_path
    )

    print(f"\n🎬 VIDEO READY FOR PRESENTATION!")
    print(f"📹 Play this during your thesis defense")
    print(f"🤖 Physical robot will be demonstrated live alongside!")
    print(f"\nGOOD LUCK TOMORROW! 🏆🎓")


if __name__ == "__main__":
    main()
