#!/usr/bin/env python3
"""
🏆 DR CHAMPIONSHIP EDITION - TOP-DOWN VIEW 🏆
Better camera angle for joint failure visualization
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
import json

class DRChampionTopDownRecorder:
    """DR Championship Video Creator with improved top-down camera view"""

    def __init__(self):
        self.frame_size = (1920, 1080)
        self.fps = 60  # High quality 60fps
        self.baseline_performance = None

        # Test duration per level (shorter for faster generation)
        self.frames_per_level = 600  # 10 seconds per level at 60fps

        # Joint failure test levels (FORCED failures!)
        self.failure_levels = [
            {'name': 'NO FAILURES', 'joints': [], 'prob': 0.0},
            {'name': 'HIP_1 LOCKED', 'joints': ['hip_1'], 'prob': 1.0},
            {'name': 'ANKLE_1 LOCKED', 'joints': ['ankle_1'], 'prob': 1.0},
            {'name': 'HIP_2 LOCKED', 'joints': ['hip_2'], 'prob': 1.0},
            {'name': 'ANKLE_2 LOCKED', 'joints': ['ankle_2'], 'prob': 1.0},
            {'name': 'HIP_3 LOCKED', 'joints': ['hip_3'], 'prob': 1.0},
            {'name': 'ANKLE_3 LOCKED', 'joints': ['ankle_3'], 'prob': 1.0},
            {'name': 'HIP_4 LOCKED', 'joints': ['hip_4'], 'prob': 1.0},
            {'name': 'ANKLE_4 LOCKED', 'joints': ['ankle_4'], 'prob': 1.0},
        ]

        # Colors (BGR format for OpenCV)
        self.colors = {
            'champion': (0, 215, 255),    # Gold
            'excellent': (0, 255, 0),     # Green
            'good': (0, 255, 255),        # Yellow
            'challenge': (0, 165, 255),   # Orange
            'extreme': (0, 0, 255),       # Red
            'background': (0, 0, 0),      # Black
            'text': (255, 255, 255),      # White
            'failure': (255, 0, 255)      # Magenta for failed joints
        }

        # Joint name mapping
        self.joint_display_names = {
            'hip_1': 'Front Left Hip',
            'ankle_1': 'Front Left Ankle',
            'hip_2': 'Front Right Hip',
            'ankle_2': 'Front Right Ankle',
            'hip_3': 'Rear Left Hip',
            'ankle_3': 'Rear Left Ankle',
            'hip_4': 'Rear Right Hip',
            'ankle_4': 'Rear Right Ankle'
        }

    def setup_topdown_camera(self, env):
        """Setup better camera angle for top-down view"""
        try:
            # Access the MuJoCo model and data
            model = env.envs[0].unwrapped.model
            data = env.envs[0].unwrapped.data

            # Get the camera ID (usually 0 for default camera)
            cam_id = 0

            # Set top-down camera position and orientation
            # Position camera above and slightly behind the robot
            model.cam_pos[cam_id] = [0, -2, 3]  # x, y, z position (higher z for top-down)
            model.cam_quat[cam_id] = [0.7071, 0.7071, 0, 0]  # Quaternion for looking down

            print("  ✅ Top-down camera configured")
            return True
        except Exception as e:
            print(f"  ⚠️ Camera setup failed: {e}")
            return False

    def force_joint_failure(self, env, joint_names):
        """FORCE specific joints to fail by locking their actions"""
        if not joint_names:
            return None

        # Direct action index mapping for RealAnt
        joint_to_action_index = {
            'hip_1': 0,    # Front left hip
            'ankle_1': 1,  # Front left ankle
            'hip_2': 2,    # Front right hip
            'ankle_2': 3,  # Front right ankle
            'hip_3': 4,    # Rear left hip
            'ankle_3': 5,  # Rear left ankle
            'hip_4': 6,    # Rear right hip
            'ankle_4': 7,  # Rear right ankle
        }

        joint_indices = []
        for joint_name in joint_names:
            if joint_name in joint_to_action_index:
                joint_indices.append(joint_to_action_index[joint_name])

        return joint_indices

    def apply_joint_failure(self, action, failed_joint_indices):
        """Lock specific joint actions to 0 (failure)"""
        if not failed_joint_indices:
            return action

        action_copy = action.copy()
        for joint_idx in failed_joint_indices:
            if joint_idx < len(action_copy[0]):
                action_copy[0][joint_idx] = 0.0  # Lock joint

        return action_copy

    def get_performance_color(self, retention_pct):
        """Get color based on retention percentage"""
        if retention_pct >= 95:
            return self.colors['champion']  # Gold for excellent
        elif retention_pct >= 80:
            return self.colors['excellent']  # Green for good
        elif retention_pct >= 60:
            return self.colors['good']       # Yellow for ok
        elif retention_pct >= 40:
            return self.colors['challenge']  # Orange for challenging
        else:
            return self.colors['extreme']    # Red for extreme

    def create_topdown_overlay(self, frame, failure_level, current_velocity, episode_progress,
                              current_distance, retention_pct, failed_joints):
        """Create enhanced overlay with better visibility for top-down view"""
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Larger, more prominent HUD for better visibility
        cv2.rectangle(overlay, (0, 0), (w, 180), self.colors['background'], -1)
        cv2.rectangle(overlay, (0, h-120), (w, h), self.colors['background'], -1)
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Larger title for better visibility
        title = "DR: TOP-DOWN JOINT FAILURE ROBUSTNESS"
        cv2.putText(frame, title, (50, 50), font, 1.3, self.colors['champion'], 4)

        # Current challenge level (larger text)
        challenge_text = f"SCENARIO: {failure_level['name']}"
        cv2.putText(frame, challenge_text, (50, 90), font, 1.0, self.colors['text'], 3)

        # Failed joints display (more prominent)
        if failed_joints:
            joints_text = "FAILED: " + ", ".join([self.joint_display_names.get(j, j) for j in failed_joints])
            cv2.putText(frame, joints_text, (50, 130), font, 0.8, self.colors['failure'], 3)
        else:
            cv2.putText(frame, "ALL JOINTS OPERATIONAL", (50, 130), font, 0.8, self.colors['excellent'], 3)

        # Performance metrics (larger and more visible)
        metrics_y = 160
        cv2.putText(frame, f"VELOCITY: {current_velocity:.3f} m/s",
                   (50, metrics_y), font, 0.8, self.colors['text'], 2)

        cv2.putText(frame, f"DISTANCE: {current_distance:.1f}m",
                   (400, metrics_y), font, 0.8, self.colors['text'], 2)

        # Large retention percentage
        retention_color = self.get_performance_color(retention_pct)
        cv2.putText(frame, f"RETENTION: {retention_pct:.1f}%",
                   (750, metrics_y), font, 1.0, retention_color, 3)

        # Robot indicator in center of screen
        robot_x = w // 2
        robot_y = h // 2

        # Draw robot position indicator
        if failed_joints:
            # Pulsing red circle for failed state
            pulse = int(100 + 50 * np.sin(episode_progress * 10))
            cv2.circle(frame, (robot_x, robot_y), 30, (0, 0, pulse), 3)
            cv2.putText(frame, "ROBOT", (robot_x - 30, robot_y + 50), font, 0.6, self.colors['failure'], 2)
        else:
            # Green circle for normal operation
            cv2.circle(frame, (robot_x, robot_y), 25, self.colors['excellent'], 3)
            cv2.putText(frame, "ROBOT", (robot_x - 30, robot_y + 50), font, 0.6, self.colors['excellent'], 2)

        # Progress bar (larger)
        bar_width = 600
        bar_height = 25
        bar_x = w - bar_width - 50
        bar_y = h - 80

        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (100, 100, 100), -1)

        # Progress fill
        progress_width = int(bar_width * episode_progress)
        progress_color = self.get_performance_color(retention_pct)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height),
                     progress_color, -1)

        # Progress text (larger)
        cv2.putText(frame, f"SCENARIO PROGRESS: {episode_progress*100:.0f}%",
                   (bar_x, bar_y - 15), font, 0.7, self.colors['text'], 2)

        return frame

    def record_topdown_championship(self, model_path, vec_path, output_path):
        """Record the complete DR championship with top-down view"""
        print("=" * 80)
        print("DR CHAMPIONSHIP EDITION - TOP-DOWN VIEW")
        print("Better camera angle for joint failure visualization")
        print("=" * 80)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, self.frame_size)

        total_frames = 0
        performance_data = []

        for level_idx, failure_level in enumerate(self.failure_levels):
            print(f"\nTesting failure level {level_idx+1}/{len(self.failure_levels)}: {failure_level['name']}")
            print(f"Target frames for this level: {self.frames_per_level}")

            # Create environment with improved camera
            def make_env():
                env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')
                env = SuccessRewardWrapper(env)
                return env

            env = DummyVecEnv([make_env])

            # Setup top-down camera
            self.setup_topdown_camera(env)

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

            # Get joint indices for failure injection
            failed_joint_indices = self.force_joint_failure(env, failure_level['joints'])

            # Record episode
            obs = env.reset()
            positions = []
            frames_this_level = 0

            print(f"  Recording with top-down view...")

            for step in range(1000):  # Max steps per failure level
                if frames_this_level >= self.frames_per_level:
                    break

                # Get action from model
                action, _ = model.predict(obs, deterministic=True)

                # Apply joint failure
                if failed_joint_indices:
                    action = self.apply_joint_failure(action, failed_joint_indices)

                obs, reward, done, info = env.step(action)

                # Track position
                x_pos = env.envs[0].unwrapped.data.qpos[0]
                positions.append(x_pos)

                # Render with top-down view
                frame = env.envs[0].render()
                if frame is not None:
                    frame = cv2.resize(frame, self.frame_size)

                    # Calculate metrics
                    if len(positions) >= 2:
                        current_distance = positions[-1] - positions[0]
                        time_elapsed = len(positions) * 0.05
                        current_velocity = current_distance / time_elapsed
                    else:
                        current_distance = 0
                        current_velocity = 0

                    retention_pct = (current_velocity / self.baseline_performance) * 100 if self.baseline_performance and self.baseline_performance > 0 else 0
                    episode_progress = frames_this_level / self.frames_per_level

                    # Apply enhanced top-down overlay
                    frame = self.create_topdown_overlay(
                        frame, failure_level, current_velocity, episode_progress,
                        current_distance, retention_pct, failure_level['joints']
                    )

                    video_writer.write(frame)
                    frames_this_level += 1
                    total_frames += 1

                if done[0]:
                    break

            # Store performance data
            if len(positions) >= 2:
                final_distance = positions[-1] - positions[0]
                time_taken = len(positions) * 0.05
                final_velocity = final_distance / time_taken

                # Set baseline
                if self.baseline_performance is None and failure_level['name'] == 'NO FAILURES':
                    self.baseline_performance = final_velocity
                    print(f"  ✅ Baseline performance set: {self.baseline_performance:.3f} m/s")

                retention = (final_velocity / self.baseline_performance) * 100 if self.baseline_performance > 0 else 0

                performance_data.append({
                    'failure_mode': failure_level['name'],
                    'failed_joints': failure_level['joints'],
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

        print(f"\nDR CHAMPIONSHIP TOP-DOWN VIDEO COMPLETE!")
        print(f"Video: {output_path}")
        print(f"Duration: {total_frames/self.fps:.1f} seconds")

        return output_path, performance_data

def main():
    """Create the DR Championship Edition video with top-down view"""

    # Use V7.3 Multi-Objective model (best balanced performance)
    model_path = "experiments/v7_3_acdr_multi_objective_jui50qpd/final_model.zip"
    vec_path = "experiments/v7_3_acdr_multi_objective_jui50qpd/vec_normalize.pkl"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"videos/DR_CHAMPION_TOPDOWN_{timestamp}.mp4"

    os.makedirs("videos", exist_ok=True)

    recorder = DRChampionTopDownRecorder()

    video_path, performance_data = recorder.record_topdown_championship(
        model_path, vec_path, output_path
    )

    print(f"\nDR CHAMPIONSHIP TOP-DOWN EDITION COMPLETE!")
    print(f"Video: {video_path}")

    print(f"\nCHAMPIONSHIP HIGHLIGHTS:")
    for data in performance_data:
        failure_mode = data['failure_mode']
        retention = data['retention_pct']
        frames = data.get('frames_recorded', 0)
        joints = ', '.join(data['failed_joints']) if data['failed_joints'] else 'None'
        print(f"  {failure_mode}: {retention:.1f}% retention, Failed: {joints} ({frames} frames)")

if __name__ == "__main__":
    main()