#!/usr/bin/env python3
"""
🏆 V7 ACDR CHAMPIONSHIP EDITION 🏆
Professional demonstration of Fixed Hard2Easy curriculum-based domain randomization
Based on SR2L champion video methodology
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

class V7ACDRChampionRecorder:
    """V7 ACDR Championship Video Creator with joint failure demonstrations"""

    def __init__(self):
        self.frame_size = (1920, 1080)
        self.fps = 60  # High quality 60fps
        self.baseline_performance = None  # Will be calculated from no-failure performance

        # Duration per test scenario
        self.frames_per_scenario = 600  # 10 seconds per scenario at 60fps

        # Championship test scenarios: different joint failure patterns
        self.test_scenarios = [
            {"name": "PERFECT OPERATION", "failed_joints": [], "k": 1.0, "description": "BASELINE PERFORMANCE"},
            {"name": "FRONT-LEFT HIP DEAD", "failed_joints": [0], "k": 0.0, "description": "SINGLE HIP FAILURE"},
            {"name": "FRONT-LEFT ANKLE DEAD", "failed_joints": [1], "k": 0.0, "description": "SINGLE ANKLE FAILURE"},
            {"name": "FRONT-RIGHT HIP DEAD", "failed_joints": [2], "k": 0.0, "description": "CROSS-BODY ADAPTATION"},
            {"name": "FRONT-RIGHT ANKLE DEAD", "failed_joints": [3], "k": 0.0, "description": "ASYMMETRIC GAIT"},
            {"name": "REAR-LEFT HIP DEAD", "failed_joints": [4], "k": 0.0, "description": "REAR PROPULSION LOSS"},
            {"name": "REAR-LEFT ANKLE DEAD", "failed_joints": [5], "k": 0.0, "description": "CRITICAL STABILITY TEST"},
            {"name": "REAR-RIGHT HIP DEAD", "failed_joints": [6], "k": 0.0, "description": "POWER LIMB FAILURE"},
            {"name": "REAR-RIGHT ANKLE DEAD", "failed_joints": [7], "k": 0.0, "description": "EXTREME ROBUSTNESS"},
        ]

        # Championship colors (BGR format for OpenCV)
        self.colors = {
            'champion': (0, 215, 255),    # Gold
            'excellent': (0, 255, 0),     # Green
            'good': (0, 255, 255),        # Yellow
            'challenge': (0, 165, 255),   # Orange
            'extreme': (0, 0, 255),       # Red
            'background': (0, 0, 0),      # Black
            'text': (255, 255, 255),      # White
            'curriculum': (255, 0, 255),  # Magenta for curriculum highlight
            'failed_pulse': (0, 0, 255),  # Pulsing red for failed joints
            'robot_body': (150, 150, 150), # Gray for robot body
        }

        # Joint mapping for visualization
        self.joint_names = {
            0: "FRONT-LEFT HIP",
            1: "FRONT-LEFT ANKLE",
            2: "FRONT-RIGHT HIP",
            3: "FRONT-RIGHT ANKLE",
            4: "REAR-LEFT HIP",
            5: "REAR-LEFT ANKLE",
            6: "REAR-RIGHT HIP",
            7: "REAR-RIGHT ANKLE"
        }

    def get_performance_color(self, retention_pct):
        """Get color based on retention percentage"""
        if retention_pct >= 100:
            return self.colors['champion']
        elif retention_pct >= 80:
            return self.colors['excellent']
        elif retention_pct >= 60:
            return self.colors['good']
        elif retention_pct >= 40:
            return self.colors['challenge']
        else:
            return self.colors['extreme']

    def create_epic_overlay(self, frame, scenario, current_velocity,
                           episode_progress, current_distance, retention_pct, frames_this_scenario=0):
        """Create championship overlay with scenario information"""
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Semi-transparent background for HUD (adjusted to not overlap robot diagram)
        cv2.rectangle(overlay, (0, 0), (w-460, 220), self.colors['background'], -1)  # Leave space for robot diagram
        cv2.rectangle(overlay, (0, h-150), (w, h), self.colors['background'], -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Championship title
        title = "V7 ACDR: CURRICULUM-BASED DOMAIN RANDOMIZATION CHAMPION"
        cv2.putText(frame, title, (50, 50), font, 1.0, self.colors['curriculum'], 3)

        # Current scenario
        scenario_text = f"SCENARIO: {scenario['name']}"
        cv2.putText(frame, scenario_text, (50, 90), font, 0.8, self.colors['text'], 2)

        # Scenario description
        desc_text = scenario['description']
        cv2.putText(frame, desc_text, (50, 120), font, 0.7, self.colors['champion'], 2)

        # Failed joints indicator
        if scenario['failed_joints']:
            joint_text = f"FAILED JOINTS: {scenario['failed_joints']} (k={scenario['k']})"
            cv2.putText(frame, joint_text, (50, 150), font, 0.6, self.colors['extreme'], 2)
            cv2.putText(frame, "COMPLETE ACTUATOR FAILURE", (50, 175), font, 0.6, self.colors['extreme'], 2)
        else:
            cv2.putText(frame, "ALL ACTUATORS OPERATIONAL", (50, 150), font, 0.6, self.colors['excellent'], 2)

        # Performance metrics dashboard
        metrics_y = 200
        cv2.putText(frame, f"VELOCITY: {current_velocity:.3f} m/s",
                   (50, metrics_y), font, 0.7, self.colors['text'], 2)

        cv2.putText(frame, f"DISTANCE: {current_distance:.1f}m",
                   (300, metrics_y), font, 0.7, self.colors['text'], 2)

        # Retention percentage
        retention_color = self.get_performance_color(retention_pct)
        cv2.putText(frame, f"RETENTION: {retention_pct:.1f}%",
                   (550, metrics_y), font, 0.8, retention_color, 2)

        # Curriculum achievement indicator
        if retention_pct > 80:
            curriculum_text = "CURRICULUM SUCCESS!"
            curriculum_color = self.colors['champion']
        elif retention_pct > 50:
            curriculum_text = "ROBUST ADAPTATION"
            curriculum_color = self.colors['good']
        elif retention_pct > 20:
            curriculum_text = "PARTIAL RECOVERY"
            curriculum_color = self.colors['challenge']
        else:
            curriculum_text = "CRITICAL FAILURE"
            curriculum_color = self.colors['extreme']

        cv2.putText(frame, curriculum_text, (800, metrics_y), font, 0.7, curriculum_color, 2)

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
        cv2.putText(frame, f"SCENARIO PROGRESS: {episode_progress*100:.0f}%",
                   (bar_x, bar_y - 10), font, 0.6, self.colors['text'], 2)

        # Special annotations for extreme scenarios
        if len(scenario['failed_joints']) > 0:
            cv2.putText(frame, "FIXED CURRICULUM LEARNING IN ACTION!",
                       (50, h - 50), font, 1.0, self.colors['curriculum'], 3)
            cv2.putText(frame, "HARD2EASY APPROACH: DEAD JOINTS -> ROBUST LOCOMOTION",
                       (50, h - 20), font, 0.7, self.colors['curriculum'], 2)

        # Add robot highlighting to simulation
        frame = self.highlight_robot_in_simulation(frame, scenario)

        # Joint failure visualizer (with frame count for pulsing effect)
        self.draw_joint_failure_visualizer(frame, scenario, frames_this_scenario)

        return frame

    def draw_joint_failure_visualizer(self, frame, scenario, frame_count=0):
        """Draw enhanced robot diagram with joint failure visualization"""
        h, w = frame.shape[:2]

        # Main robot diagram position (top-right corner, larger and more prominent)
        diagram_x = w - 440
        diagram_y = 20
        diagram_w = 420
        diagram_h = 320

        # Prominent background panel with thicker border
        cv2.rectangle(frame, (diagram_x - 15, diagram_y - 15),
                     (diagram_x + diagram_w + 15, diagram_y + diagram_h + 15),
                     (20, 20, 20), -1)  # Darker background
        cv2.rectangle(frame, (diagram_x - 15, diagram_y - 15),
                     (diagram_x + diagram_w + 15, diagram_y + diagram_h + 15),
                     self.colors['champion'], 4)  # Gold border for prominence

        # Title
        cv2.putText(frame, "QUADRUPED JOINT STATUS",
                   (diagram_x, diagram_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   self.colors['text'], 2)

        # Robot body center
        body_center_x = diagram_x + diagram_w // 2
        body_center_y = diagram_y + diagram_h // 2
        body_width = 120
        body_height = 60

        # Draw robot body (rectangular)
        cv2.rectangle(frame,
                     (body_center_x - body_width//2, body_center_y - body_height//2),
                     (body_center_x + body_width//2, body_center_y + body_height//2),
                     self.colors['robot_body'], -1)
        cv2.rectangle(frame,
                     (body_center_x - body_width//2, body_center_y - body_height//2),
                     (body_center_x + body_width//2, body_center_y + body_height//2),
                     self.colors['text'], 2)

        # Robot direction indicator
        cv2.putText(frame, "FRONT", (body_center_x - 25, body_center_y - body_height//2 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        cv2.arrowedLine(frame, (body_center_x, body_center_y - body_height//2 - 5),
                       (body_center_x, body_center_y - body_height//2 - 20),
                       self.colors['text'], 2)

        # Leg positions (4 legs from body center)
        leg_length = 80
        joint_radius = 12

        # Leg positions: [front-left, front-right, rear-left, rear-right]
        leg_positions = [
            # Front-left leg
            (body_center_x - body_width//2, body_center_y - body_height//2),
            # Front-right leg
            (body_center_x + body_width//2, body_center_y - body_height//2),
            # Rear-left leg
            (body_center_x - body_width//2, body_center_y + body_height//2),
            # Rear-right leg
            (body_center_x + body_width//2, body_center_y + body_height//2)
        ]

        # Joint pairs for each leg: [(hip_joint_id, ankle_joint_id), ...]
        joint_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
        leg_names = ["FRONT-LEFT", "FRONT-RIGHT", "REAR-LEFT", "REAR-RIGHT"]

        # Draw each leg with joints
        for leg_idx, ((hip_joint_id, ankle_joint_id), (leg_x, leg_y)) in enumerate(zip(joint_pairs, leg_positions)):
            # Calculate joint positions
            hip_x, hip_y = leg_x, leg_y + 25  # Hip joint offset from body
            ankle_x, ankle_y = leg_x, leg_y + leg_length  # Ankle at end of leg

            # Draw leg line
            cv2.line(frame, (hip_x, hip_y), (ankle_x, ankle_y), self.colors['robot_body'], 4)

            # Hip joint
            hip_failed = hip_joint_id in scenario['failed_joints']
            if hip_failed:
                # Pulsing effect for failed joints
                pulse = int(127 + 128 * np.sin(frame_count * 0.3))
                hip_color = (0, 0, pulse)  # Pulsing red
                hip_thickness = -1  # Filled
            else:
                hip_color = self.colors['excellent']
                hip_thickness = -1

            cv2.circle(frame, (hip_x, hip_y), joint_radius, hip_color, hip_thickness)
            cv2.circle(frame, (hip_x, hip_y), joint_radius, self.colors['text'], 2)

            # Ankle joint
            ankle_failed = ankle_joint_id in scenario['failed_joints']
            if ankle_failed:
                # Pulsing effect for failed joints
                pulse = int(127 + 128 * np.sin(frame_count * 0.3))
                ankle_color = (0, 0, pulse)  # Pulsing red
                ankle_thickness = -1  # Filled
            else:
                ankle_color = self.colors['excellent']
                ankle_thickness = -1

            cv2.circle(frame, (ankle_x, ankle_y), joint_radius, ankle_color, ankle_thickness)
            cv2.circle(frame, (ankle_x, ankle_y), joint_radius, self.colors['text'], 2)

            # Joint labels
            cv2.putText(frame, f"H{hip_joint_id}", (hip_x - 8, hip_y + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.colors['text'], 1)
            cv2.putText(frame, f"A{ankle_joint_id}", (ankle_x - 8, ankle_y + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.colors['text'], 1)

            # Leg name
            cv2.putText(frame, leg_names[leg_idx], (leg_x - 35, leg_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)

            # DEAD indicators for failed joints
            if hip_failed:
                cv2.putText(frame, "DEAD", (hip_x + 15, hip_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['extreme'], 2)
            if ankle_failed:
                cv2.putText(frame, "DEAD", (ankle_x + 15, ankle_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['extreme'], 2)

        # Legend at bottom of diagram
        legend_y = diagram_y + diagram_h - 40
        cv2.putText(frame, "LEGEND:", (diagram_x, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)

        # Working joint indicator
        cv2.circle(frame, (diagram_x + 70, legend_y - 8), 8, self.colors['excellent'], -1)
        cv2.putText(frame, "WORKING", (diagram_x + 85, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)

        # Failed joint indicator (pulsing)
        pulse = int(127 + 128 * np.sin(frame_count * 0.3))
        cv2.circle(frame, (diagram_x + 180, legend_y - 8), 8, (0, 0, pulse), -1)
        cv2.putText(frame, "FAILED (k=0.0)", (diagram_x + 195, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['extreme'], 1)

        # Current failure status
        if scenario['failed_joints']:
            for joint_id in scenario['failed_joints']:
                joint_name = self.joint_names[joint_id]
                status_text = f">>> {joint_name} COMPLETELY DISABLED <<<"
                cv2.putText(frame, status_text, (diagram_x, legend_y + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['extreme'], 2)
        else:
            cv2.putText(frame, ">>> ALL JOINTS OPERATIONAL <<<", (diagram_x, legend_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['excellent'], 2)

    def highlight_robot_in_simulation(self, frame, scenario):
        """Add highlighting to the actual robot in the simulation"""
        h, w = frame.shape[:2]

        # Create overlay for robot highlighting
        highlight_overlay = frame.copy()

        # Find robot center approximately (MuJoCo typically centers the robot)
        robot_center_x = w // 2
        robot_center_y = int(h * 0.7)  # Robot usually in lower 2/3 of frame

        # Draw highlighting circle around robot area
        if scenario['failed_joints']:
            # Red pulsing circle for failed joints
            pulse = int(100 + 50 * np.sin(len(scenario['failed_joints']) * 0.5))
            highlight_color = (0, 0, pulse)
            circle_radius = 150
        else:
            # Green steady circle for normal operation
            highlight_color = self.colors['excellent']
            circle_radius = 120

        # Draw highlighting circle
        cv2.circle(highlight_overlay, (robot_center_x, robot_center_y),
                  circle_radius, highlight_color, 6)

        # Add text overlay directly on simulation
        if scenario['failed_joints']:
            for joint_id in scenario['failed_joints']:
                joint_name = self.joint_names[joint_id]
                # Position text near robot
                text_x = robot_center_x - 150
                text_y = robot_center_y + circle_radius + 30

                cv2.putText(highlight_overlay, f"DISABLED: {joint_name}",
                           (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                           self.colors['extreme'], 3)

                # Add arrow pointing to robot
                cv2.arrowedLine(highlight_overlay,
                               (text_x + 100, text_y - 15),
                               (robot_center_x, robot_center_y + circle_radius - 20),
                               self.colors['extreme'], 4)

        # Blend with original frame
        return cv2.addWeighted(frame, 0.8, highlight_overlay, 0.2, 0)

    def record_championship_sequence(self, model_path, vec_path, output_path):
        """Record championship sequence with joint failure scenarios"""
        print("=" * 80)
        print("V7 ACDR CHAMPIONSHIP EDITION")
        print("Fixed Hard2Easy Curriculum-Based Domain Randomization")
        print("=" * 80)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, self.frame_size)

        total_frames = 0
        performance_data = []

        for scenario_idx, scenario in enumerate(self.test_scenarios):
            print(f"\nRecording scenario {scenario_idx+1}/{len(self.test_scenarios)}: {scenario['name']}")
            print(f"Target frames for this scenario: {self.frames_per_scenario}")

            # Create environment with rendering
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

            # Record episode with current scenario
            obs = env.reset()
            positions = []
            rewards = []
            frames_this_scenario = 0

            print(f"  Starting scenario with joints {scenario['failed_joints']} at k={scenario['k']}")

            for step in range(1000):  # Max steps per scenario
                if frames_this_scenario >= self.frames_per_scenario:
                    break

                # Get action from model
                action, _ = model.predict(obs, deterministic=True)

                # Apply joint failure
                modified_action = action.copy()
                for joint_idx in scenario['failed_joints']:
                    if joint_idx < len(modified_action[0]):
                        modified_action[0][joint_idx] = scenario['k'] * action[0][joint_idx]

                obs, reward, done, info = env.step(modified_action)

                # Track metrics
                x_pos = env.envs[0].unwrapped.data.qpos[0]
                positions.append(x_pos)
                rewards.append(reward[0])

                # Render and create overlay
                frame = env.envs[0].render()
                if frame is not None:
                    frame = cv2.resize(frame, self.frame_size)

                    # Calculate current metrics (net forward displacement)
                    if len(positions) >= 2:
                        current_distance = positions[-1] - positions[0]
                        time_elapsed = len(positions) * 0.05  # 20Hz timestep
                        current_velocity = current_distance / time_elapsed
                    else:
                        current_distance = 0
                        current_velocity = 0

                    retention_pct = (current_velocity / self.baseline_performance) * 100 if self.baseline_performance and self.baseline_performance > 0 else 0
                    episode_progress = frames_this_scenario / self.frames_per_scenario

                    # Apply championship overlay
                    frame = self.create_epic_overlay(
                        frame, scenario, current_velocity, episode_progress,
                        current_distance, retention_pct, frames_this_scenario
                    )

                    video_writer.write(frame)
                    frames_this_scenario += 1
                    total_frames += 1

                if done[0]:
                    print(f"    Episode ended early at step {step}")
                    break

            # Store performance data
            if len(positions) >= 2:
                initial_x = positions[0]
                final_x = positions[-1]
                final_distance = final_x - initial_x
                time_taken = len(positions) * 0.05
                final_velocity = final_distance / time_taken

                # Set baseline from first (no-failure) run
                if self.baseline_performance is None:
                    self.baseline_performance = final_velocity
                    print(f"  ✅ Baseline performance set: {self.baseline_performance:.3f} m/s")

                retention = (final_velocity / self.baseline_performance) * 100 if self.baseline_performance > 0 else 0

                performance_data.append({
                    'scenario': scenario['name'],
                    'failed_joints': scenario['failed_joints'],
                    'k_value': scenario['k'],
                    'velocity': final_velocity,
                    'retention_pct': retention,
                    'distance': final_distance,
                    'frames_recorded': frames_this_scenario
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
    """Create the V7 ACDR Championship Edition video"""

    model_path = "experiments/v7_acdr_hard2easy_fixed_9wbi14fc/final_model.zip"
    vec_path = "experiments/v7_acdr_hard2easy_fixed_9wbi14fc/vec_normalize.pkl"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"V7_ACDR_CHAMPION_{timestamp}.mp4"

    recorder = V7ACDRChampionRecorder()
    video_path, performance_data = recorder.record_championship_sequence(
        model_path, vec_path, output_path
    )

    print(f"\n🏆 V7 ACDR CHAMPIONSHIP COMPLETE!")
    print(f"📹 Video: {video_path}")
    print(f"📊 Performance data: {output_path.replace('.mp4', '_performance.json')}")

    # Print summary
    print(f"\n📈 PERFORMANCE SUMMARY:")
    for data in performance_data:
        print(f"  {data['scenario']}: {data['velocity']:.3f} m/s ({data['retention_pct']:.1f}%)")

if __name__ == "__main__":
    main()