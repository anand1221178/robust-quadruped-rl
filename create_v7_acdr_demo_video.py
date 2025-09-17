#!/usr/bin/env python3
"""
Create Demo Video for Fixed V7 ACDR Hard2Easy Model
Demonstrates single joint failure robustness with visual proof
"""

import os
import sys
import numpy as np
import gymnasium as gym
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from envs.success_reward_wrapper import SuccessRewardWrapper
import realant_sim


def create_acdr_demo_video():
    """Create demonstration video of Fixed V7 ACDR Hard2Easy model."""

    print("🎬 CREATING V7 ACDR FIXED HARD2EASY DEMO VIDEO")
    print("="*60)

    # Video setup
    video_filename = f"V7_ACDR_Fixed_Hard2Easy_Demo_{np.random.randint(1000, 9999)}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20
    video_writer = None

    # Model paths
    model_path = "experiments/v7_acdr_hard2easy_fixed_9wbi14fc/final_model.zip"
    vec_path = "experiments/v7_acdr_hard2easy_fixed_9wbi14fc/vec_normalize.pkl"

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    # Test scenarios: different single joint failures
    test_scenarios = [
        {"name": "Normal Operation", "failed_joints": [], "k": 1.0, "color": (0, 255, 0)},
        {"name": "Hip Joint 0 Failed", "failed_joints": [0], "k": 0.0, "color": (255, 100, 100)},
        {"name": "Ankle Joint 1 Failed", "failed_joints": [1], "k": 0.0, "color": (255, 150, 50)},
        {"name": "Hip Joint 2 Failed", "failed_joints": [2], "k": 0.0, "color": (255, 200, 100)},
        {"name": "Ankle Joint 3 Failed", "failed_joints": [3], "k": 0.0, "color": (200, 255, 100)},
        {"name": "Hip Joint 4 Failed", "failed_joints": [4], "k": 0.0, "color": (100, 255, 200)},
        {"name": "Ankle Joint 5 Failed", "failed_joints": [5], "k": 0.0, "color": (100, 200, 255)},
        {"name": "Hip Joint 6 Failed", "failed_joints": [6], "k": 0.0, "color": (150, 100, 255)},
        {"name": "Ankle Joint 7 Failed", "failed_joints": [7], "k": 0.0, "color": (255, 100, 200)},
    ]

    total_scenarios = len(test_scenarios)
    episode_length = 150  # Steps per scenario

    for scenario_idx, scenario in enumerate(test_scenarios):
        print(f"\n🎯 Scenario {scenario_idx + 1}/{total_scenarios}: {scenario['name']}")

        # Create environment
        def make_env():
            env = gym.make('RealAntMujoco-v0')
            env = SuccessRewardWrapper(env)
            env = Monitor(env)
            return env

        env = DummyVecEnv([make_env])

        # Load VecNormalize
        if os.path.exists(vec_path):
            env = VecNormalize.load(vec_path, env)
            env.training = False
            env.norm_reward = False

        # Load model
        model = PPO.load(model_path, env=env)

        # Reset environment
        obs = env.reset()

        # Track metrics
        positions = []
        step_count = 0
        total_reward = 0

        for step in range(episode_length):
            # Get action
            action, _ = model.predict(obs, deterministic=True)

            # Apply joint failure
            modified_action = action.copy()
            for joint_idx in scenario["failed_joints"]:
                if joint_idx < len(modified_action[0]):
                    modified_action[0][joint_idx] = scenario["k"] * action[0][joint_idx]

            # Step environment
            obs, reward, done, info = env.step(modified_action)
            total_reward += reward[0]

            # Get position
            try:
                x_pos = env.envs[0].unwrapped.data.qpos[0]
                positions.append(x_pos)
            except:
                positions.append(0.0)

            # Render frame
            frame = env.render(mode='rgb_array')
            if frame is not None and len(frame.shape) == 3:
                # Resize and convert to BGR for OpenCV
                frame = cv2.resize(frame, (1280, 720))
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Add overlay text
                scenario_color = scenario["color"]
                cv2.rectangle(frame_bgr, (10, 10), (600, 120), (0, 0, 0), -1)
                cv2.rectangle(frame_bgr, (10, 10), (600, 120), scenario_color, 3)

                # Main scenario text
                cv2.putText(frame_bgr, f"V7 ACDR Fixed Hard2Easy", (20, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame_bgr, f"Scenario: {scenario['name']}", (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, scenario_color, 2)

                # Performance metrics
                if len(positions) >= 2:
                    distance = positions[-1] - positions[0]
                    velocity = distance / (step * 0.05) if step > 0 else 0.0
                else:
                    distance = 0.0
                    velocity = 0.0

                cv2.putText(frame_bgr, f"Distance: {distance:.2f}m  Velocity: {velocity:.3f}m/s",
                           (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame_bgr, f"Step: {step}/{episode_length}  Reward: {total_reward:.0f}",
                           (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Failed joints indicator
                if scenario["failed_joints"]:
                    failed_text = f"FAILED JOINTS: {scenario['failed_joints']} (k={scenario['k']})"
                    cv2.putText(frame_bgr, failed_text, (620, 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Progress bar
                progress = (scenario_idx * episode_length + step) / (total_scenarios * episode_length)
                bar_width = 400
                bar_height = 10
                bar_x = 1280 - bar_width - 20
                bar_y = 700

                cv2.rectangle(frame_bgr, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
                cv2.rectangle(frame_bgr, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
                cv2.putText(frame_bgr, f"Overall Progress: {progress*100:.1f}%", (bar_x, bar_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Initialize video writer
                if video_writer is None:
                    height, width = frame_bgr.shape[:2]
                    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

                # Write frame
                video_writer.write(frame_bgr)

            step_count += 1
            if done[0]:
                break

        # Calculate final metrics for this scenario
        if len(positions) >= 2:
            final_distance = positions[-1] - positions[0]
            final_velocity = final_distance / (step_count * 0.05) if step_count > 0 else 0.0
        else:
            final_distance = 0.0
            final_velocity = 0.0

        print(f"   Final Distance: {final_distance:.2f}m")
        print(f"   Final Velocity: {final_velocity:.3f}m/s")
        print(f"   Total Reward: {total_reward:.0f}")
        print(f"   Success: {'✅' if final_velocity > 0.02 else '❌'}")

        env.close()

    # Close video writer
    if video_writer is not None:
        video_writer.release()
        print(f"\n🎬 VIDEO CREATED: {video_filename}")
        print(f"📊 Total Duration: ~{(total_scenarios * episode_length) / fps:.1f} seconds")
        print(f"🎯 Scenarios Tested: {total_scenarios}")
        print("\n✅ FIXED V7 ACDR DEMONSTRATION COMPLETE!")
    else:
        print("❌ Failed to create video")


if __name__ == "__main__":
    create_acdr_demo_video()