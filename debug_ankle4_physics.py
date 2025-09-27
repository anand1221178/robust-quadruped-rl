#!/usr/bin/env python3
"""
Debug script to identify the exact cause of ankle_4 physics glitch
Logs detailed physics state before, during, and after ankle_4 failure
"""
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
from gymnasium.wrappers import TimeLimit
import realant_sim
import time

def debug_ankle4_physics():
    """Debug ankle_4 physics glitch with detailed logging"""

    print("=" * 80)
    print("ANKLE_4 PHYSICS DEBUG - DETAILED STATE LOGGING")
    print("=" * 80)

    # Model to test (V7.10C where bug occurs)
    model_name = "v7_10c_symmetric_training_050b8r4m"
    model_path = f"experiments/{model_name}/final_model.zip"
    vec_path = f"experiments/{model_name}/vec_normalize.pkl"

    # Create environment with extended episodes
    def make_env():
        base_env = gym.make('RealAntMujoco-v0', render_mode='rgb_array', disable_env_checker=True)
        while isinstance(base_env, TimeLimit):
            base_env = base_env.env
        env = TimeLimit(base_env, max_episode_steps=2500)
        env = SuccessRewardWrapper(env)
        return env

    env = DummyVecEnv([make_env])

    # Load normalization and model
    try:
        env = VecNormalize.load(vec_path, env)
        env.training = False
        env.norm_reward = False
        print("✅ VecNormalize loaded")
    except:
        print("❌ No VecNormalize")

    model = PPO.load(model_path)
    print("✅ Model loaded")
    print()

    # Test different ankle_4 locking values
    lock_values_to_test = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, -0.3]

    for lock_value in lock_values_to_test:
        print(f"\n{'='*60}")
        print(f"Testing ankle_4 lock value: {lock_value} (angle ~{lock_value*90:.1f}°)")
        print(f"{'='*60}")

        obs = env.reset()

        # Let robot walk normally for 2 seconds
        print("Phase 1: Normal walking for 2 seconds...")
        normal_positions = []
        for step in range(120):  # 2 seconds at 60Hz
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            # Get robot state
            x_pos = env.envs[0].unwrapped.data.qpos[0]
            normal_positions.append(x_pos)

            if step % 30 == 0:
                print(f"  Step {step}: Position={x_pos:.3f}m")

        normal_velocity = (normal_positions[-1] - normal_positions[0]) / 2.0
        print(f"  Normal velocity: {normal_velocity:.3f} m/s")

        # Now lock ankle_4
        print(f"\nPhase 2: Locking ankle_4 at value={lock_value}")
        print("Monitoring for physics glitches...")

        glitch_detected = False
        prev_pos = normal_positions[-1]
        prev_velocities = []

        for step in range(180):  # 3 seconds with locked ankle
            action, _ = model.predict(obs, deterministic=True)

            # LOCK ANKLE_4 (joint index 7)
            action[0][7] = lock_value

            # Get detailed state BEFORE step
            unwrapped_env = env.envs[0].unwrapped

            # Joint positions and velocities
            joint_pos = unwrapped_env.data.qpos.copy()
            joint_vel = unwrapped_env.data.qvel.copy()

            # Ankle_4 specific state (joint index 7 in qpos)
            ankle4_angle = joint_pos[7] if len(joint_pos) > 7 else 0
            ankle4_velocity = joint_vel[7] if len(joint_vel) > 7 else 0

            # Robot height (Z position)
            robot_height = joint_pos[2] if len(joint_pos) > 2 else 0

            # Step environment
            try:
                obs, reward, done, info = env.step(action)
            except Exception as e:
                print(f"  ❌ EXCEPTION at step {step}: {e}")
                glitch_detected = True
                break

            # Get position after step
            x_pos = unwrapped_env.data.qpos[0]
            z_pos = unwrapped_env.data.qpos[2] if len(unwrapped_env.data.qpos) > 2 else 0

            # Detect physics glitches
            position_jump = abs(x_pos - prev_pos)

            # Log every 10 steps or if glitch detected
            if step % 10 == 0 or position_jump > 1.0 or z_pos > 0.5:
                print(f"  Step {step}:")
                print(f"    Position: X={x_pos:.3f}, Z={z_pos:.3f} (height)")
                print(f"    Ankle_4: angle={ankle4_angle:.3f}, vel={ankle4_velocity:.3f}")
                print(f"    Position jump: {position_jump:.3f}m")

                if position_jump > 1.0:
                    print(f"    ⚠️ LARGE POSITION JUMP DETECTED!")
                    glitch_detected = True

                if z_pos > 0.5:
                    print(f"    ⚠️ ROBOT FLOATING! Height={z_pos:.3f}")
                    glitch_detected = True

                if abs(ankle4_velocity) > 10:
                    print(f"    ⚠️ EXTREME ANKLE VELOCITY!")
                    glitch_detected = True

            prev_pos = x_pos

            # Check if simulation is stuck
            prev_velocities.append(x_pos)
            if len(prev_velocities) > 30:  # Check last 0.5 seconds
                prev_velocities.pop(0)
                if max(prev_velocities) - min(prev_velocities) < 0.01:
                    print(f"    ⚠️ ROBOT STUCK! No movement for 0.5s")
                    glitch_detected = True
                    break

            if done[0]:
                print(f"  Episode ended at step {step}")
                break

        # Summary for this lock value
        print(f"\nSummary for lock_value={lock_value}:")
        if glitch_detected:
            print("  ❌ PHYSICS GLITCH DETECTED")
        else:
            final_velocity = (x_pos - normal_positions[-1]) / 3.0
            retention = (final_velocity / normal_velocity * 100) if normal_velocity > 0 else 0
            print(f"  ✅ No glitches detected")
            print(f"  Final velocity: {final_velocity:.3f} m/s")
            print(f"  Retention: {retention:.1f}%")

        # Small delay before next test
        time.sleep(1)

    print("\n" + "="*80)
    print("DEBUGGING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    debug_ankle4_physics()