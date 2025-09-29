#!/usr/bin/env python3
"""Debug script to track exact joint failures and robot behavior"""

import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import realant_sim


def debug_joint_failure(model_path, vec_path, joint_to_test, delay_steps=0):
    """Debug a specific joint failure with detailed logging"""

    print(f"=" * 60)
    print(f"DEBUGGING JOINT FAILURE: {joint_to_test}")
    print(f"Delay steps: {delay_steps}")
    print(f"=" * 60)

    # Joint mapping
    joint_names = ['hip_1', 'ankle_1', 'hip_2', 'ankle_2', 'hip_3', 'ankle_3', 'hip_4', 'ankle_4']
    joint_indices = {name: idx for idx, name in enumerate(joint_names)}

    if joint_to_test not in joint_indices:
        print(f"Invalid joint: {joint_to_test}. Valid joints: {list(joint_indices.keys())}")
        return

    failed_joint_idx = joint_indices[joint_to_test]

    # Create environment
    env = gym.make('RealAntMujoco-v0')
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(vec_path, env)
    model = PPO.load(model_path)

    obs = env.reset()

    # Data collection
    positions = []
    joint_positions = []
    joint_velocities = []
    actions_taken = []
    rewards_received = []

    print(f"\nRunning 300 steps with {joint_to_test} failure...")
    print(f"Joint will {'be locked immediately' if delay_steps == 0 else f'lock after {delay_steps} steps'}")

    for step in range(300):
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        original_action = action.copy()

        # Apply joint failure if delay period has passed
        joint_locked = step >= delay_steps
        if joint_locked:
            action[0][failed_joint_idx] = 0.0  # Lock the joint

        # Step environment
        obs, reward, done, info = env.step(action)

        # Extract robot state
        if hasattr(env.envs[0].env, 'data'):
            robot_data = env.envs[0].env.data
            x_pos = robot_data.qpos[0]
            y_pos = robot_data.qpos[1]

            # Get all joint positions and velocities
            joint_pos = robot_data.qpos[7:15].copy()  # 8 joint positions
            joint_vel = robot_data.qvel[6:14].copy()  # 8 joint velocities

        else:
            x_pos, y_pos = 0, 0
            joint_pos = np.zeros(8)
            joint_vel = np.zeros(8)

        # Store data
        positions.append([x_pos, y_pos])
        joint_positions.append(joint_pos)
        joint_velocities.append(joint_vel)
        actions_taken.append(original_action[0])
        rewards_received.append(reward[0])

        # Print detailed info every 50 steps or when joint locks
        if step % 50 == 0 or (step == delay_steps and delay_steps > 0):
            status = "🔒 LOCKED" if joint_locked else "🔓 FREE"
            print(f"\nStep {step:3d}: {status}")
            print(f"  Position: x={x_pos:6.3f}, y={y_pos:6.3f}")
            print(f"  {joint_to_test} position: {joint_pos[failed_joint_idx]:6.3f} rad")
            print(f"  {joint_to_test} velocity: {joint_vel[failed_joint_idx]:6.3f} rad/s")
            print(f"  {joint_to_test} action: {original_action[0][failed_joint_idx]:6.3f} → {action[0][failed_joint_idx]:6.3f}")
            print(f"  Reward: {reward[0]:6.3f}")

            # Check if robot is spinning
            if len(positions) >= 10:
                recent_x = [p[0] for p in positions[-10:]]
                recent_y = [p[1] for p in positions[-10:]]
                x_range = max(recent_x) - min(recent_x)
                y_range = max(recent_y) - min(recent_y)
                if x_range < 0.1 and y_range > 0.2:
                    print(f"  🌀 SPINNING DETECTED! x_range={x_range:.3f}, y_range={y_range:.3f}")

    # Final analysis
    print(f"\n" + "=" * 60)
    print(f"FINAL ANALYSIS FOR {joint_to_test}")
    print(f"=" * 60)

    positions = np.array(positions)
    total_distance = np.linalg.norm(positions[-1] - positions[0])
    forward_distance = positions[-1][0] - positions[0][0]
    lateral_distance = abs(positions[-1][1] - positions[0][1])

    total_time = len(positions) * 0.05  # 50Hz
    forward_velocity = forward_distance / total_time

    print(f"Total distance traveled: {total_distance:.3f}m")
    print(f"Forward distance: {forward_distance:.3f}m")
    print(f"Lateral movement: {lateral_distance:.3f}m")
    print(f"Forward velocity: {forward_velocity:.3f} m/s")
    print(f"Average reward: {np.mean(rewards_received):.3f}")

    # Check for spinning behavior
    x_positions = positions[:, 0]
    y_positions = positions[:, 1]
    x_range = np.max(x_positions) - np.min(x_positions)
    y_range = np.max(y_positions) - np.min(y_positions)

    if lateral_distance > forward_distance:
        print(f"🌀 ROTATION/SPINNING BEHAVIOR DETECTED")
        print(f"   Lateral movement ({lateral_distance:.3f}m) > Forward movement ({forward_distance:.3f}m)")

    if x_range < 1.0 and y_range > 1.0:
        print(f"🌀 CIRCULAR MOTION DETECTED")
        print(f"   X range: {x_range:.3f}m, Y range: {y_range:.3f}m")

    # Joint analysis
    joint_positions = np.array(joint_positions)
    failed_joint_final_pos = joint_positions[-1, failed_joint_idx]
    failed_joint_range = np.max(joint_positions[:, failed_joint_idx]) - np.min(joint_positions[:, failed_joint_idx])

    print(f"\n{joint_to_test} Analysis:")
    print(f"  Final position: {failed_joint_final_pos:.3f} rad ({np.degrees(failed_joint_final_pos):.1f}°)")
    print(f"  Position range: {failed_joint_range:.3f} rad")

    return {
        'joint': joint_to_test,
        'forward_velocity': forward_velocity,
        'total_distance': total_distance,
        'lateral_distance': lateral_distance,
        'spinning': lateral_distance > forward_distance,
        'final_joint_position': failed_joint_final_pos,
        'positions': positions
    }


def test_all_joints(model_path, vec_path, delay_steps=0):
    """Test all joints with detailed logging"""

    print(f"🔍 TESTING ALL JOINTS WITH {delay_steps} DELAY STEPS")
    print(f"Model: {model_path}")

    joint_names = ['hip_1', 'ankle_1', 'hip_2', 'ankle_2', 'hip_3', 'ankle_3', 'hip_4', 'ankle_4']
    results = {}

    for joint in joint_names:
        print(f"\n{'='*80}")
        result = debug_joint_failure(model_path, vec_path, joint, delay_steps)
        results[joint] = result

    # Summary
    print(f"\n\n🏆 SUMMARY OF ALL JOINT FAILURES")
    print(f"{'='*80}")
    print(f"{'Joint':<10} {'Velocity':<10} {'Distance':<10} {'Lateral':<10} {'Spinning'}")
    print(f"{'='*80}")

    for joint, result in results.items():
        spinning_icon = "🌀" if result['spinning'] else "→"
        print(f"{joint:<10} {result['forward_velocity']:>8.3f} {result['total_distance']:>8.3f} {result['lateral_distance']:>8.3f} {spinning_icon}")

    return results


if __name__ == "__main__":
    model_path = "experiments/v7_13_fresh_normalization_57ea0nls/final_model.zip"
    vec_path = "experiments/v7_13_fresh_normalization_57ea0nls/vec_normalize.pkl"

    # Test with immediate locking (no delay)
    results = test_all_joints(model_path, vec_path, delay_steps=0)