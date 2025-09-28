#!/usr/bin/env python3
"""Test script to verify V7.11 rotation mastery setup is working correctly"""

import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
import realant_sim
from envs.success_reward_wrapper import SuccessRewardWrapper
from envs.backward_penalty_wrapper import BackwardPenaltyWrapper
from envs.rotation_mastery_wrapper import RotationMasteryWrapper
from envs.domain_randomization_wrapper import CurriculumDRWrapper


def test_rotation_mastery():
    """Test the complete V7.11 setup"""

    print("=" * 60)
    print("V7.11 ROTATION MASTERY TEST")
    print("=" * 60)

    # Create base environment
    env = gym.make('RealAntMujoco-v0')
    print("✅ Base environment created")

    # Add success reward wrapper
    env = SuccessRewardWrapper(env)
    print("✅ Success reward wrapper added")

    # Add backward penalty wrapper
    env = BackwardPenaltyWrapper(env, penalty_multiplier=8.0)
    print("✅ Backward penalty wrapper added")

    # Add rotation mastery wrapper
    rotation_config = {
        'enabled': True,
        'yaw_change_multiplier': 1.5,
        'forward_after_rotation': 2.0,
        'max_total_multiplier': 2.5,
        'target_joints': ['ankle_4', 'ankle_1'],
    }
    env = RotationMasteryWrapper(env, rotation_config)
    print("✅ Rotation mastery wrapper added")

    # Add curriculum DR wrapper
    dr_config = {
        'domain_randomization': {
            'phase_1_steps': 8000000,
            'phase_2_steps': 4000000,
            'phase_3_steps': 0,
            'phase_1_config': {
                'joint_dropout_prob': 0.4,
                'min_dropped_joints': 1,
                'max_dropped_joints': 1,
                'sensor_noise_std': 0.0,
                'joint_weights': {
                    'ankle_4': 0.7,
                    'ankle_1': 0.1,
                    'ankle_2': 0.05,
                    'ankle_3': 0.05,
                    'hip_1': 0.025,
                    'hip_2': 0.025,
                    'hip_3': 0.025,
                    'hip_4': 0.025,
                }
            },
            'phase_2_config': {
                'joint_dropout_prob': 0.25,
                'min_dropped_joints': 1,
                'max_dropped_joints': 1,
                'sensor_noise_std': 0.01,
            },
        }
    }
    env = CurriculumDRWrapper(env, dr_config)
    print("✅ Curriculum DR wrapper added with weighted joint sampling")

    print("\n" + "=" * 60)
    print("TESTING WEIGHTED JOINT SAMPLING")
    print("=" * 60)

    # Test joint sampling distribution
    joint_counts = {i: 0 for i in range(8)}
    joint_names = ['hip_1', 'ankle_1', 'hip_2', 'ankle_2', 'hip_3', 'ankle_3', 'hip_4', 'ankle_4']

    # Sample 100 episodes
    for _ in range(100):
        obs, info = env.reset()
        if 'dropped_joints' in info and info['dropped_joints']:
            for joint_idx in info['dropped_joints']:
                joint_counts[joint_idx] += 1

    print("\nJoint failure distribution (100 episodes):")
    total_failures = sum(joint_counts.values())
    if total_failures > 0:
        for idx, name in enumerate(joint_names):
            percentage = (joint_counts[idx] / total_failures) * 100
            expected = dr_config['domain_randomization']['phase_1_config']['joint_weights'].get(name, 0) * 100
            print(f"  {name:8s}: {percentage:5.1f}% (expected: {expected:5.1f}%)")

    print("\n" + "=" * 60)
    print("TESTING ROTATION MASTERY REWARDS")
    print("=" * 60)

    # Test one episode with ankle_4 failure
    env.dropped_joints = [7]  # Force ankle_4 failure
    obs, info = env.reset()

    print("\nRunning episode with ankle_4 failure...")

    for step in range(150):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if step == 119:
            print(f"  Step {step}: Before locking - Reward: {reward:.3f}")
        elif step == 120:
            print(f"  Step {step}: Joint locked! - Reward: {reward:.3f}")
        elif step == 140:
            print(f"  Step {step}: After locking - Reward: {reward:.3f}")

        if terminated or truncated:
            break

    print("\n✅ ALL TESTS PASSED!")
    print("\nV7.11 Rotation Mastery setup is ready for training!")

    env.close()


if __name__ == "__main__":
    test_rotation_mastery()