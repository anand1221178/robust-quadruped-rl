#!/usr/bin/env python3
"""
Debug script to identify exact observation differences between environments
that cause NaN values in neural network
"""

import gymnasium as gym
import numpy as np
import sys
sys.path.append('/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/src')

# Import RealAnt environments
import realant_sim
from envs.success_reward_wrapper import SuccessRewardWrapper
from envs.systematic_curriculum_wrapper import SystematicCurriculumWrapper

def compare_environments():
    """Compare observations from baseline vs curriculum environments"""

    print("🔍 ANALYZING OBSERVATION DIFFERENCES...")

    # Create baseline environment (Phase 0 environment)
    baseline_env = gym.make('RealAntMujoco-v0')
    baseline_env = SuccessRewardWrapper(baseline_env)

    # Create curriculum environment (Phase 1+ environment)
    curriculum_config = {
        'normal_walking_duration': 0,  # V2 mode
        'single_joint_duration': 3000000,
        'anatomical_combinations': [["hip_1", "ankle_1"], ["hip_2", "ankle_2"]],
        'diagonal_combinations': [["hip_1", "hip_4"]],
        'functional_combinations': [["hip_1", "hip_2"]],
        'critical_triple_combinations': []
    }
    curriculum_env = gym.make('RealAntMujoco-v0')
    curriculum_env = SuccessRewardWrapper(curriculum_env)
    curriculum_env = SystematicCurriculumWrapper(curriculum_env, curriculum_config)

    print(f"\n📊 ENVIRONMENT COMPARISON:")
    print(f"   Baseline: RealAnt + SuccessRewardWrapper")
    print(f"   Curriculum: RealAnt + SuccessRewardWrapper + SystematicCurriculumWrapper")

    # Reset both environments
    baseline_obs, baseline_info = baseline_env.reset(seed=42)
    curriculum_obs, curriculum_info = curriculum_env.reset(seed=42)

    print(f"\n🔢 OBSERVATION ANALYSIS:")
    print(f"   Baseline shape: {baseline_obs.shape}")
    print(f"   Curriculum shape: {curriculum_obs.shape}")
    print(f"   Shapes match: {baseline_obs.shape == curriculum_obs.shape}")

    # Check observation differences
    obs_diff = np.abs(baseline_obs - curriculum_obs)
    max_diff = np.max(obs_diff)
    mean_diff = np.mean(obs_diff)

    print(f"\n📈 OBSERVATION VALUES:")
    print(f"   Max difference: {max_diff:.6f}")
    print(f"   Mean difference: {mean_diff:.6f}")
    print(f"   Non-zero differences: {np.sum(obs_diff > 1e-10)}")

    if max_diff > 1e-10:
        print(f"\n⚠️  OBSERVATION DIFFERENCES FOUND:")
        different_indices = np.where(obs_diff > 1e-10)[0]
        for i in different_indices[:5]:  # Show first 5 differences
            print(f"   Index {i}: baseline={baseline_obs[i]:.6f}, curriculum={curriculum_obs[i]:.6f}, diff={obs_diff[i]:.6f}")
    else:
        print(f"\n✅ OBSERVATIONS ARE IDENTICAL!")

    # Check observation ranges for VecNormalize compatibility
    print(f"\n📊 OBSERVATION RANGES:")
    print(f"   Baseline - min: {np.min(baseline_obs):.6f}, max: {np.max(baseline_obs):.6f}")
    print(f"   Curriculum - min: {np.min(curriculum_obs):.6f}, max: {np.max(curriculum_obs):.6f}")

    # Check for any extreme values that could cause NaN
    baseline_extreme = np.any(np.abs(baseline_obs) > 100)
    curriculum_extreme = np.any(np.abs(curriculum_obs) > 100)

    print(f"\n🚨 EXTREME VALUE CHECK:")
    print(f"   Baseline has extreme values (>100): {baseline_extreme}")
    print(f"   Curriculum has extreme values (>100): {curriculum_extreme}")

    if baseline_extreme or curriculum_extreme:
        print(f"   Baseline extreme indices: {np.where(np.abs(baseline_obs) > 100)[0]}")
        print(f"   Curriculum extreme indices: {np.where(np.abs(curriculum_obs) > 100)[0]}")

    # Check action spaces
    print(f"\n🎯 ACTION SPACE COMPARISON:")
    print(f"   Baseline action space: {baseline_env.action_space}")
    print(f"   Curriculum action space: {curriculum_env.action_space}")
    print(f"   Action spaces match: {baseline_env.action_space == curriculum_env.action_space}")

    # Test a few environment steps
    print(f"\n🚶 ENVIRONMENT STEP TEST:")
    action = np.zeros(8)  # Neutral action

    baseline_obs2, baseline_reward, baseline_done, baseline_trunc, baseline_info2 = baseline_env.step(action)
    curriculum_obs2, curriculum_reward, curriculum_done, curriculum_trunc, curriculum_info2 = curriculum_env.step(action)

    step_obs_diff = np.abs(baseline_obs2 - curriculum_obs2)
    step_max_diff = np.max(step_obs_diff)

    print(f"   After 1 step - max obs difference: {step_max_diff:.6f}")
    print(f"   Baseline reward: {baseline_reward:.6f}")
    print(f"   Curriculum reward: {curriculum_reward:.6f}")

    # Clean up
    baseline_env.close()
    curriculum_env.close()

    return max_diff, step_max_diff

if __name__ == "__main__":
    max_diff, step_max_diff = compare_environments()

    if max_diff < 1e-10 and step_max_diff < 1e-10:
        print(f"\n✅ CONCLUSION: Observations are identical!")
        print(f"   NaN issue must be coming from somewhere else...")
    else:
        print(f"\n🚨 CONCLUSION: Observation differences detected!")
        print(f"   Initial diff: {max_diff:.6f}")
        print(f"   Step diff: {step_max_diff:.6f}")
        print(f"   This could be causing the NaN values!")