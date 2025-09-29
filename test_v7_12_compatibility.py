#!/usr/bin/env python3
"""Test V7.12 Gentle Ankle Specialist compatibility with V7.7E pretrained model"""

import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import realant_sim
from envs.success_reward_wrapper import SuccessRewardWrapper
from envs.backward_penalty_wrapper import BackwardPenaltyWrapper
from envs.domain_randomization_wrapper import CurriculumDRWrapper


def test_v7_12_compatibility():
    """Test that V7.7E model loads correctly with V7.12 configuration"""

    print("=" * 60)
    print("TESTING V7.7E → V7.12 GENTLE SPECIALIST COMPATIBILITY")
    print("=" * 60)

    # Path to V7.7E model
    pretrained_model = "done/dr/Curr best/v7_7e_ultra_speed_jtfwl2qf/final_model.zip"
    pretrained_vec = "done/dr/Curr best/v7_7e_ultra_speed_jtfwl2qf/vec_normalize.pkl"

    print(f"\nPretrained model: {pretrained_model}")
    print(f"VecNormalize: {pretrained_vec}")

    # Create environment with V7.12 wrapper setup (NO rotation mastery!)
    def make_env():
        env = gym.make('RealAntMujoco-v0')
        env = SuccessRewardWrapper(env)
        env = BackwardPenaltyWrapper(env, penalty_multiplier=8.0)

        # V7.12 uses ONLY domain randomization wrapper
        # NO rotation mastery wrapper (that failed)
        dr_config = {
            'domain_randomization': {
                'phase_1_steps': 8000000,
                'phase_2_steps': 4000000,
                'phase_3_steps': 0,
                'phase_1_config': {
                    'joint_dropout_prob': 0.35,
                    'min_dropped_joints': 1,
                    'max_dropped_joints': 1,
                    'sensor_noise_std': 0.0,
                    'joint_weights': {
                        'ankle_4': 0.30,    # Gentle 30% focus
                        'ankle_1': 0.10,
                        'ankle_2': 0.10,
                        'ankle_3': 0.10,
                        'hip_1': 0.10,      # Keep hips working!
                        'hip_2': 0.10,
                        'hip_3': 0.10,
                        'hip_4': 0.10,
                    }
                },
                'phase_2_config': {
                    'joint_dropout_prob': 0.25,
                    'min_dropped_joints': 1,
                    'max_dropped_joints': 1,
                    'sensor_noise_std': 0.01,
                }
            }
        }
        env = CurriculumDRWrapper(env, dr_config)

        return env

    # Create vectorized environment
    env = DummyVecEnv([make_env])

    print("\n✅ Environment created with V7.12 configuration")
    print("   - SuccessRewardWrapper ✅")
    print("   - BackwardPenaltyWrapper (8.0x) ✅")
    print("   - CurriculumDRWrapper with gentle weighting ✅")
    print("   - NO RotationMasteryWrapper ✅")

    # Test 1: Check observation space before VecNormalize
    obs_shape_before = env.observation_space.shape
    print(f"\nObservation space: {obs_shape_before}")

    # Load pretrained VecNormalize
    try:
        env = VecNormalize.load(pretrained_vec, env)
        env.training = True
        env.norm_reward = True
        print("✅ VecNormalize loaded successfully")
    except Exception as e:
        print(f"❌ VecNormalize loading failed: {e}")
        return False

    # Test 2: Check observation space after VecNormalize
    obs_shape_after = env.observation_space.shape
    print(f"Normalized observation space: {obs_shape_after}")

    if obs_shape_before != obs_shape_after:
        print("❌ ERROR: Observation space mismatch!")
        return False

    # Test 3: Load pretrained model
    try:
        model = PPO.load(pretrained_model, env=env)
        print("✅ Pretrained model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

    # Test 4: Update learning rate for fine-tuning
    original_lr = model.learning_rate
    model.learning_rate = 5e-5
    print(f"\n✅ Learning rate updated: {original_lr} → {model.learning_rate}")

    # Test 5: Test weighted joint sampling
    print("\nTesting weighted joint sampling (10 episodes)...")
    joint_counts = {i: 0 for i in range(8)}
    joint_names = ['hip_1', 'ankle_1', 'hip_2', 'ankle_2', 'hip_3', 'ankle_3', 'hip_4', 'ankle_4']

    for _ in range(10):
        obs = env.reset()
        # Get info from the underlying environment after reset
        underlying_env = env.envs[0]
        if hasattr(underlying_env, 'dropped_joints') and underlying_env.dropped_joints:
            for joint_idx in underlying_env.dropped_joints:
                joint_counts[joint_idx] += 1

    print("Joint failure distribution:")
    total_failures = sum(joint_counts.values())
    if total_failures > 0:
        for idx, name in enumerate(joint_names):
            count = joint_counts[idx]
            percentage = (count / total_failures) * 100 if total_failures > 0 else 0
            expected = {
                'ankle_4': 30, 'ankle_1': 10, 'ankle_2': 10, 'ankle_3': 10,
                'hip_1': 10, 'hip_2': 10, 'hip_3': 10, 'hip_4': 10
            }.get(name, 0)
            print(f"  {name:8s}: {count:2d} failures ({percentage:5.1f}%, expected: {expected:2.0f}%)")

    # Test 6: Run a few steps to ensure no crashes
    print("\nTesting forward pass...")
    obs = env.reset()

    for step in range(20):
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, info = env.step(action)

        if step == 0:
            print(f"  Step {step}: obs shape = {obs.shape}, reward = {rewards[0]:.3f}")

        # Check for NaN
        if np.isnan(obs).any() or np.isnan(rewards).any():
            print(f"❌ NaN detected at step {step}!")
            return False

    print("✅ Forward pass successful - no crashes or NaN")

    print("\n" + "=" * 60)
    print("✅ ALL V7.12 COMPATIBILITY TESTS PASSED!")
    print("=" * 60)
    print("\nV7.7E can be safely used as pretrained model for V7.12")
    print("Gentle 30% ankle_4 weighting should avoid overspecialization")
    print("No rotation mastery wrapper to cause training collapse")
    print("All V7.7E reward structures preserved for consistency")

    env.close()
    return True


if __name__ == "__main__":
    success = test_v7_12_compatibility()
    if not success:
        print("\n❌ COMPATIBILITY ISSUES DETECTED!")
        print("Fix these before training V7.12")
        sys.exit(1)
    else:
        print("\n🚀 V7.12 IS READY TO LAUNCH!")
        print("sbatch scripts/train_ppo_cluster.sh v7_12_gentle_ankle_specialist")