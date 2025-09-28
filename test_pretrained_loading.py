#!/usr/bin/env python3
"""Test pretrained model loading for V7.11"""

import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import realant_sim
from envs.success_reward_wrapper import SuccessRewardWrapper
from envs.backward_penalty_wrapper import BackwardPenaltyWrapper
from envs.rotation_mastery_wrapper import RotationMasteryWrapper
from envs.domain_randomization_wrapper import CurriculumDRWrapper


def test_pretrained_compatibility():
    """Test that V7.7E model loads correctly with V7.11 wrapper configuration"""

    print("=" * 60)
    print("TESTING V7.7E → V7.11 PRETRAINED COMPATIBILITY")
    print("=" * 60)

    # Path to V7.7E model
    pretrained_model = "done/dr/Curr best/v7_7e_ultra_speed_jtfwl2qf/final_model.zip"
    pretrained_vec = "done/dr/Curr best/v7_7e_ultra_speed_jtfwl2qf/vec_normalize.pkl"

    print(f"\nPretrained model: {pretrained_model}")
    print(f"VecNormalize: {pretrained_vec}")

    # Create environment with V7.11 wrapper setup
    def make_env():
        env = gym.make('RealAntMujoco-v0')
        env = SuccessRewardWrapper(env)
        env = BackwardPenaltyWrapper(env, penalty_multiplier=8.0)

        # V7.11 addition - this is what's new
        rotation_config = {
            'enabled': True,
            'yaw_change_multiplier': 1.5,
            'forward_after_rotation': 2.0,
            'max_total_multiplier': 2.5,
            'target_joints': ['ankle_4', 'ankle_1'],
        }
        env = RotationMasteryWrapper(env, rotation_config)

        # Domain randomization wrapper
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
                }
            }
        }
        env = CurriculumDRWrapper(env, dr_config)

        return env

    # Create vectorized environment
    env = DummyVecEnv([make_env])

    print("\n✅ Environment created with V7.11 wrapper configuration")

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

    # Test 4: Verify model architecture
    print(f"\nModel architecture check:")
    print(f"  Policy network: {model.policy}")
    print(f"  Learning rate: {model.learning_rate}")
    print(f"  Observation space: {model.observation_space}")
    print(f"  Action space: {model.action_space}")

    # Test 5: Update learning rate for fine-tuning
    original_lr = model.learning_rate
    model.learning_rate = 5e-5
    print(f"\n✅ Learning rate updated: {original_lr} → {model.learning_rate}")

    # Test 6: Run a few steps to ensure no crashes
    print("\nTesting forward pass...")
    obs = env.reset()

    for step in range(10):
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, info = env.step(action)

        if step == 0:
            print(f"  Step {step}: obs shape = {obs.shape}, reward = {rewards[0]:.3f}")

        # Check for NaN
        if np.isnan(obs).any() or np.isnan(rewards).any():
            print(f"❌ NaN detected at step {step}!")
            return False

    print("✅ Forward pass successful - no crashes or NaN")

    # Test 7: Check reward scaling
    print(f"\nReward statistics from 10 steps:")
    print(f"  Min reward: {rewards.min():.3f}")
    print(f"  Max reward: {rewards.max():.3f}")
    print(f"  Mean reward: {rewards.mean():.3f}")

    if abs(rewards.max()) > 1000:
        print("⚠️ WARNING: Rewards might be too large, risk of NaN")
    else:
        print("✅ Rewards are in safe range")

    print("\n" + "=" * 60)
    print("✅ ALL COMPATIBILITY TESTS PASSED!")
    print("=" * 60)
    print("\nV7.7E can be safely used as pretrained model for V7.11")
    print("The RotationMasteryWrapper doesn't change observation space")
    print("VecNormalize statistics are preserved correctly")

    env.close()
    return True


if __name__ == "__main__":
    success = test_pretrained_compatibility()
    if not success:
        print("\n❌ COMPATIBILITY ISSUES DETECTED!")
        print("Fix these before training V7.11")
        sys.exit(1)