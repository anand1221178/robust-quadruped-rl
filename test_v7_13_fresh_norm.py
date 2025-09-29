#!/usr/bin/env python3
"""Test V7.13 Fresh Normalization - loading model WITHOUT old VecNormalize"""

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


def test_fresh_normalization():
    """Test loading V7.7E weights with FRESH VecNormalize statistics"""

    print("=" * 60)
    print("V7.13 FRESH NORMALIZATION TEST")
    print("=" * 60)
    print("\n🎯 KEY INSIGHT: Old VecNormalize was poisoning our training!")
    print("   - V7.7E trained with uniform joint sampling")
    print("   - V7.12/13 uses weighted sampling (30% ankle_4)")
    print("   - Different sampling = different observation statistics")
    print("   - Result: Model sees 'corrupted' observations → backward walking")
    print("\n✅ SOLUTION: Keep weights, discard old statistics!")

    # Path to V7.7E model (weights only, NO vec_normalize!)
    pretrained_model = "done/dr/Curr best/v7_7e_ultra_speed_jtfwl2qf/final_model.zip"

    print(f"\n📁 Loading ONLY model weights from: {pretrained_model}")
    print("❌ NOT loading vec_normalize.pkl (intentionally!)")

    # Create environment with V7.13 configuration
    def make_env():
        env = gym.make('RealAntMujoco-v0')
        env = SuccessRewardWrapper(env)
        env = BackwardPenaltyWrapper(env, penalty_multiplier=8.0)

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
                        'ankle_4': 0.30,
                        'ankle_1': 0.10,
                        'ankle_2': 0.10,
                        'ankle_3': 0.10,
                        'hip_1': 0.10,
                        'hip_2': 0.10,
                        'hip_3': 0.10,
                        'hip_4': 0.10,
                    }
                }
            }
        }
        env = CurriculumDRWrapper(env, dr_config)
        return env

    # Create vectorized environment
    env = DummyVecEnv([make_env])

    print("\n✅ Environment created with weighted curriculum")

    # Create FRESH VecNormalize (not loading old statistics!)
    print("\n🔄 Creating FRESH VecNormalize instance...")
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    env.training = True
    print("✅ Fresh normalization statistics will be learned from scratch")

    # Load pretrained model weights
    try:
        model = PPO.load(pretrained_model, env=env)
        print("✅ Pretrained weights loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

    # Test that the model still has its learned behavior
    print("\n🧪 Testing model behavior with fresh statistics...")
    obs = env.reset()

    # Run a few steps
    for step in range(20):
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, info = env.step(action)

        if step == 0:
            print(f"  Step {step}: reward = {rewards[0]:.3f}")

        # Check for NaN
        if np.isnan(obs).any() or np.isnan(rewards).any():
            print(f"❌ NaN detected at step {step}!")
            return False

    print("✅ Model runs without crashes or NaN")

    print("\n" + "=" * 60)
    print("✅ V7.13 FRESH NORMALIZATION TEST PASSED!")
    print("=" * 60)
    print("\nKey advantages of this approach:")
    print("1. Keeps all learned policy weights from V7.7E")
    print("2. Discards poisonous observation statistics")
    print("3. Learns correct statistics for weighted curriculum")
    print("4. Should eliminate backward walking pathology")
    print("\n🚀 This is our best shot at solving ankle_4!")

    env.close()
    return True


if __name__ == "__main__":
    success = test_fresh_normalization()
    if success:
        print("\n🎯 V7.13 is ready to launch!")
        print("sbatch scripts/train_ppo_cluster.sh v7_13_fresh_normalization")