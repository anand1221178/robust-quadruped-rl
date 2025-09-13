#!/usr/bin/env python3
"""
Debug script to find the exact cause of NaN values in V2 phase switching
Since observations are identical, the issue must be elsewhere
"""

import gymnasium as gym
import numpy as np
import torch
import sys
sys.path.append('/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/src')

# Import RealAnt environments
import realant_sim
from envs.success_reward_wrapper import SuccessRewardWrapper
from envs.systematic_curriculum_wrapper import SystematicCurriculumWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

def debug_nan_cause():
    """Find the exact cause of NaN values during phase switching"""

    print("🕵️ DEBUGGING NaN CAUSE IN PHASE SWITCHING...")

    # Create Phase 0 environment (baseline)
    def make_phase0_env():
        env = gym.make('RealAntMujoco-v0')
        env = SuccessRewardWrapper(env)
        env = Monitor(env)
        return env

    phase0_env = DummyVecEnv([make_phase0_env])
    phase0_env = VecNormalize(phase0_env, norm_obs=True, norm_reward=True)

    print("✅ Phase 0 environment created")

    # Create a simple PPO model
    model = PPO('MlpPolicy', phase0_env, verbose=0, learning_rate=0.0003, batch_size=64, n_steps=64)

    print("✅ PPO model created")

    # Train for a few steps in Phase 0
    print("🏃 Training Phase 0 for 500 steps...")
    model.learn(total_timesteps=500, progress_bar=False)

    print("✅ Phase 0 training completed")

    # Now test the model with Phase 0 environment
    obs = phase0_env.reset()
    print(f"📊 Phase 0 obs shape: {obs.shape}, range: [{np.min(obs):.3f}, {np.max(obs):.3f}]")

    # Get action from model
    action, _states = model.predict(obs, deterministic=True)
    print(f"🎯 Phase 0 action: {action}")
    print(f"   Action shape: {action.shape}, range: [{np.min(action):.3f}, {np.max(action):.3f}]")
    print(f"   Any NaN in action: {np.any(np.isnan(action))}")

    # Check model weights for NaN
    has_nan_weights = False
    for name, param in model.policy.named_parameters():
        if torch.any(torch.isnan(param)):
            print(f"🚨 NaN found in model weights: {name}")
            has_nan_weights = True

    if not has_nan_weights:
        print("✅ No NaN in model weights after Phase 0")

    # Now create Phase 1 environment (curriculum)
    curriculum_config = {
        'normal_walking_duration': 0,  # V2 mode
        'single_joint_duration': 500,
        'anatomical_combinations': [["hip_1", "ankle_1"]],
        'diagonal_combinations': [["hip_1", "hip_4"]],
        'functional_combinations': [["hip_1", "hip_2"]],
        'critical_triple_combinations': []
    }

    def make_curriculum_env():
        env = gym.make('RealAntMujoco-v0')
        env = SuccessRewardWrapper(env)
        env = SystematicCurriculumWrapper(env, curriculum_config)
        env = Monitor(env)
        return env

    curriculum_env = DummyVecEnv([make_curriculum_env])

    print("🎯 Created curriculum environment")

    # CRITICAL TEST: Create new VecNormalize vs reusing old one
    print("\n🧪 TESTING VECNORMALIZE APPROACHES:")

    # Approach 1: Create fresh VecNormalize
    fresh_vec_env = VecNormalize(DummyVecEnv([make_curriculum_env]), norm_obs=True, norm_reward=True)
    fresh_obs = fresh_vec_env.reset()

    print(f"   Fresh VecNormalize obs range: [{np.min(fresh_obs):.3f}, {np.max(fresh_obs):.3f}]")

    try:
        fresh_action, _ = model.predict(fresh_obs, deterministic=True)
        print(f"   Fresh VecNormalize action: SUCCESS, range: [{np.min(fresh_action):.3f}, {np.max(fresh_action):.3f}]")
        fresh_nan = np.any(np.isnan(fresh_action))
        print(f"   Fresh approach NaN: {fresh_nan}")
    except Exception as e:
        print(f"   Fresh VecNormalize: FAILED - {e}")

    # Approach 2: Copy VecNormalize stats (our current approach)
    copy_vec_env = VecNormalize(DummyVecEnv([make_curriculum_env]), training=True, norm_obs=True, norm_reward=True)

    # Copy the statistics
    copy_vec_env.obs_rms = phase0_env.obs_rms
    copy_vec_env.ret_rms = phase0_env.ret_rms

    copy_obs = copy_vec_env.reset()
    print(f"   Copied VecNormalize obs range: [{np.min(copy_obs):.3f}, {np.max(copy_obs):.3f}]")

    try:
        copy_action, _ = model.predict(copy_obs, deterministic=True)
        print(f"   Copied VecNormalize action: SUCCESS, range: [{np.min(copy_action):.3f}, {np.max(copy_action):.3f}]")
        copy_nan = np.any(np.isnan(copy_action))
        print(f"   Copied approach NaN: {copy_nan}")
    except Exception as e:
        print(f"   Copied VecNormalize: FAILED - {e}")

    # Approach 3: No VecNormalize transfer at all
    no_vec_env = DummyVecEnv([make_curriculum_env])
    no_vec_obs = no_vec_env.reset()
    print(f"   No VecNormalize obs range: [{np.min(no_vec_obs):.3f}, {np.max(no_vec_obs):.3f}]")

    try:
        no_vec_action, _ = model.predict(no_vec_obs, deterministic=True)
        print(f"   No VecNormalize action: SUCCESS, range: [{np.min(no_vec_action):.3f}, {np.max(no_vec_action):.3f}]")
        no_vec_nan = np.any(np.isnan(no_vec_action))
        print(f"   No VecNormalize approach NaN: {no_vec_nan}")
    except Exception as e:
        print(f"   No VecNormalize: FAILED - {e}")

    # Check observation stats
    print(f"\n📈 VECNORMALIZE STATS ANALYSIS:")
    print(f"   Phase 0 obs_rms mean: {np.mean(phase0_env.obs_rms.mean):.6f}")
    print(f"   Phase 0 obs_rms var: {np.mean(phase0_env.obs_rms.var):.6f}")

    fresh_vec_env.reset()  # Initialize stats
    print(f"   Fresh obs_rms mean: {np.mean(fresh_vec_env.obs_rms.mean):.6f}")
    print(f"   Fresh obs_rms var: {np.mean(fresh_vec_env.obs_rms.var):.6f}")

    # Clean up
    phase0_env.close()
    fresh_vec_env.close()
    copy_vec_env.close()
    no_vec_env.close()

    print(f"\n🎯 CONCLUSION:")
    print(f"   The NaN issue is likely in the VecNormalize statistics transfer!")

if __name__ == "__main__":
    debug_nan_cause()