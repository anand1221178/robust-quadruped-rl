#!/usr/bin/env python3
"""
CRITICAL TEST: Does SR2L help without VecNormalize?
Test SR2L model on different noise types WITHOUT VecNormalize
"""

import gymnasium as gym
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
import json
from datetime import datetime

# Import environment
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from envs.success_reward_wrapper import SuccessRewardWrapper
import realant_sim

def add_noise(obs, noise_type, intensity):
    """Add different types of noise to observations"""
    obs_copy = obs.copy()
    joint_indices = list(range(13, 29))  # Joint observations

    if noise_type == 'gaussian':
        for idx in joint_indices:
            obs_copy[idx] += np.random.normal(0, intensity)
    elif noise_type == 'poisson':
        for idx in joint_indices:
            obs_copy[idx] += np.random.poisson(intensity) - intensity
    elif noise_type == 'salt_pepper':
        for idx in joint_indices:
            if np.random.random() < intensity:
                obs_copy[idx] = np.random.choice([obs[idx] - 1, obs[idx] + 1])

    return obs_copy

def test_model(model_path, noise_type, intensity, num_episodes=50, use_vecnormalize=True):
    """Test model with specific noise type"""

    # Load model
    model = PPO.load(model_path)

    # Create environment (no VecNormalize!)
    env = gym.make('RealAntMujoco-v0')
    env = SuccessRewardWrapper(env)

    # If testing without VecNormalize, don't load vec_normalize.pkl
    # If testing with VecNormalize, load it
    if use_vecnormalize:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        vec_normalize_path = Path(model_path).parent / "vec_normalize.pkl"
        if vec_normalize_path.exists():
            env = DummyVecEnv([lambda: env])
            env = VecNormalize.load(str(vec_normalize_path), env)
            env.training = False
            env.norm_reward = False
            print(f"✅ Loaded VecNormalize from {vec_normalize_path}")
        else:
            print(f"⚠️  VecNormalize not found, testing without it")
            use_vecnormalize = False

    distances = []

    for episode in range(num_episodes):
        if use_vecnormalize:
            obs = env.reset()
            # Unwrap to get base environment
            base_env = env.envs[0]
            while hasattr(base_env, 'env'):
                base_env = base_env.env
        else:
            obs, _ = env.reset()
            # Unwrap to get base environment
            base_env = env
            while hasattr(base_env, 'env') and not hasattr(base_env, 'get_body_com'):
                base_env = base_env.env

        done = False
        positions = [base_env.get_body_com("torso")[0]]

        while not done:
            # Add noise to observations
            noisy_obs = add_noise(obs[0] if use_vecnormalize else obs, noise_type, intensity)

            # Get action
            action, _ = model.predict(noisy_obs if not use_vecnormalize else np.array([noisy_obs]), deterministic=True)

            # Step
            if use_vecnormalize:
                obs, reward, done, info = env.step(action)
                done = done[0]
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            # Track position
            if not done:
                current_pos = base_env.get_body_com("torso")[0]
                positions.append(current_pos)

        distance = positions[-1] - positions[0]
        distances.append(distance)

        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode+1}/{num_episodes}: {np.mean(distances[-10:]):.2f}m avg")

    return {
        'mean': np.mean(distances),
        'std': np.std(distances),
        'distances': distances
    }

def main():
    print("="*80)
    print("CRITICAL TEST: SR2L Value Without VecNormalize")
    print("="*80)

    # Model paths
    m1_path = "done/ppo_baseline_ueqbjf2x/best_model/best_model.zip"
    m2_path = "done/ppo_sr2l_forward_m7gtjtpa/final_model.zip"

    noise_types = ['gaussian', 'poisson', 'salt_pepper']
    intensity = 0.1  # σ=0.10

    results = {}

    for model_name, model_path in [('M1_baseline', m1_path), ('M2_sr2l', m2_path)]:
        print(f"\n{'='*80}")
        print(f"Testing {model_name}")
        print(f"{'='*80}")

        results[model_name] = {}

        for with_vec in [True, False]:
            vec_label = 'with_vecnormalize' if with_vec else 'without_vecnormalize'
            print(f"\n{vec_label.upper()}:")
            results[model_name][vec_label] = {}

            for noise_type in noise_types:
                print(f"\n  {noise_type} (σ=0.10):")
                result = test_model(model_path, noise_type, intensity, num_episodes=50, use_vecnormalize=with_vec)
                results[model_name][vec_label][noise_type] = result
                print(f"  ✅ Result: {result['mean']:.2f}m ± {result['std']:.2f}")

    # Save results
    output_dir = Path("evaluations/sr2l_vecnormalize_test")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"sr2l_vecnormalize_test_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS: Does SR2L add value without VecNormalize?")
    print("="*80)

    for noise_type in noise_types:
        print(f"\n{noise_type.upper()}:")
        m1_with = results['M1_baseline']['with_vecnormalize'][noise_type]['mean']
        m1_without = results['M1_baseline']['without_vecnormalize'][noise_type]['mean']
        m2_with = results['M2_sr2l']['with_vecnormalize'][noise_type]['mean']
        m2_without = results['M2_sr2l']['without_vecnormalize'][noise_type]['mean']

        print(f"  M1 WITH Vec:    {m1_with:.2f}m")
        print(f"  M1 WITHOUT Vec: {m1_without:.2f}m")
        print(f"  M2 WITH Vec:    {m2_with:.2f}m")
        print(f"  M2 WITHOUT Vec: {m2_without:.2f}m")

        # The KEY comparison: WITHOUT VecNormalize
        improvement = ((m2_without - m1_without) / m1_without) * 100
        print(f"  SR2L improvement (no Vec): {improvement:+.1f}%")

        if improvement > 5:
            print(f"  ✅ SR2L HELPS without VecNormalize!")
        elif improvement < -5:
            print(f"  ❌ SR2L HURTS without VecNormalize")
        else:
            print(f"  ⚠️  No significant difference")

if __name__ == "__main__":
    main()
