#!/usr/bin/env python3
"""
Policy Diagnostics for RealANT Deployment

This script analyzes the trained policy to understand:
1. What action ranges it actually outputs
2. VecNormalize statistics
3. Observation ranges
4. Whether behavior matches expectations

Run this BEFORE deploying to real robot to understand what we're dealing with!
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import realant_sim  # Register environment

def analyze_policy(model_path: str, vec_normalize_path: str, num_episodes: int = 100):
    """
    Run policy in simulation and analyze its behavior.

    Args:
        model_path: Path to trained model (.zip)
        vec_normalize_path: Path to VecNormalize stats (.pkl)
        num_episodes: Number of episodes to run for analysis
    """

    print("=" * 70)
    print("RealANT Policy Diagnostics")
    print("=" * 70)

    # 1. Create environment
    print("\n1. Creating environment...")
    def make_env():
        env = gym.make('RealAntMujoco-v0', render_mode=None)
        return env

    vec_env = DummyVecEnv([make_env])

    # 2. Load VecNormalize
    print(f"\n2. Loading VecNormalize from: {vec_normalize_path}")
    vec_env = VecNormalize.load(vec_normalize_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    print(f"   VecNormalize Stats:")
    print(f"   - Observation mean shape: {vec_env.obs_rms.mean.shape}")
    print(f"   - Observation mean: {vec_env.obs_rms.mean[:10]}... (first 10)")
    print(f"   - Observation std: {np.sqrt(vec_env.obs_rms.var[:10])}... (first 10)")
    print(f"   - Norm observations: {vec_env.normalize_obs}")
    print(f"   - Clip observations: {vec_env.clip_obs}")

    # 3. Load policy
    print(f"\n3. Loading policy from: {model_path}")
    model = PPO.load(model_path)

    # Get action space info
    print(f"\n4. Action Space Analysis:")
    print(f"   - Action space: {vec_env.action_space}")
    print(f"   - Action low: {vec_env.action_space.low}")
    print(f"   - Action high: {vec_env.action_space.high}")

    # 4. Run episodes and collect statistics
    print(f"\n5. Running {num_episodes} episodes to analyze behavior...")

    all_actions = []
    all_raw_obs = []
    all_norm_obs = []
    all_rewards = []
    all_distances = []

    for episode in range(num_episodes):
        obs = vec_env.reset()
        done = False
        episode_reward = 0
        episode_distance = 0

        while not done:
            # Get unnormalized observation (if possible)
            raw_obs = vec_env.get_original_obs() if hasattr(vec_env, 'get_original_obs') else None

            # Get action from policy
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, done, info = vec_env.step(action)

            # Collect statistics
            all_actions.append(action[0])
            all_norm_obs.append(obs[0])
            if raw_obs is not None:
                all_raw_obs.append(raw_obs[0])

            episode_reward += reward[0]
            if len(info[0]) > 0 and 'x_position' in info[0]:
                episode_distance = info[0]['x_position']

        all_rewards.append(episode_reward)
        all_distances.append(episode_distance)

        if (episode + 1) % 20 == 0:
            print(f"   Episodes {episode+1}/{num_episodes} complete...")

    # 5. Analyze collected data
    all_actions = np.array(all_actions)
    all_norm_obs = np.array(all_norm_obs)

    print(f"\n{'=' * 70}")
    print("DIAGNOSTIC RESULTS")
    print(f"{'=' * 70}")

    print(f"\n📊 ACTION STATISTICS (what policy actually outputs):")
    print(f"   Shape: {all_actions.shape}")
    print(f"   Global stats:")
    print(f"     Min:  {all_actions.min():.4f}")
    print(f"     Max:  {all_actions.max():.4f}")
    print(f"     Mean: {all_actions.mean():.4f}")
    print(f"     Std:  {all_actions.std():.4f}")

    print(f"\n   Per-joint stats:")
    for i in range(8):
        joint_actions = all_actions[:, i]
        print(f"     Joint {i+1}: min={joint_actions.min():+.3f}, "
              f"max={joint_actions.max():+.3f}, "
              f"mean={joint_actions.mean():+.3f}, "
              f"std={joint_actions.std():.3f}")

    print(f"\n📈 PERFORMANCE STATISTICS:")
    print(f"   Average reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"   Average distance: {np.mean(all_distances):.3f}m ± {np.std(all_distances):.3f}m")
    print(f"   Max distance: {np.max(all_distances):.3f}m")
    print(f"   Min distance: {np.min(all_distances):.3f}m")

    print(f"\n🔍 OBSERVATION STATISTICS:")
    print(f"   Normalized observations shape: {all_norm_obs.shape}")
    print(f"   Normalized obs range: [{all_norm_obs.min():.2f}, {all_norm_obs.max():.2f}]")

    if len(all_raw_obs) > 0:
        all_raw_obs = np.array(all_raw_obs)
        print(f"   Raw observations shape: {all_raw_obs.shape}")
        print(f"   Raw obs range: [{all_raw_obs.min():.2f}, {all_raw_obs.max():.2f}]")

    # 6. Action distribution analysis
    print(f"\n📉 ACTION DISTRIBUTION:")
    bins = np.array([-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
    hist, _ = np.histogram(all_actions.flatten(), bins=bins)
    total = hist.sum()

    for i in range(len(hist)):
        pct = 100 * hist[i] / total
        bar = '█' * int(pct / 2)
        print(f"   [{bins[i]:+.2f}, {bins[i+1]:+.2f}]: {bar} {pct:.1f}%")

    print(f"\n{'=' * 70}")
    print("DEPLOYMENT RECOMMENDATIONS")
    print(f"{'=' * 70}")

    action_range = all_actions.max() - all_actions.min()
    if action_range < 1.0:
        print(f"\n⚠️  Policy uses LIMITED action range: {action_range:.2f}")
        print(f"    Policy was probably trained with action clipping or specific ranges")
        print(f"    Recommendation: Scale actions by {action_range/2:.2f} for deployment")
    else:
        print(f"\n✅ Policy uses FULL action range: {action_range:.2f}")

    if abs(all_actions.mean()) > 0.1:
        print(f"\n⚠️  Actions have non-zero mean: {all_actions.mean():.3f}")
        print(f"    This might indicate a bias that needs to be accounted for")
    else:
        print(f"\n✅ Actions are roughly centered around zero")

    print(f"\n💡 CRITICAL INSIGHTS:")
    print(f"   1. MuJoCo multiplies actions by 2 in the PID controller")
    print(f"   2. If policy outputs action=1.0, MuJoCo targets position=2.0 radians")
    print(f"   3. For real robot: target_position_rad = 2.0 * policy_action")
    print(f"   4. Then convert to degrees and Dynamixel units")

    print(f"\n✅ Diagnostics complete!")
    return {
        'actions': all_actions,
        'rewards': all_rewards,
        'distances': all_distances,
        'norm_obs': all_norm_obs,
        'vec_normalize': vec_env
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose RealANT policy before deployment")
    parser.add_argument("--model", required=True, help="Path to trained model (.zip)")
    parser.add_argument("--vec-normalize", required=True, help="Path to VecNormalize (.pkl)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to analyze")

    args = parser.parse_args()

    analyze_policy(args.model, args.vec_normalize, args.episodes)
