#!/usr/bin/env python3
"""
Reconstruct learning curves with MULTI-SEED AVERAGING (5 seeds per model)
Addresses supervisor requirement: Show 5-seed averaging with error bands
Output: learning_curves_reconstructed.pdf (single plot, rewards only)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import realant_sim

# Multi-seed configurations - models in experiments/
MODELS = {
    'M1_baseline': {
        'seeds': [42, 123, 456, 789, 999],
        'pattern': 'M1_baseline_seed{seed}_*',
        'label': 'M1 (Baseline)',
        'color': '#2E86AB',
        'total_steps': 32_000_000,
        'eval_interval': 1_000_000,
    },
    'M2_sr2l': {
        'seeds': [42, 123, 456, 789, 999],
        'pattern': 'M2_sr2l_seed{seed}_*',
        'label': 'M2 (SR2L)',
        'color': '#9C27B0',
        'total_steps': 32_000_000,
        'eval_interval': 1_000_000,
    },
    'M3_dr': {
        'seeds': [42, 123, 456, 789, 999],
        'pattern': 'M3_dr_seed{seed}_*',
        'label': 'M3 (DR)',
        'color': '#FF6F00',
        'total_steps': 32_000_000,
        'eval_interval': 1_000_000,
    },
    'M4_combo': {
        'seeds': [42, 123, 456, 789, 999],
        'pattern': 'M4_combo_seed{seed}_*',
        'label': 'M4 (Combined)',
        'color': '#D32F2F',
        'total_steps': 30_000_000,
        'eval_interval': 1_000_000,
    }
}

def evaluate_checkpoint_raw_rewards(model_path, vec_normalize_path, n_eval_episodes=10):
    """Evaluate checkpoint and get TRAINING-SCALE rewards (not normalized)"""
    try:
        # Create environment WITH SuccessRewardWrapper (matches training)
        from envs.success_reward_wrapper import SuccessRewardWrapper
        base_env = gym.make('RealAntMujoco-v0')
        base_env = SuccessRewardWrapper(base_env)  # CRITICAL: Must match training
        env = DummyVecEnv([lambda: base_env])

        # DON'T load VecNormalize for reward calculation
        # We want RAW rewards from SuccessRewardWrapper

        # But DO normalize observations for the policy
        if Path(vec_normalize_path).exists():
            vec_norm = VecNormalize.load(vec_normalize_path, env)
            vec_norm.training = False
            vec_norm.norm_reward = False  # Don't normalize rewards
            env = vec_norm

        # Load model
        model = PPO.load(model_path, env=env)

        # Evaluate
        episode_rewards = []

        for _ in range(n_eval_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)

                # Get RAW reward from SuccessRewardWrapper
                # VecNormalize might still normalize it, so we need the raw value
                if 'raw_reward' in info[0]:
                    episode_reward += info[0]['raw_reward']
                else:
                    episode_reward += reward[0]

                if done[0]:
                    break

            episode_rewards.append(episode_reward)

        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards)
        }

    except Exception as e:
        print(f"    Error evaluating: {e}")
        import traceback
        traceback.print_exc()
        return None

def find_model_dir(pattern, seed):
    """Find model directory matching pattern"""
    experiments_dir = Path(__file__).parent.parent / 'experiments'
    search_pattern = pattern.format(seed=seed)
    matches = list(experiments_dir.glob(search_pattern))

    if len(matches) == 0:
        return None
    return matches[0]

def reconstruct_model_multiseed(model_key, config):
    """Reconstruct learning curve averaging across all seeds"""
    print(f"\n{'='*80}")
    print(f"Reconstructing: {config['label']}")
    print('='*80)

    all_seed_results = {}  # {seed: [{steps, mean_reward, ...}, ...]}

    # Collect data from each seed
    for seed in config['seeds']:
        print(f"\n  Seed {seed}...")

        model_dir = find_model_dir(config['pattern'], seed)
        if not model_dir:
            print(f"    ❌ Not found")
            continue

        checkpoint_dir = model_dir / 'checkpoints'
        vec_normalize_path = model_dir / 'vec_normalize.pkl'

        if not checkpoint_dir.exists():
            print(f"    ❌ No checkpoints")
            continue

        # Evaluate checkpoints
        seed_results = []
        checkpoints = sorted(checkpoint_dir.glob('checkpoint_*_steps.zip'))

        for ckpt in tqdm(checkpoints, desc=f"    Seed {seed}", leave=False):
            # Extract step number
            step_str = ckpt.stem.split('_')[1]
            steps = int(step_str)

            # Evaluate
            result = evaluate_checkpoint_raw_rewards(str(ckpt), str(vec_normalize_path), n_eval_episodes=10)

            if result:
                seed_results.append({
                    'steps': steps,
                    'mean_reward': result['mean_reward']
                })

        if seed_results:
            all_seed_results[seed] = seed_results
            print(f"    ✅ {len(seed_results)} checkpoints")

    if not all_seed_results:
        print(f"  ❌ No data collected!")
        return None

    # Aggregate across seeds
    print(f"\n  Aggregating across {len(all_seed_results)} seeds...")

    all_steps = sorted(set(
        r['steps'] for seed_results in all_seed_results.values()
        for r in seed_results
    ))

    aggregated = []
    for steps in all_steps:
        rewards_at_step = []

        for seed, seed_results in all_seed_results.items():
            matching = [r for r in seed_results if r['steps'] == steps]
            if matching:
                rewards_at_step.append(matching[0]['mean_reward'])

        if len(rewards_at_step) >= 3:  # Need at least 3 seeds
            aggregated.append({
                'steps': steps,
                'mean_reward': np.mean(rewards_at_step),
                'std_reward': np.std(rewards_at_step),
                'num_seeds': len(rewards_at_step)
            })

    print(f"  ✅ Aggregated {len(aggregated)} checkpoints")
    return aggregated

def create_learning_curves():
    """Create multi-seed learning curves (SINGLE PLOT - rewards only)"""
    print("\n" + "="*80)
    print("MULTI-SEED LEARNING CURVES (n=5 seeds)")
    print("="*80)

    # Reconstruct all models
    all_results = {}
    for model_key, config in MODELS.items():
        results = reconstruct_model_multiseed(model_key, config)
        if results:
            all_results[model_key] = results

    if not all_results:
        print("\n❌ No data reconstructed!")
        return

    # Create single plot (matching current paper figure)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle('Training Dynamics: Multi-Seed Average (n=5 seeds)\n(All Models Trained to 32M Steps)',
                 fontsize=14, fontweight='bold')

    # Plot each model
    for model_key, results in all_results.items():
        config = MODELS[model_key]

        steps = np.array([r['steps'] for r in results]) / 1e6
        rewards = np.array([r['mean_reward'] for r in results])
        stds = np.array([r['std_reward'] for r in results])

        # Main line
        ax.plot(steps, rewards,
                label=config['label'],
                color=config['color'],
                linewidth=2.5,
                marker='o',
                markersize=4)

        # Error bands (±1 SD across seeds)
        ax.fill_between(steps,
                         rewards - stds,
                         rewards + stds,
                         color=config['color'],
                         alpha=0.15)

    ax.set_xlabel('Training Steps (millions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Episode Reward', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=0)

    plt.tight_layout()

    # Save
    output_path = Path(__file__).parent / 'learning_curves_reconstructed.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")

    # Save data
    data_path = Path(__file__).parent / 'learning_curves_multiseed_data.json'
    with open(data_path, 'w') as f:
        json.dump({k: v for k, v in all_results.items()}, f, indent=2, default=lambda x: float(x) if isinstance(x, np.number) else x)
    print(f"✅ Saved: {data_path}")

if __name__ == '__main__':
    create_learning_curves()
