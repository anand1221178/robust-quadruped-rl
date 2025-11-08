#!/usr/bin/env python3
"""
Reconstruct learning curves by AVERAGING ACROSS SEEDS
Addresses Supervisor Requirement: 5-seed averaging for learning curves

This evaluates all 20 trained models (4 models × 5 seeds) and computes:
- Mean reward across seeds at each checkpoint
- Standard deviation across seeds (not across episodes!)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

# Register custom environment
import realant_sim

# Multi-seed model configurations
MODELS = {
    'M1_baseline': {
        'seeds': [42, 123, 456, 789, 999],
        'base_path': '/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/experiments',
        'pattern': 'M1_baseline_seed{seed}_*',  # W&B run ID will vary
        'label': 'M1 (Baseline)',
        'color': '#2E86AB',
        'total_steps': 32_000_000,
        'eval_interval': 1_000_000,
    },
    'M2_sr2l': {
        'seeds': [42, 123, 456, 789, 999],
        'base_path': '/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/experiments',
        'pattern': 'M2_sr2l_seed{seed}_*',
        'label': 'M2 (SR2L)',
        'color': '#9C27B0',
        'total_steps': 32_000_000,
        'eval_interval': 1_000_000,
    },
    'M3_dr': {
        'seeds': [42, 123, 456, 789, 999],
        'base_path': '/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/experiments',
        'pattern': 'M3_dr_seed{seed}_*',
        'label': 'M3 (DR)',
        'color': '#FF6F00',
        'total_steps': 32_000_000,
        'eval_interval': 1_000_000,
    },
    'M4_combo': {
        'seeds': [42, 123, 456, 789, 999],
        'base_path': '/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/experiments',
        'pattern': 'M4_combo_seed{seed}_*',
        'label': 'M4 (Combined)',
        'color': '#D32F2F',
        'total_steps': 30_000_000,
        'eval_interval': 1_000_000,
    }
}

def evaluate_checkpoint(model_path, vec_normalize_path, n_eval_episodes=10):
    """Evaluate a single checkpoint"""
    try:
        # Create environment
        env = gym.make('RealAntMujoco-v0')
        env = DummyVecEnv([lambda: env])

        # Load VecNormalize
        if Path(vec_normalize_path).exists():
            env = VecNormalize.load(vec_normalize_path, env)
            env.training = False
            env.norm_reward = False

        # Load model
        model = PPO.load(model_path, env=env)

        # Evaluate
        episode_rewards = []
        episode_distances = []

        for _ in range(n_eval_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            positions = []

            try:
                unwrapped = env.envs[0].unwrapped
                start_x = unwrapped.data.qpos[0]
            except:
                start_x = 0.0

            while not done:
                action, _states = model.predict(obs, deterministic=True)

                try:
                    positions.append(unwrapped.data.qpos[0])
                except:
                    positions.append(0.0)

                obs, reward, done, info = env.step(action)
                episode_reward += reward[0]

                if done[0]:
                    break

            final_position = positions[-1] if positions else start_x
            distance = abs(final_position - start_x)
            episode_distances.append(distance)
            episode_rewards.append(episode_reward)

        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_distance': np.mean(episode_distances)
        }

    except Exception as e:
        print(f"    Error evaluating: {e}")
        return None

def find_model_directory(base_path, pattern, seed):
    """Find model directory matching pattern with seed"""
    search_pattern = pattern.format(seed=seed)
    base = Path(base_path)

    if not base.exists():
        print(f"  ⚠️  Base path doesn't exist: {base}")
        return None

    # Search for matching directories
    matches = list(base.glob(search_pattern))

    if len(matches) == 0:
        print(f"  ⚠️  No match for pattern: {search_pattern}")
        return None
    elif len(matches) > 1:
        print(f"  ⚠️  Multiple matches for {search_pattern}, using first")

    return matches[0]

def reconstruct_model_multiseed(model_key, config):
    """Reconstruct learning curve for one model across all seeds"""
    print(f"\n{'='*80}")
    print(f"Reconstructing: {config['label']} (MULTI-SEED)")
    print('='*80)

    all_seed_results = {}  # {seed: [{steps, reward, ...}, ...]}

    # Collect data from each seed
    for seed in config['seeds']:
        print(f"\n  Processing seed {seed}...")

        # Find model directory
        model_dir = find_model_directory(config['base_path'], config['pattern'], seed)
        if not model_dir:
            continue

        checkpoint_dir = model_dir / 'checkpoints'
        vec_normalize_path = model_dir / 'vec_normalize.pkl'

        if not checkpoint_dir.exists():
            print(f"    ❌ No checkpoints at {checkpoint_dir}")
            continue

        # Find all checkpoints
        checkpoints = sorted(checkpoint_dir.glob('checkpoint_*_steps.zip'))
        print(f"    Found {len(checkpoints)} checkpoints")

        seed_results = []

        for checkpoint in tqdm(checkpoints, desc=f"  Seed {seed}"):
            # Extract step number
            step_str = checkpoint.stem.split('_')[1]
            steps = int(step_str)

            # Only evaluate at intervals
            if steps % config['eval_interval'] != 0:
                continue

            result = evaluate_checkpoint(str(checkpoint), str(vec_normalize_path))

            if result:
                result['steps'] = steps
                seed_results.append(result)

        all_seed_results[seed] = seed_results
        print(f"    ✅ Evaluated {len(seed_results)} checkpoints for seed {seed}")

    if not all_seed_results:
        print(f"  ❌ No data collected for {model_key}!")
        return None

    # Aggregate across seeds
    print(f"\n  Aggregating across {len(all_seed_results)} seeds...")
    aggregated = aggregate_across_seeds(all_seed_results, config['eval_interval'], config['total_steps'])

    return aggregated

def aggregate_across_seeds(all_seed_results, eval_interval, total_steps):
    """Aggregate results across seeds"""
    # Find all checkpoint steps
    all_steps = set()
    for seed_results in all_seed_results.values():
        all_steps.update([r['steps'] for r in seed_results])

    all_steps = sorted(all_steps)

    aggregated = []

    for steps in all_steps:
        # Collect results at this step from all seeds
        rewards_at_step = []
        distances_at_step = []

        for seed, seed_results in all_seed_results.items():
            # Find result at this step
            matching = [r for r in seed_results if r['steps'] == steps]
            if matching:
                rewards_at_step.append(matching[0]['mean_reward'])
                distances_at_step.append(matching[0]['mean_distance'])

        if rewards_at_step:
            aggregated.append({
                'steps': steps,
                'mean_reward': np.mean(rewards_at_step),
                'std_reward_across_seeds': np.std(rewards_at_step),  # STD ACROSS SEEDS!
                'mean_distance': np.mean(distances_at_step),
                'std_distance_across_seeds': np.std(distances_at_step),
                'num_seeds': len(rewards_at_step)
            })

    return aggregated

def create_multiseed_learning_curves():
    """Create learning curves from multi-seed data"""
    print("\n" + "="*80)
    print("RECONSTRUCTING MULTI-SEED LEARNING CURVES")
    print("="*80)
    print("\nAveraging across 5 seeds per model (20 total training runs)")
    print("="*80)

    # Reconstruct curves for all models
    all_results = {}

    for model_key, config in MODELS.items():
        results = reconstruct_model_multiseed(model_key, config)
        if results:
            all_results[model_key] = results

    if not all_results:
        print("\n❌ No data reconstructed!")
        return

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Training Dynamics: Multi-Seed Average (n=5 seeds)',
                 fontsize=16, fontweight='bold', y=0.995)

    # Plot 1: Mean Reward (averaged across seeds)
    for model_key, results in all_results.items():
        config = MODELS[model_key]

        steps = [r['steps'] for r in results]
        rewards = [r['mean_reward'] for r in results]
        stds = [r['std_reward_across_seeds'] for r in results]

        ax1.plot(np.array(steps) / 1e6, rewards,
                label=config['label'],
                color=config['color'],
                linewidth=2.5,
                marker='o',
                markersize=4,
                alpha=0.9)

        # Error bands show VARIANCE ACROSS SEEDS (not across episodes!)
        ax1.fill_between(np.array(steps) / 1e6,
                         np.array(rewards) - np.array(stds),
                         np.array(rewards) + np.array(stds),
                         color=config['color'],
                         alpha=0.2,
                         label=f'{config["label"]} (±1 SD across seeds)')

    ax1.set_xlabel('Training Steps (millions)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Episode Reward', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Reward Progression: 5-Seed Average',
                  fontsize=13, fontweight='bold', pad=10)
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)

    # Plot 2: Distance
    for model_key, results in all_results.items():
        config = MODELS[model_key]

        steps = [r['steps'] for r in results]
        distances = [r['mean_distance'] for r in results]
        stds = [r['std_distance_across_seeds'] for r in results]

        ax2.plot(np.array(steps) / 1e6, distances,
                label=config['label'],
                color=config['color'],
                linewidth=2.5,
                marker='o',
                markersize=4,
                alpha=0.9)

        ax2.fill_between(np.array(steps) / 1e6,
                         np.array(distances) - np.array(stds),
                         np.array(distances) + np.array(stds),
                         color=config['color'],
                         alpha=0.2)

    ax2.set_xlabel('Training Steps (millions)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Distance (m)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Locomotion Distance: 5-Seed Average',
                  fontsize=13, fontweight='bold', pad=10)
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)

    plt.tight_layout()

    # Save
    output_dir = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/paper/figures")
    output_file = output_dir / "learning_curves_multiseed.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
    print(f"\n✅ Multi-seed curves saved to: {output_file}")

    output_file_png = output_dir / "learning_curves_multiseed.png"
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    print(f"✅ PNG preview: {output_file_png}")

    # Save data
    for model_key, results in all_results.items():
        json_file = output_dir / f"learning_curve_{model_key}_multiseed.json"
        results_serializable = [
            {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
             for k, v in r.items()}
            for r in results
        ]
        with open(json_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print(f"✅ Saved {model_key} multiseed data: {json_file}")

    plt.show()

    print("\n" + "="*80)
    print("✅ Multi-seed learning curve reconstruction complete!")
    print("="*80)

if __name__ == "__main__":
    create_multiseed_learning_curves()
