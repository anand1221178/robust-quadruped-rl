#!/usr/bin/env python3
"""
Reconstruct learning curves by evaluating saved checkpoints
Addresses Peer Review Issue #23: Training dynamics without W&B logs

This is MORE rigorous than W&B logs because we use the exact evaluation protocol
"""

import sys
from pathlib import Path

# Add src to path to import custom environment
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

# Model configurations
MODELS = {
    'M1_baseline': {
        'path': '/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/done/ppo_baseline_ueqbjf2x',
        'label': 'M1 (Baseline)',
        'color': '#2E86AB',
        'total_steps': 10_000_000,
        'eval_interval': 500_000,  # Evaluate every 500K steps
        'checkpoint_pattern': 'model_{steps}_steps.zip'  # M1 uses model_* pattern
    },
    'M2_sr2l': {
        'path': '/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/done/ppo_sr2l_forward_m7gtjtpa',
        'label': 'M2 (SR2L)',
        'color': '#9C27B0',
        'total_steps': 20_000_000,
        'eval_interval': 1_000_000,  # Evaluate every 1M steps
        'checkpoint_pattern': 'checkpoint_{steps}_steps.zip'  # M2 uses checkpoint_* pattern
    },
    'M3_dr': {
        'path': '/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/done/v7_7e_ultra_speed_jtfwl2qf',
        'label': 'M3 (DR)',
        'color': '#FF6F00',
        'total_steps': 32_000_000,
        'eval_interval': 1_000_000,  # Evaluate every 1M steps
        'checkpoint_pattern': 'checkpoint_{steps}_steps.zip'  # M3 uses checkpoint_* pattern
    },
    'M4_combo': {
        'path': '/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/done/ultimate_robustness_combo_ju7lfsk2',
        'label': 'M4 (Combined)',
        'color': '#D32F2F',
        'total_steps': 30_000_000,
        'eval_interval': 1_000_000,  # Evaluate every 1M steps
        'checkpoint_pattern': 'checkpoint_{steps}_steps.zip'  # M4 uses checkpoint_* pattern
    }
}

def evaluate_checkpoint(model_path, vec_normalize_path, n_eval_episodes=10):
    """Evaluate a single checkpoint"""
    try:
        # Create environment
        env = gym.make('RealAntMujoco-v0')
        env = DummyVecEnv([lambda: env])

        # Load VecNormalize if exists
        if Path(vec_normalize_path).exists():
            env = VecNormalize.load(vec_normalize_path, env)
            env.training = False
            env.norm_reward = False

        # Load model
        model = PPO.load(model_path, env=env)

        # Evaluate
        episode_rewards = []
        episode_lengths = []
        episode_distances = []

        for _ in range(n_eval_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            positions = []

            # Get initial position
            try:
                unwrapped = env.envs[0].unwrapped
                start_x = unwrapped.data.qpos[0]
            except:
                start_x = 0.0

            while not done:
                action, _states = model.predict(obs, deterministic=True)

                # Track position BEFORE step (to avoid post-reset position)
                try:
                    current_x = unwrapped.data.qpos[0]
                    positions.append(current_x)
                except:
                    positions.append(0.0)

                obs, reward, done, info = env.step(action)

                episode_reward += reward[0]
                episode_length += 1

                if done[0]:
                    break

            # Calculate distance using position list
            final_position = positions[-1] if positions else start_x
            distance = abs(final_position - start_x)
            episode_distances.append(distance)

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'mean_distance': np.mean(episode_distances) if episode_distances else 0
        }

    except Exception as e:
        print(f"    Error evaluating: {e}")
        return None

def reconstruct_model_curve(model_key, config):
    """Reconstruct learning curve for one model"""
    print(f"\n{'='*80}")
    print(f"Reconstructing: {config['label']}")
    print('='*80)

    checkpoint_dir = Path(config['path']) / 'checkpoints'
    vec_normalize_path = Path(config['path']) / 'vec_normalize.pkl'

    if not checkpoint_dir.exists():
        print(f"  ❌ No checkpoints found at {checkpoint_dir}")
        return None

    # Determine checkpoint pattern (model_* or checkpoint_*)
    checkpoint_prefix = config.get('checkpoint_pattern', 'model_{steps}_steps.zip').split('_')[0]
    checkpoint_pattern = f'{checkpoint_prefix}_*_steps.zip'

    # Find all checkpoints
    checkpoints = sorted(checkpoint_dir.glob(checkpoint_pattern))
    print(f"  Found {len(checkpoints)} checkpoints")

    if len(checkpoints) == 0:
        print(f"  ❌ No checkpoints matching pattern {checkpoint_pattern}")
        return None

    # Evaluate at intervals
    results = []
    eval_interval = config['eval_interval']

    for checkpoint in tqdm(checkpoints, desc=f"Evaluating {model_key}"):
        # Extract step number from filename
        # Handle both model_1000000_steps.zip and checkpoint_1000000_steps.zip
        step_str = checkpoint.stem.split('_')[1]
        steps = int(step_str)

        # Only evaluate at intervals
        if steps % eval_interval != 0:
            continue

        print(f"\n  Evaluating checkpoint at {steps:,} steps...")
        result = evaluate_checkpoint(str(checkpoint), str(vec_normalize_path))

        if result:
            result['steps'] = steps
            results.append(result)
            print(f"    Reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
            print(f"    Distance: {result['mean_distance']:.2f}m")

    return results

def create_learning_curves():
    """Create learning curves from reconstructed data"""
    print("\n" + "="*80)
    print("RECONSTRUCTING LEARNING CURVES FROM CHECKPOINTS")
    print("="*80)
    print("\nThis evaluates saved checkpoints to create learning curves.")
    print("More rigorous than W&B logs - uses exact evaluation protocol!")

    # Reconstruct curves for all models
    all_results = {}

    for model_key, config in MODELS.items():
        results = reconstruct_model_curve(model_key, config)
        if results:
            all_results[model_key] = results

    if not all_results:
        print("\n❌ No data reconstructed!")
        return

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Training Dynamics: Convergence Analysis from Checkpoint Evaluation',
                 fontsize=16, fontweight='bold', y=0.995)

    # Plot 1: Mean Reward
    for model_key, results in all_results.items():
        config = MODELS[model_key]

        # Sort results by step number (not alphabetically!)
        results_sorted = sorted(results, key=lambda x: x['steps'])

        steps = [r['steps'] for r in results_sorted]
        rewards = [r['mean_reward'] for r in results_sorted]
        stds = [r['std_reward'] for r in results_sorted]

        ax1.plot(np.array(steps) / 1e6, rewards,
                label=config['label'],
                color=config['color'],
                linewidth=2.5,
                marker='o',
                markersize=4,
                alpha=0.9)

        # Add error bands
        ax1.fill_between(np.array(steps) / 1e6,
                         np.array(rewards) - np.array(stds),
                         np.array(rewards) + np.array(stds),
                         color=config['color'],
                         alpha=0.15)

    ax1.set_xlabel('Training Steps (millions)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Episode Reward', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Reward Progression: Convergence via Checkpoint Evaluation',
                  fontsize=13, fontweight='bold', pad=10)
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)

    # Plot 2: Mean Distance
    for model_key, results in all_results.items():
        config = MODELS[model_key]

        # Sort results by step number (not alphabetically!)
        results_sorted = sorted(results, key=lambda x: x['steps'])

        steps = [r['steps'] for r in results_sorted]
        distances = [r['mean_distance'] for r in results_sorted]

        ax2.plot(np.array(steps) / 1e6, distances,
                label=config['label'],
                color=config['color'],
                linewidth=2.5,
                marker='o',
                markersize=4,
                alpha=0.9)

    ax2.set_xlabel('Training Steps (millions)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Distance (m)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Locomotion Distance: Performance Over Training',
                  fontsize=13, fontweight='bold', pad=10)
    ax2.legend(fontsize=11, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)

    plt.tight_layout()

    # Save figure
    output_dir = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/paper/figures")
    output_file = output_dir / "learning_curves_reconstructed.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf', facecolor='white')
    print(f"\n✅ Learning curves saved to: {output_file}")

    output_file_png = output_dir / "learning_curves_reconstructed.png"
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ PNG preview saved to: {output_file_png}")

    # Save data (convert numpy types to Python types for JSON serialization)
    for model_key, results in all_results.items():
        json_file = output_dir / f"learning_curve_{model_key}_reconstructed.json"
        # Convert numpy types to native Python types
        results_serializable = []
        for r in results:
            r_clean = {
                'steps': int(r['steps']),
                'mean_reward': float(r['mean_reward']),
                'std_reward': float(r['std_reward']),
                'mean_length': float(r['mean_length']),
                'mean_distance': float(r['mean_distance'])
            }
            results_serializable.append(r_clean)

        with open(json_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print(f"✅ Saved {model_key} data to: {json_file}")

    plt.show()

    # Print convergence analysis
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS")
    print("="*80)

    for model_key, results in all_results.items():
        config = MODELS[model_key]
        if len(results) >= 3:
            # Compare last 3 checkpoints for stability
            final_rewards = [r['mean_reward'] for r in results[-3:]]
            final_mean = np.mean(final_rewards)
            final_std = np.std(final_rewards)
            cv = (final_std / final_mean) * 100 if final_mean != 0 else 0

            print(f"\n{config['label']}:")
            print(f"  Checkpoints evaluated: {len(results)}")
            print(f"  Final performance (last 3 checkpoints):")
            print(f"    Reward: {final_mean:.2f} ± {final_std:.2f}")
            print(f"    Stability (CV): {cv:.1f}%")
            print(f"    {'✅ Converged' if cv < 5 else '⚠️ Still improving'}")

    print("\n" + "="*80)
    print("✅ Learning curve reconstruction complete!")
    print("="*80)

if __name__ == "__main__":
    create_learning_curves()
