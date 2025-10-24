#!/usr/bin/env python3
"""
Export learning curves from W&B for publication figure
Addresses Peer Review Issue #23: Training dynamics analysis
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# W&B configuration
ENTITY = "anandpatel1221178-university-of-the-witswatersrand"
PROJECT = "robust-quadruped-rl"

# Model configurations
MODELS = {
    'M1_baseline': {
        'run_id': 'ueqbjf2x',
        'label': 'M1 (Baseline)',
        'color': '#2E86AB',
        'steps': 10_000_000
    },
    'M2_sr2l': {
        'run_id': 'm7gtjtpa',
        'label': 'M2 (SR2L)',
        'color': '#9C27B0',
        'steps': 20_000_000
    },
    'M3_dr': {
        'run_id': 'jtfwl2qf',
        'label': 'M3 (DR)',
        'color': '#FF6F00',
        'steps': 32_000_000
    },
    'M4_combo': {
        'run_id': 'ju7lfsk2',
        'label': 'M4 (Combined)',
        'color': '#D32F2F',
        'steps': 30_000_000
    }
}

def download_run_data(run_id, metrics=['rollout/ep_rew_mean']):
    """Download training data from W&B"""
    print(f"Downloading run {run_id}...")

    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")

    # Get run history
    history = run.history(keys=metrics + ['_step'])

    print(f"  Downloaded {len(history)} data points")
    print(f"  Columns: {history.columns.tolist()}")

    return history, run

def create_learning_curves_figure():
    """Create publication-quality learning curves figure"""

    # Initialize W&B API
    print("\n" + "="*80)
    print("EXPORTING W&B LEARNING CURVES")
    print("="*80)

    # Download data for all models
    model_data = {}

    for model_key, config in MODELS.items():
        try:
            history, run = download_run_data(
                config['run_id'],
                metrics=['rollout/ep_rew_mean', 'rollout/ep_len_mean']
            )

            model_data[model_key] = {
                'history': history,
                'config': config,
                'run': run
            }

        except Exception as e:
            print(f"  ⚠️  Error downloading {model_key}: {e}")
            continue

    if not model_data:
        print("\n❌ No data downloaded! Check your W&B credentials.")
        return

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Training Dynamics: Convergence Analysis Across Methods',
                 fontsize=16, fontweight='bold', y=0.995)

    # Plot 1: Episode Reward
    for model_key, data in model_data.items():
        history = data['history']
        config = data['config']

        if 'rollout/ep_rew_mean' in history.columns:
            # Smooth with rolling average
            smoothed = history['rollout/ep_rew_mean'].rolling(window=50, min_periods=1).mean()

            ax1.plot(history['_step'] / 1e6, smoothed,
                    label=config['label'],
                    color=config['color'],
                    linewidth=2.5,
                    alpha=0.9)

            # Add light raw data in background
            ax1.plot(history['_step'] / 1e6, history['rollout/ep_rew_mean'],
                    color=config['color'],
                    linewidth=0.5,
                    alpha=0.15)

    ax1.set_xlabel('Training Steps (millions)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Episode Reward', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Reward Progression: All Models Show Convergence',
                  fontsize=13, fontweight='bold', pad=10)
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)

    # Add convergence annotations
    ax1.axvline(x=10, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.text(10, ax1.get_ylim()[1]*0.95, 'M1 trained\n(10M steps)',
            ha='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.3',
            facecolor='white', alpha=0.7))

    # Plot 2: Episode Length
    for model_key, data in model_data.items():
        history = data['history']
        config = data['config']

        if 'rollout/ep_len_mean' in history.columns:
            # Smooth with rolling average
            smoothed = history['rollout/ep_len_mean'].rolling(window=50, min_periods=1).mean()

            ax2.plot(history['_step'] / 1e6, smoothed,
                    label=config['label'],
                    color=config['color'],
                    linewidth=2.5,
                    alpha=0.9)

    ax2.set_xlabel('Training Steps (millions)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Episode Length (steps)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Episode Length: Stability Over Training',
                  fontsize=13, fontweight='bold', pad=10)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)

    plt.tight_layout()

    # Save figure
    output_dir = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/paper/figures")
    output_file = output_dir / "learning_curves.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf', facecolor='white')
    print(f"\n✅ Learning curves saved to: {output_file}")

    output_file_png = output_dir / "learning_curves.png"
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ PNG preview saved to: {output_file_png}")

    # Save raw data to CSV for analysis
    for model_key, data in model_data.items():
        csv_file = output_dir / f"learning_curve_{model_key}.csv"
        data['history'].to_csv(csv_file, index=False)
        print(f"✅ Saved {model_key} data to: {csv_file}")

    plt.show()

    # Print convergence analysis
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS")
    print("="*80)

    for model_key, data in model_data.items():
        history = data['history']
        config = data['config']

        if 'rollout/ep_rew_mean' in history.columns:
            # Compute final performance (last 10% of training)
            final_10pct = int(len(history) * 0.9)
            final_reward = history['rollout/ep_rew_mean'].iloc[final_10pct:].mean()
            final_std = history['rollout/ep_rew_mean'].iloc[final_10pct:].std()

            print(f"\n{config['label']}:")
            print(f"  Total steps: {config['steps']:,}")
            print(f"  Final reward (last 10%): {final_reward:.2f} ± {final_std:.2f}")
            print(f"  Stability (CV): {(final_std/final_reward)*100:.1f}%")

    print("\n" + "="*80)
    print("✅ Learning curve analysis complete!")
    print("="*80)

if __name__ == "__main__":
    create_learning_curves_figure()
