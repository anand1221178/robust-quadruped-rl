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

# Model configurations (UPDATED: October 27, 2025 - 32M retrained models)
MODELS = {
    'M1_baseline': {
        'run_id': 'ym2jcllj',  # NEW: M1_baseline_32M_RETRAINED
        'label': 'M1 (Baseline - 32M)',
        'color': '#2E86AB',
        'steps': 32_000_000  # All models now trained to 32M
    },
    'M2_sr2l': {
        'run_id': 'ze09p0vf',  # NEW: M2_sr2l_32M_RETRAINED
        'label': 'M2 (SR2L - 32M)',
        'color': '#9C27B0',
        'steps': 32_000_000  # All models now trained to 32M
    },
    'M3_dr': {
        'run_id': 'jtfwl2qf',  # KEPT: V7.7E Champion
        'label': 'M3 (DR - V7.7E)',
        'color': '#FF6F00',
        'steps': 32_000_000
    },
    'M4_combo': {
        'run_id': 'ju7lfsk2',  # KEPT: Existing combo
        'label': 'M4 (Combined)',
        'color': '#D32F2F',
        'steps': 30_000_000  # This one is actually 30M
    }
}

def download_run_data(run_id, metrics=['rollout/ep_rew_mean']):
    """Download training data from W&B"""
    print(f"Downloading run {run_id}...")

    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")

    # Get run history - fetch ALL data, then filter (keys parameter doesn't work reliably)
    history = run.history()

    print(f"  Downloaded {len(history)} data points")
    if len(history) > 0:
        print(f"  Available metrics: {[c for c in history.columns if not c.startswith('_')][:5]}...")
    else:
        print(f"  ⚠️ No data downloaded")

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

    # Create figure with single plot
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    fig.suptitle('Training Reward Curves: Convergence Analysis',
                 fontsize=16, fontweight='bold', y=0.98)

    # Plot 1: Episode Reward
    for model_key, data in model_data.items():
        history = data['history']
        config = data['config']

        if 'rollout/ep_rew_mean' in history.columns:
            # Smooth with larger rolling average for cleaner curves
            smoothed = history['rollout/ep_rew_mean'].rolling(window=100, min_periods=1).mean()

            ax1.plot(history['global_step'] / 1e6, smoothed,
                    label=config['label'],
                    color=config['color'],
                    linewidth=3.0,
                    alpha=0.95)

    ax1.set_xlabel('Training Steps (millions)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Mean Episode Reward', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=12, loc='best', framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=0)

    # Add y-axis formatting for readability
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    plt.tight_layout()

    # Save figure (updated filename to match paper: learning_curves_reconstructed.pdf)
    output_dir = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/paper/figures")
    output_file = output_dir / "learning_curves_reconstructed.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf', facecolor='white')
    print(f"\n Learning curves saved to: {output_file}")

    output_file_png = output_dir / "learning_curves_reconstructed.png"
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f" PNG preview saved to: {output_file_png}")

    # Save raw data to CSV for analysis
    for model_key, data in model_data.items():
        csv_file = output_dir / f"learning_curve_{model_key}.csv"
        data['history'].to_csv(csv_file, index=False)
        print(f" Saved {model_key} data to: {csv_file}")

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
    print(" Learning curve analysis complete!")
    print("="*80)

if __name__ == "__main__":
    create_learning_curves_figure()
