#!/usr/bin/env python3
"""
Analyze M4 Collapse: Extract loss component magnitudes during 14-20M step collapse
Addresses supervisor request for quantitative gradient/loss evidence
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# W&B configuration
ENTITY = "anandpatel1221178-university-of-the-witswatersrand"
PROJECT = "robust-quadruped-rl"

# M4 (Combined) run - the one that collapsed
M4_RUN_ID = 'ju7lfsk2'  # M4 combo model

def download_m4_loss_data():
    """Download loss components from M4 training"""
    print("\n" + "="*80)
    print("DOWNLOADING M4 LOSS COMPONENT DATA")
    print("="*80)

    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{M4_RUN_ID}")

    # Get full history
    print(f"Downloading M4 (run {M4_RUN_ID})...")
    history = run.history()

    print(f"  Downloaded {len(history)} data points")
    print(f"  Available columns:")
    for col in sorted([c for c in history.columns if not c.startswith('_')])[:20]:
        print(f"    - {col}")

    return history, run

def create_loss_breakdown_figure():
    """Create figure showing loss component magnitudes during collapse"""

    history, run = download_m4_loss_data()

    if history.empty:
        print("\n❌ No data downloaded!")
        return

    # Find available loss-related columns
    loss_columns = [c for c in history.columns if 'loss' in c.lower() and not c.startswith('_')]
    reward_columns = [c for c in history.columns if 'rew' in c.lower() and not c.startswith('_')]

    print(f"\n📊 Loss-related columns found: {loss_columns}")
    print(f"📊 Reward-related columns found: {reward_columns}")

    # Convert steps to millions
    steps_M = history['global_step'] / 1e6 if 'global_step' in history.columns else history['_step'] / 1e6

    # Focus on collapse period: 14-20M steps
    collapse_mask = (steps_M >= 14) & (steps_M <= 20)
    full_range_mask = steps_M <= 30  # Full training

    # Create figure with 2 panels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('M4 Training Collapse Analysis: Loss Component Magnitudes (14-20M Steps)',
                 fontsize=16, fontweight='bold', y=0.98)

    # Panel 1: Training reward showing collapse
    if 'rollout/ep_rew_mean' in history.columns:
        reward_data = history['rollout/ep_rew_mean']
        smoothed_reward = reward_data.rolling(window=50, min_periods=1).mean()

        ax1.plot(steps_M[full_range_mask], smoothed_reward[full_range_mask],
                 linewidth=3, color='#D32F2F', label='M4 Episode Reward')

        # Highlight collapse region
        ax1.axvspan(14, 20, alpha=0.3, color='red', label='Collapse Period')
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

        ax1.set_xlabel('Training Steps (millions)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Mean Episode Reward', fontsize=13, fontweight='bold')
        ax1.set_title('(a) Training Reward Collapse', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 30)

        # Add annotations
        collapse_reward = smoothed_reward[collapse_mask].min()
        ax1.annotate(f'Collapse:\n{collapse_reward:.0f}',
                     xy=(17, collapse_reward), xytext=(17, collapse_reward - 50000),
                     fontsize=11, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                     arrowprops=dict(arrowstyle='->', lw=2, color='red'))

    # Panel 2: Loss component breakdown (if available)
    ax2_has_data = False

    for loss_col in loss_columns[:3]:  # Plot up to 3 loss components
        if loss_col in history.columns:
            loss_data = history[loss_col].abs()  # Take absolute value
            smoothed_loss = loss_data.rolling(window=50, min_periods=1).mean()

            # Plot full range
            label = loss_col.replace('train/', '').replace('_', ' ').title()
            ax2.plot(steps_M[full_range_mask], smoothed_loss[full_range_mask],
                     linewidth=2.5, label=label, alpha=0.9)
            ax2_has_data = True

    if ax2_has_data:
        # Highlight collapse region
        ax2.axvspan(14, 20, alpha=0.3, color='red')

        ax2.set_xlabel('Training Steps (millions)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Loss Magnitude (absolute value)', fontsize=13, fontweight='bold')
        ax2.set_title('(b) Loss Component Magnitudes Over Training', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11, loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 30)
        ax2.set_yscale('log')  # Log scale to see all components
    else:
        ax2.text(0.5, 0.5, 'No loss component data available\n(PPO may not log individual loss terms)',
                 ha='center', va='center', fontsize=14, transform=ax2.transAxes)
        ax2.set_title('(b) Loss Components (Not Available)', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save
    output_dir = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/paper/figures")
    output_file = output_dir / "figure_m4_collapse_analysis.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf', facecolor='white')
    print(f"\n M4 collapse analysis saved to: {output_file}")

    output_file_png = output_dir / "figure_m4_collapse_analysis.png"
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f" PNG preview saved to: {output_file_png}")

    # Save collapse period data to CSV
    collapse_data = history[collapse_mask][['global_step'] + loss_columns + reward_columns].copy()
    csv_file = output_dir / "m4_collapse_data_14_20M.csv"
    collapse_data.to_csv(csv_file, index=False)
    print(f" Collapse period data saved to: {csv_file}")

    # Print statistics
    print("\n" + "="*80)
    print("COLLAPSE PERIOD STATISTICS (14-20M steps)")
    print("="*80)

    if 'rollout/ep_rew_mean' in history.columns:
        collapse_rewards = history[collapse_mask]['rollout/ep_rew_mean']
        print(f"\nEpisode Reward during collapse:")
        print(f"  Min: {collapse_rewards.min():.2f}")
        print(f"  Max: {collapse_rewards.max():.2f}")
        print(f"  Mean: {collapse_rewards.mean():.2f}")
        print(f"  Std: {collapse_rewards.std():.2f}")

    for loss_col in loss_columns[:3]:
        if loss_col in history.columns:
            collapse_loss = history[collapse_mask][loss_col]
            print(f"\n{loss_col} during collapse:")
            print(f"  Min: {collapse_loss.min():.4f}")
            print(f"  Max: {collapse_loss.max():.4f}")
            print(f"  Mean: {collapse_loss.mean():.4f}")

    print("\n" + "="*80)
    print(" M4 collapse analysis complete!")

    plt.show()

if __name__ == "__main__":
    create_loss_breakdown_figure()
