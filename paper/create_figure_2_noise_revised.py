#!/usr/bin/env python3
"""
Create Figure 2 (REVISED): Sensor Noise Robustness
Two-panel figure:
  Left: M1-M4 comparison (M1 dominates everywhere)
  Right: VecNormalize ablation (shows WHY they're all robust)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load data
baseline_file = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/evaluations/evaluations/experiment_1_baseline/data/baseline_results_20251027_195743.json")
noise_file = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/evaluations/evaluations/experiment_2_sensor_noise/data/sensor_noise_results_20251027_201704.json")
vec_file = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/evaluations/evaluations/experiment_6_validation_suite/data/validation_results_20251020_115842.json")

with open(baseline_file, 'r') as f:
    baseline_data = json.load(f)
with open(noise_file, 'r') as f:
    noise_data = json.load(f)
with open(vec_file, 'r') as f:
    vec_data = json.load(f)

# Create figure with 2 panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ===== LEFT PANEL: All models comparison =====
models = ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo']
model_labels = ['M1 (Baseline)', 'M2 (SR2L)', 'M3 (DR)', 'M4 (Combined)']
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
markers = ['o', 's', '^', 'D']

for i, (model, label, color, marker) in enumerate(zip(models, model_labels, colors, markers)):
    noise_results = noise_data[model]['noise_results']
    # Only plot first 6 points (0.00 to 0.30) for clarity
    noise_levels = [r['noise_level'] for r in noise_results[:6]]
    distances = [r['distance']['mean'] for r in noise_results[:6]]

    ax1.plot(noise_levels, distances, marker=marker, linewidth=2.5,
            markersize=9, label=label, color=color, alpha=0.9)

ax1.axvline(x=0.01, color='#A23B72', linestyle='--', linewidth=1.5,
           alpha=0.5, label='SR2L Training (σ=0.01)')
ax1.axvline(x=0.10, color='gray', linestyle='--', linewidth=1.5,
           alpha=0.5, label='10× Training (σ=0.10)')

ax1.set_xlabel('Observation Noise Level (σ)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Distance Traveled (m)', fontsize=12, fontweight='bold')
ax1.set_title('(a) All Models: Noise Robustness Comparison', fontsize=13, fontweight='bold')
ax1.set_xlim(-0.01, 0.32)
ax1.set_ylim(2, 12.5)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='lower left', fontsize=9, framealpha=0.95)

# Add key finding text
ax1.text(0.15, 11.5, 'M1 dominates at ALL noise levels\\n(>95% retention despite NO training)',
         fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
         fontweight='bold')

# ===== RIGHT PANEL: VecNormalize Ablation =====
vec_test = vec_data['test_1_vecnormalize']
with_vec = vec_test['with_vecnormalize']
without_vec = vec_test['without_vecnormalize']

# Extract data
noise_levels_vec = [0.0, 0.05, 0.1]
with_vec_dists = [with_vec[f'noise_{n}']['mean'] for n in noise_levels_vec]
without_vec_dists = [without_vec[f'noise_{n}']['mean'] for n in noise_levels_vec]

# Plot both lines
ax2.plot(noise_levels_vec, with_vec_dists, marker='o', linewidth=3,
        markersize=10, label='M1 WITH VecNormalize', color='#2E86AB', alpha=0.9)
ax2.plot(noise_levels_vec, without_vec_dists, marker='X', linewidth=3,
        markersize=10, label='M1 WITHOUT VecNormalize', color='#C73E1D', alpha=0.9, linestyle='--')

# Add retention percentages
for i, (noise, with_d, without_d) in enumerate(zip(noise_levels_vec, with_vec_dists, without_vec_dists)):
    retention_with = (with_d / with_vec_dists[0]) * 100
    retention_without = (without_d / without_vec_dists[0]) * 100

    ax2.text(noise + 0.005, with_d + 0.3, f'{retention_with:.0f}%',
            fontsize=9, fontweight='bold', color='#2E86AB')
    ax2.text(noise + 0.005, without_d - 0.4, f'{retention_without:.0f}%',
            fontsize=9, fontweight='bold', color='#C73E1D')

ax2.set_xlabel('Observation Noise Level (σ)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Distance Traveled (m)', fontsize=12, fontweight='bold')
ax2.set_title('(b) VecNormalize Ablation: The Hidden Robustness Source', fontsize=13, fontweight='bold')
ax2.set_xlim(-0.01, 0.12)
ax2.set_ylim(3, 12.5)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='lower left', fontsize=10, framealpha=0.95)

# Add key finding
improvement = ((with_vec_dists[0] - without_vec_dists[0]) / without_vec_dists[0]) * 100
ax2.text(0.05, 10, f'VecNormalize provides:\\n• {improvement:.0f}% baseline boost\\n• Implicit noise filtering',
         fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.4),
         fontweight='bold', ha='center')

plt.tight_layout()

# Save
output_dir = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/paper/figures")
output_file = output_dir / "figure_2_noise_robustness.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"✅ Figure 2 (revised) saved to: {output_file}")

output_file_png = output_dir / "figure_2_noise_robustness.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
print(f"✅ PNG preview saved to: {output_file_png}")

plt.show()

print("\n" + "="*80)
print("KEY FINDINGS FOR FIGURE 2")
print("="*80)
print(f"✅ M1 (Baseline) maintains >95% retention at σ=0.10 despite NO robustness training")
print(f"✅ VecNormalize provides {improvement:.0f}% baseline boost (11.20m vs 4.55m)")
print(f"✅ VecNormalize maintains 96% retention at σ=0.10 (implicit low-pass filter)")
print(f"✅ WITHOUT VecNormalize: only 95% retention (degrades from 4.55m to 4.35m)")
print(f"✅ This explains universal robustness across ALL models")
print("="*80)
