#!/usr/bin/env python3
"""
Create Figure 5: Joint-Noise Interaction (SIMPLIFIED)
Clean heatmap + worst-case comparison only
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load Experiment 7 data (UPDATED: October 27, 2025)
exp7_file = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/evaluations/evaluations/experiment_7_joint_noise_ablation/data/joint_noise_ablation_20251027_223017.json")

with open(exp7_file, 'r') as f:
    data = json.load(f)

models = ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo']
model_labels = ['M1 (Baseline)', 'M2 (SR2L)', 'M3 (DR)', 'M4 (Combined)']
colors = ['#2E86AB', '#9C27B0', '#FF6F00', '#D32F2F']

# Create figure with 2 panels (top: heatmap, bottom: worst-case)
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 1, hspace=0.35, height_ratios=[2, 1])

fig.suptitle('Joint-Noise Interaction Analysis',
             fontsize=16, fontweight='bold', y=0.98)

# ===== TOP: Heatmap for M3 (DR) - our best model =====
ax1 = fig.add_subplot(gs[0])

m3_data = data['M3_dr']
baseline = m3_data['baseline_distance']

# Organize by joint and noise level
joints = ['hip_1', 'ankle_1', 'hip_2', 'ankle_2', 'hip_3', 'ankle_3', 'hip_4', 'ankle_4']
noise_levels = sorted(list(set([r['noise_level'] for r in m3_data['ablation_results']])))

print(f"Noise levels found: {noise_levels}")

# Create matrix: rows = joints, columns = noise levels
matrix_data = np.zeros((len(joints), len(noise_levels)))

for i, joint in enumerate(joints):
    for j, noise in enumerate(noise_levels):
        # Find matching result - average if duplicates
        matches = [r for r in m3_data['ablation_results']
                  if r['failed_joint'] == joint and abs(r['noise_level'] - noise) < 0.001]
        if matches:
            avg_distance = np.mean([m['distance']['mean'] for m in matches])
            retention = (avg_distance / baseline) * 100
            matrix_data[i, j] = retention
        else:
            matrix_data[i, j] = np.nan

# Plot heatmap
im = ax1.imshow(matrix_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

# Set labels
joint_labels = [j.replace('_', ' ').title() for j in joints]
noise_labels = [f'σ={n:.2f}' if n < 1 else f'σ={n:.1f}' for n in noise_levels]

ax1.set_xticks(np.arange(len(noise_levels)))
ax1.set_yticks(np.arange(len(joints)))
ax1.set_xticklabels(noise_labels, fontsize=12)
ax1.set_yticklabels(joint_labels, fontsize=12)

# Add text annotations
for i in range(len(joints)):
    for j in range(len(noise_levels)):
        if not np.isnan(matrix_data[i, j]):
            text = ax1.text(j, i, f'{matrix_data[i, j]:.0f}%',
                           ha='center', va='center', color='black',
                           fontsize=10, fontweight='bold')

ax1.set_xlabel('Sensor Noise Level', fontsize=13, fontweight='bold')
ax1.set_ylabel('Failed Joint', fontsize=13, fontweight='bold')
ax1.set_title('(a) M3 (DR) Retention Matrix: Joint Failures × Sensor Noise',
              fontsize=13, fontweight='bold', pad=10)

# Add colorbar
cbar = plt.colorbar(im, ax=ax1, orientation='vertical', pad=0.02)
cbar.set_label('Retention (%)', fontsize=12, fontweight='bold')

# ===== BOTTOM: Worst-Case Scenario Comparison =====
ax2 = fig.add_subplot(gs[1])

# Find worst-case scenario for each model
worst_case_data = []

for model, label in zip(models, model_labels):
    model_data = data[model]
    baseline_dist = model_data['baseline_distance']

    # Find minimum retention across all scenarios
    min_retention = 100
    worst_joint = ""
    worst_noise = 0

    for result in model_data['ablation_results']:
        distance = result['distance']['mean']
        retention = (distance / baseline_dist) * 100

        if retention < min_retention:
            min_retention = retention
            worst_joint = result['failed_joint']
            worst_noise = result['noise_level']

    worst_case_data.append({
        'model': label,
        'retention': min_retention,
        'joint': worst_joint,
        'noise': worst_noise
    })

# Create bar chart
retentions = [w['retention'] for w in worst_case_data]
bars = ax2.bar(range(len(models)), retentions,
               color=colors, alpha=0.8, edgecolor='black', linewidth=2)

ax2.set_ylabel('Worst-Case Retention (%)', fontsize=13, fontweight='bold')
ax2.set_xlabel('Model', fontsize=13, fontweight='bold')
ax2.set_title('(b) Worst-Case Scenario Performance',
              fontsize=13, fontweight='bold', pad=10)
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(model_labels, fontsize=12)
ax2.set_ylim(0, max(retentions) * 1.3)
ax2.grid(True, alpha=0.3, axis='y')

# Add detailed labels
for i, (bar, w) in enumerate(zip(bars, worst_case_data)):
    joint_name = w['joint'].replace('_', ' ').title() if w['joint'] else 'None'
    noise_str = f"σ={w['noise']:.2f}" if w['noise'] < 1 else f"σ={w['noise']:.1f}"

    ax2.text(i, w['retention'] + max(retentions) * 0.05,
             f"{w['retention']:.1f}%\n{joint_name}\n{noise_str}",
             ha='center', fontsize=10, fontweight='bold')

# Add summary
fig.text(0.5, 0.02,
         'KEY FINDINGS: Ankle_4 universal worst-case joint | M3 (DR) maintains best worst-case robustness',
         ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()

# Save
output_dir = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/paper/figures")
output_file = output_dir / "figure_5_joint_noise_ablation.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf', facecolor='white')
print(f"✅ Joint-noise ablation figure saved to: {output_file}")

output_file_png = output_dir / "figure_5_joint_noise_ablation.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ PNG preview saved to: {output_file_png}")

# plt.show()  # Commented out

# Print summary
print("\n" + "="*80)
print("WORST-CASE SCENARIOS")
print("="*80)
for w in worst_case_data:
    joint_name = w['joint'].replace('_', ' ').title() if w['joint'] else 'None'
    noise_str = f"σ={w['noise']:.2f}" if w['noise'] < 1 else f"σ={w['noise']:.1f}"
    print(f"{w['model']:20s}: {w['retention']:5.1f}% - {joint_name:15s} + {noise_str}")

print("\n" + "="*80)
print("✅ Clean, simple visualization complete!")
print("✅ M3 (DR) shows best robustness under combined stress")
