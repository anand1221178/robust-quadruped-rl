#!/usr/bin/env python3
"""
Create Figure: Robustness Methods Comparison (M2, M3, M4)
Shows how different training methods perform across robustness challenges
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load data
baseline_file = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/evaluations/evaluations/experiment_1_baseline/data/baseline_results_20251020_092458.json")
noise_file = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/evaluations/evaluations/experiment_2_sensor_noise/data/sensor_noise_results_20251020_001128.json")
joint_file = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/evaluations/evaluations/experiment_3_joint_failures/data/joint_failure_results_20251013_111836.json")
combined_file = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/evaluations/evaluations/experiment_4_combined_stress/data/combined_stress_results_20251020_105735.json")

with open(baseline_file, 'r') as f:
    baseline_data = json.load(f)
with open(noise_file, 'r') as f:
    noise_data = json.load(f)
with open(joint_file, 'r') as f:
    joint_data = json.load(f)
with open(combined_file, 'r') as f:
    combined_data = json.load(f)

# Models to compare
models = ['M2_sr2l', 'M3_dr', 'M4_combo']
model_labels = ['M2 (SR2L)', 'M3 (DR)', 'M4 (SR2L+DR)']
colors = ['#9C27B0', '#FF6F00', '#D32F2F']

# Create comprehensive comparison figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

fig.suptitle('Robustness Training Methods: SR2L vs DR vs Combined\n(Which training approach handles different failure modes best?)',
             fontsize=16, fontweight='bold', y=0.98)

# ===== TOP ROW: Baseline Performance =====
ax1 = fig.add_subplot(gs[0, :])

baseline_distances = [baseline_data[model]['distance']['mean'] for model in models]
baseline_stds = [baseline_data[model]['distance']['std'] for model in models]

x_pos = np.arange(len(models))
bars = ax1.bar(x_pos, baseline_distances, yerr=baseline_stds,
               color=colors, alpha=0.8, capsize=5, edgecolor='black', linewidth=1.5)

ax1.set_ylabel('Distance Traveled (m)', fontsize=13, fontweight='bold')
ax1.set_title('(a) Baseline Performance (No Failures)\nTradeoff: Robustness Training Sacrifices Speed',
              fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(model_labels, fontsize=12)
ax1.set_ylim(0, 12)
ax1.grid(True, alpha=0.3, axis='y')

# Add performance values on bars
for i, (bar, dist) in enumerate(zip(bars, baseline_distances)):
    ax1.text(i, dist + baseline_stds[i] + 0.3, f'{dist:.2f}m',
             ha='center', fontsize=11, fontweight='bold')

# ===== MIDDLE LEFT: Sensor Noise Robustness =====
ax2 = fig.add_subplot(gs[1, 0])

# Get noise robustness at σ=0.10 (10x training noise)
noise_retention = []
for model in models:
    baseline_dist = baseline_data[model]['distance']['mean']
    # Find σ=0.10 result
    noise_results = noise_data[model]['noise_results']
    noise_010 = [r for r in noise_results if abs(r['noise_level'] - 0.10) < 0.01][0]
    noise_dist = noise_010['distance']['mean']
    retention = (noise_dist / baseline_dist) * 100
    noise_retention.append(retention)

bars2 = ax2.bar(x_pos, noise_retention, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.axhline(y=100, color='black', linestyle='--', linewidth=2, label='100% (No degradation)')
ax2.set_ylabel('Retention (%)', fontsize=12, fontweight='bold')
ax2.set_title('(b) Sensor Noise (σ=0.10)\n10× Training Noise', fontsize=13, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(model_labels, fontsize=11, rotation=15)
ax2.set_ylim(85, 115)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

for i, (bar, ret) in enumerate(zip(bars2, noise_retention)):
    color = 'darkgreen' if ret > 100 else 'darkred' if ret < 90 else 'darkorange'
    ax2.text(i, ret + 2, f'{ret:.1f}%', ha='center', fontsize=10, fontweight='bold', color=color)

# ===== MIDDLE CENTER: Joint Failure Robustness =====
ax3 = fig.add_subplot(gs[1, 1])

# Average retention across all 8 individual joints
joint_retention = []
for model in models:
    baseline_dist = baseline_data[model]['distance']['mean']

    # Get all individual joint results
    individual_results = joint_data[model]['joint_results']
    retentions = []
    for joint_result in individual_results:
        joint_dist = joint_result['distance']['mean']
        ret = (joint_dist / baseline_dist) * 100
        retentions.append(ret)

    avg_retention = np.mean(retentions)
    joint_retention.append(avg_retention)

bars3 = ax3.bar(x_pos, joint_retention, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.axhline(y=100, color='black', linestyle='--', linewidth=2, label='100% (No degradation)')
ax3.set_ylabel('Avg Retention (%)', fontsize=12, fontweight='bold')
ax3.set_title('(c) Joint Failures\nAverage Across 8 Joints', fontsize=13, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(model_labels, fontsize=11, rotation=15)
ax3.set_ylim(20, 60)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

for i, (bar, ret) in enumerate(zip(bars3, joint_retention)):
    ax3.text(i, ret + 2, f'{ret:.1f}%', ha='center', fontsize=10, fontweight='bold', color='darkgreen')

# ===== MIDDLE RIGHT: Combined Stress =====
ax4 = fig.add_subplot(gs[1, 2])

# Get combined stress results (joint failures + sensor noise)
combined_distances = []
for model in models:
    scenarios = combined_data[model]['scenario_results']
    # Average distance across all combined stress scenarios
    avg_dist = np.mean([s['distance']['mean'] for s in scenarios])
    combined_distances.append(avg_dist)

bars4 = ax4.bar(x_pos, combined_distances, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Distance (m)', fontsize=12, fontweight='bold')
ax4.set_title('(d) Combined Stress\nNoise + Joint Failures', fontsize=13, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(model_labels, fontsize=11, rotation=15)
ax4.set_ylim(0, 6)
ax4.grid(True, alpha=0.3, axis='y')

for i, (bar, dist) in enumerate(zip(bars4, combined_distances)):
    ax4.text(i, dist + 0.2, f'{dist:.2f}m', ha='center', fontsize=10, fontweight='bold')

# ===== BOTTOM: Per-Joint Heatmap =====
ax5 = fig.add_subplot(gs[2, :])

# Create heatmap data (3 models × 8 joints)
joint_names = ['Hip_1', 'Ankle_1', 'Hip_2', 'Ankle_2', 'Hip_3', 'Ankle_3', 'Hip_4', 'Ankle_4']
heatmap_data = []

for model in models:
    baseline_dist = baseline_data[model]['distance']['mean']
    individual_results = joint_data[model]['joint_results']

    row = []
    for joint_result in individual_results:
        joint_dist = joint_result['distance']['mean']
        retention = (joint_dist / baseline_dist) * 100
        row.append(retention)
    heatmap_data.append(row)

heatmap_data = np.array(heatmap_data)

im = ax5.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=60)

# Set ticks
ax5.set_xticks(np.arange(8))
ax5.set_yticks(np.arange(3))
ax5.set_xticklabels(joint_names, fontsize=11)
ax5.set_yticklabels(model_labels, fontsize=11)

# Add text annotations
for i in range(3):
    for j in range(8):
        text = ax5.text(j, i, f'{heatmap_data[i, j]:.0f}%',
                       ha='center', va='center', color='black', fontsize=10, fontweight='bold')

ax5.set_title('(e) Per-Joint Failure Retention Heatmap\n(Green = Good, Red = Poor)',
              fontsize=13, fontweight='bold', pad=15)

# Add colorbar
cbar = plt.colorbar(im, ax=ax5, orientation='horizontal', pad=0.1, aspect=30)
cbar.set_label('Retention (%)', fontsize=11, fontweight='bold')

# Add final summary text
fig.text(0.5, 0.02,
         '✅ KEY FINDINGS: M2 (SR2L) excels at sensor noise but weak on joint failures | '
         'M3 (DR) dominates joint failures | '
         'M4 (Combined) shows NO SYNERGY - worse than M3 alone',
         ha='center', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.8))

# Save
output_dir = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/paper/figures")
output_file = output_dir / "figure_robustness_methods_comparison.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf', facecolor='white')
print(f"✅ Robustness methods comparison saved to: {output_file}")

output_file_png = output_dir / "figure_robustness_methods_comparison.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ PNG preview saved to: {output_file_png}")

plt.show()

print("\n" + "="*80)
print("ROBUSTNESS METHODS COMPARISON SUMMARY")
print("="*80)
print(f"BASELINE PERFORMANCE:")
for i, (label, dist) in enumerate(zip(model_labels, baseline_distances)):
    print(f"  {label}: {dist:.2f}m")

print(f"\nSENSOR NOISE RETENTION (σ=0.10):")
for i, (label, ret) in enumerate(zip(model_labels, noise_retention)):
    print(f"  {label}: {ret:.1f}%")

print(f"\nJOINT FAILURE RETENTION (avg):")
for i, (label, ret) in enumerate(zip(model_labels, joint_retention)):
    print(f"  {label}: {ret:.1f}%")

print(f"\nCOMBINED STRESS PERFORMANCE:")
for i, (label, dist) in enumerate(zip(model_labels, combined_distances)):
    print(f"  {label}: {dist:.2f}m")

print("\n" + "="*80)
print("✅ This figure tells the complete story of robustness method specialization!")
