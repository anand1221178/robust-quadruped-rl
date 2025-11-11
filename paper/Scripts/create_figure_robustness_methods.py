#!/usr/bin/env python3
"""
Create Figure: Robustness Methods Comparison (M2, M3, M4)
Shows how different training methods perform across robustness challenges
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', message='This figure includes Axes')

# Load data
baseline_file = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/evaluations/evaluations/experiment_1_baseline/data/baseline_results_20251027_195743.json")
noise_file = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/evaluations/evaluations/experiment_2_sensor_noise/data/sensor_noise_results_20251027_201704.json")
joint_file = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/evaluations/evaluations/experiment_3_joint_failures/data/joint_failure_results_20251027_205918.json")
combined_file = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/evaluations/evaluations/experiment_4_combined_stress/data/combined_stress_results_20251027_210901.json")

with open(baseline_file, 'r') as f:
    baseline_data = json.load(f)
with open(noise_file, 'r') as f:
    noise_data = json.load(f)
with open(joint_file, 'r') as f:
    joint_data = json.load(f)
with open(combined_file, 'r') as f:
    combined_data = json.load(f)

# Models to compare - NOW INCLUDING M1!
models = ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo']
model_labels = ['M1 (Baseline)', 'M2 (SR2L)', 'M3 (DR)', 'M4 (SR2L+DR)']
colors = ['#2E86AB', '#9C27B0', '#FF6F00', '#D32F2F']

# Create SIMPLIFIED comparison figure (4 panels instead of 5)
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

fig.suptitle('Robustness Method Comparison',
             fontsize=16, fontweight='bold', y=0.96)

# ===== TOP LEFT: Baseline Performance =====
ax1 = fig.add_subplot(gs[0, 0])

baseline_distances = [baseline_data[model]['distance']['mean'] for model in models]
baseline_stds = [baseline_data[model]['distance']['std'] for model in models]

x_pos = np.arange(len(models))
bars = ax1.bar(x_pos, baseline_distances,
               color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

ax1.set_ylabel('Distance (m)', fontsize=12, fontweight='bold')
ax1.set_title('(a) Baseline Performance\n(No Failures)',
              fontsize=13, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(model_labels, fontsize=10, rotation=15)
ax1.set_ylim(0, 12)
ax1.grid(True, alpha=0.3, axis='y')

# Add performance values ON bars
for i, (bar, dist) in enumerate(zip(bars, baseline_distances)):
    ax1.text(i, dist/2, f'{dist:.1f}m',
             ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# ===== TOP RIGHT: Sensor Noise Robustness (ABSOLUTE DISTANCE) =====
ax2 = fig.add_subplot(gs[0, 1])

# Get noise robustness at σ=0.10 (10x training noise) - SHOW ABSOLUTE DISTANCE
noise_distances = []
noise_stds = []
for model in models:
    # Find σ=0.10 result
    noise_results = noise_data[model]['noise_results']
    noise_010 = [r for r in noise_results if abs(r['noise_level'] - 0.10) < 0.01][0]
    noise_dist = noise_010['distance']['mean']
    noise_std = noise_010['distance']['std']
    noise_distances.append(noise_dist)
    noise_stds.append(noise_std)

bars2 = ax2.bar(x_pos, noise_distances,
                color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Distance (m)', fontsize=12, fontweight='bold')
ax2.set_title('(b) With Sensor Noise\n(σ=0.10, 10× training level)', fontsize=13, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(model_labels, fontsize=10, rotation=15)
ax2.set_ylim(0, 12)
ax2.grid(True, alpha=0.3, axis='y')

for i, (bar, dist) in enumerate(zip(bars2, noise_distances)):
    ax2.text(i, dist/2, f'{dist:.1f}m', ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# ===== BOTTOM LEFT: Joint Failure Robustness (ABSOLUTE DISTANCE) =====
ax3 = fig.add_subplot(gs[1, 0])

# Average ABSOLUTE DISTANCE across all 8 individual joints
joint_distances = []
joint_stds = []
for model in models:
    # Get all individual joint results
    individual_results = joint_data[model]['joint_results']
    distances = [joint_result['distance']['mean'] for joint_result in individual_results]

    avg_distance = np.mean(distances)
    std_distance = np.std(distances)
    joint_distances.append(avg_distance)
    joint_stds.append(std_distance)

bars3 = ax3.bar(x_pos, joint_distances,
                color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Distance (m)', fontsize=12, fontweight='bold')
ax3.set_title('(c) With Joint Failures\n(Average across 8 joints)', fontsize=13, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(model_labels, fontsize=10, rotation=15)
ax3.set_ylim(0, 5)
ax3.grid(True, alpha=0.3, axis='y')

for i, (bar, dist) in enumerate(zip(bars3, joint_distances)):
    ax3.text(i, dist/2, f'{dist:.1f}m', ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# ===== BOTTOM RIGHT: Combined Stress =====
ax4 = fig.add_subplot(gs[1, 1])

# Get combined stress results (joint failures + sensor noise)
combined_distances = []
combined_stds = []
for model in models:
    scenarios = combined_data[model]['scenario_results']
    # Average distance across all combined stress scenarios
    distances = [s['distance']['mean'] for s in scenarios]
    avg_dist = np.mean(distances)
    std_dist = np.std(distances)
    combined_distances.append(avg_dist)
    combined_stds.append(std_dist)

bars4 = ax4.bar(x_pos, combined_distances,
                color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Distance (m)', fontsize=12, fontweight='bold')
ax4.set_title('(d) Combined Stress\n(Noise + Joint Failures)', fontsize=13, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(model_labels, fontsize=10, rotation=15)
ax4.set_ylim(0, 6)
ax4.grid(True, alpha=0.3, axis='y')

for i, (bar, dist) in enumerate(zip(bars4, combined_distances)):
    ax4.text(i, dist/2, f'{dist:.1f}m', ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# No summary text - keep figure clean
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle

# Save
output_dir = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/paper/figures")
output_file = output_dir / "figure_robustness_methods_comparison.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf', facecolor='white')
print(f" Robustness methods comparison saved to: {output_file}")

output_file_png = output_dir / "figure_robustness_methods_comparison.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f" PNG preview saved to: {output_file_png}")

# plt.show()  # Commented out

print("\n" + "="*80)
print("ROBUSTNESS METHODS COMPARISON SUMMARY (ABSOLUTE DISTANCES)")
print("="*80)
print(f"BASELINE PERFORMANCE:")
for i, (label, dist) in enumerate(zip(model_labels, baseline_distances)):
    print(f"  {label}: {dist:.1f}m")

print(f"\nSENSOR NOISE (σ=0.10):")
for i, (label, dist) in enumerate(zip(model_labels, noise_distances)):
    print(f"  {label}: {dist:.1f}m")

print(f"\nJOINT FAILURE (avg across 8 joints):")
for i, (label, dist) in enumerate(zip(model_labels, joint_distances)):
    print(f"  {label}: {dist:.1f}m")

print(f"\nCOMBINED STRESS:")
for i, (label, dist) in enumerate(zip(model_labels, combined_distances)):
    print(f"  {label}: {dist:.1f}m")

print("\n" + "="*80)
print(" SIMPLIFIED 4-panel figure with ABSOLUTE DISTANCES (no retention %)")
print(" M1 now included in all comparisons!")
print(" Clear answer: M3 performs best overall")
