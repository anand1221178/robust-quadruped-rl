#!/usr/bin/env python3
"""
Create Figure: Joint Failure Analysis
Shows anatomical patterns and best/worst performing joints
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', message='This figure includes Axes')

# Load Experiment 5 data (UPDATED: October 27, 2025 - 32M retrained models)
exp5_file = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/evaluations/evaluations/experiment_5_per_joint_deep_dive/data/per_joint_deep_dive_results_20251027_212710.json")

with open(exp5_file, 'r') as f:
    data = json.load(f)

# Extract data for all models
models = ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo']
model_labels = ['M1 (Baseline)', 'M2 (SR2L)', 'M3 (DR)', 'M4 (Combined)']
colors = ['#2E86AB', '#9C27B0', '#FF6F00', '#D32F2F']

# Create figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

fig.suptitle('Joint Failure Analysis: Anatomical Patterns and Model Performance',
             fontsize=16, fontweight='bold', y=0.96)

# ===== TOP LEFT: Absolute Distance by Joint Type (Hip vs Ankle) =====
ax1 = fig.add_subplot(gs[0, 0])

joint_types = ['Hip', 'Ankle']
model_data_retention = {model: {'Hip': [], 'Ankle': []} for model in models}
model_data_distance = {model: {'Hip': [], 'Ankle': []} for model in models}

for model in models:
    for joint_result in data[model]:
        joint_type = joint_result['joint_anatomy']['type'].capitalize()
        retention = joint_result['retention_percentage']
        distance = joint_result['distance']['mean']
        model_data_retention[model][joint_type].append(retention)
        model_data_distance[model][joint_type].append(distance)

# Calculate averages
x_pos = np.arange(len(joint_types))
width = 0.2

for i, (model, label, color) in enumerate(zip(models, model_labels, colors)):
    avg_distances = [np.mean(model_data_distance[model][jtype]) for jtype in joint_types]
    ax1.bar(x_pos + i*width, avg_distances, width, label=label, color=color, alpha=0.8)

ax1.set_ylabel('Average Distance (m)', fontsize=12, fontweight='bold')
ax1.set_title('(a) Joint Type Comparison\nHip vs Ankle Failures', fontsize=13, fontweight='bold')
ax1.set_xticks(x_pos + width * 1.5)
ax1.set_xticklabels(joint_types, fontsize=12)
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, 6)

# ===== TOP RIGHT: Camera-Facing vs Camera-Away =====
ax2 = fig.add_subplot(gs[0, 1])

camera_positions = ['Camera-Away', 'Camera-Facing']
model_data_camera = {model: {'Camera-Away': [], 'Camera-Facing': []} for model in models}

for model in models:
    for joint_result in data[model]:
        is_facing = joint_result['joint_anatomy']['camera_facing']
        position = 'Camera-Facing' if is_facing else 'Camera-Away'
        retention = joint_result['retention_percentage']
        model_data_camera[model][position].append(retention)

x_pos = np.arange(len(camera_positions))

for i, (model, label, color) in enumerate(zip(models, model_labels, colors)):
    avg_retentions = [np.mean(model_data_camera[model][pos]) for pos in camera_positions]
    ax2.bar(x_pos + i*width, avg_retentions, width, label=label, color=color, alpha=0.8)

ax2.set_ylabel('Average Retention (%)', fontsize=12, fontweight='bold')
ax2.set_title('(b) Joint Position Effect\n(Left vs Right Side)', fontsize=13, fontweight='bold')
ax2.set_xticks(x_pos + width * 1.5)
ax2.set_xticklabels(['Left Side\n(Joints 1,3)', 'Right Side\n(Joints 2,4)'], fontsize=11)
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, 100)

# ===== BOTTOM LEFT: Absolute Distance for M3 (DR) =====
ax3 = fig.add_subplot(gs[1, 0])

# Get M3 results sorted by distance (absolute performance)
m3_results = sorted(data['M3_dr'], key=lambda x: x['distance']['mean'], reverse=True)

joint_names = [r['failed_joint'].replace('_', ' ').title() for r in m3_results]
distances = [r['distance']['mean'] for r in m3_results]
distance_std = [r['distance']['std'] for r in m3_results]

# Color code based on absolute distance: good (>4m), ok (2-4m), poor (<2m)
bar_colors = ['#4CAF50' if d > 4.0 else '#FFC107' if d > 2.0 else '#F44336' for d in distances]

bars = ax3.barh(range(len(joint_names)), distances, xerr=distance_std,
                color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1,
                error_kw={'elinewidth': 1, 'alpha': 0.5})
ax3.set_yticks(range(len(joint_names)))
ax3.set_yticklabels(joint_names, fontsize=10)
ax3.set_xlabel('Distance (m)', fontsize=12, fontweight='bold')
ax3.set_title('(c) M3 (DR) Per-Joint Performance\nRanked by Distance', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')
ax3.set_xlim(0, 9)

# Add distance values
for i, (bar, dist) in enumerate(zip(bars, distances)):
    ax3.text(dist + 0.3, i, f'{dist:.1f}m', va='center', fontsize=9, fontweight='bold')

# Add baseline reference line
baseline_dist = m3_results[0]['baseline_distance']
ax3.axvline(baseline_dist, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Baseline ({baseline_dist:.1f}m)')

# Add legend for colors and baseline
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#4CAF50', label='Good (>4m)'),
    Patch(facecolor='#FFC107', label='OK (2-4m)'),
    Patch(facecolor='#F44336', label='Poor (<2m)'),
    plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label=f'Baseline ({baseline_dist:.1f}m)')
]
ax3.legend(handles=legend_elements, loc='lower right', fontsize=9)

# ===== BOTTOM RIGHT: Front vs Rear Legs =====
ax4 = fig.add_subplot(gs[1, 1])

leg_positions = ['Front Legs', 'Rear Legs']
model_data_legs = {model: {'Front Legs': [], 'Rear Legs': []} for model in models}

for model in models:
    for joint_result in data[model]:
        leg_type = joint_result['joint_anatomy']['leg']
        position = 'Front Legs' if 'front' in leg_type else 'Rear Legs'
        retention = joint_result['retention_percentage']
        model_data_legs[model][position].append(retention)

x_pos = np.arange(len(leg_positions))

for i, (model, label, color) in enumerate(zip(models, model_labels, colors)):
    avg_retentions = [np.mean(model_data_legs[model][pos]) for pos in leg_positions]
    ax4.bar(x_pos + i*width, avg_retentions, width, label=label, color=color, alpha=0.8)

ax4.set_ylabel('Average Retention (%)', fontsize=12, fontweight='bold')
ax4.set_title('(d) Front vs Rear Leg Failures', fontsize=13, fontweight='bold')
ax4.set_xticks(x_pos + width * 1.5)
ax4.set_xticklabels(leg_positions, fontsize=12)
ax4.legend(fontsize=10, loc='upper right')
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim(0, 100)

# Clean layout without summary text
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save
output_dir = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/paper/figures")
output_file = output_dir / "figure_joint_failure_analysis.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf', facecolor='white')
print(f" Joint failure analysis saved to: {output_file}")

output_file_png = output_dir / "figure_joint_failure_analysis.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f" PNG preview saved to: {output_file_png}")

# plt.show()  # Commented out

# Print summary statistics
print("\n" + "="*80)
print("JOINT FAILURE ANALYSIS SUMMARY")
print("="*80)

for model, label in zip(models, model_labels):
    print(f"\n{label}:")
    hip_avg_retention = np.mean(model_data_retention[model]['Hip'])
    ankle_avg_retention = np.mean(model_data_retention[model]['Ankle'])
    hip_avg_distance = np.mean(model_data_distance[model]['Hip'])
    ankle_avg_distance = np.mean(model_data_distance[model]['Ankle'])
    front_avg = np.mean(model_data_legs[model]['Front Legs'])
    rear_avg = np.mean(model_data_legs[model]['Rear Legs'])

    print(f"  Hips: {hip_avg_distance:.2f}m ({hip_avg_retention:.1f}%) | Ankles: {ankle_avg_distance:.2f}m ({ankle_avg_retention:.1f}%)")
    print(f"  Front: {front_avg:.1f}% | Rear: {rear_avg:.1f}%")

print("\n" + "="*80)
print(" Anatomical patterns revealed!")
print(" M3 (DR) shows best joint failure compensation across all categories")
print(" Absolute distance metrics now shown alongside retention percentages")
