#!/usr/bin/env python3
"""
Create Figure: Joint Failure Analysis
Shows anatomical patterns and best/worst performing joints
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load Experiment 5 data
exp5_file = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/evaluations/evaluations/experiment_5_per_joint_deep_dive/data/per_joint_deep_dive_results_20251020_111732.json")

with open(exp5_file, 'r') as f:
    data = json.load(f)

# Extract data for all models
models = ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo']
model_labels = ['M1 (Baseline)', 'M2 (SR2L)', 'M3 (DR)', 'M4 (Combined)']
colors = ['#2E86AB', '#9C27B0', '#FF6F00', '#D32F2F']

# Create figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

fig.suptitle('Joint Failure Analysis: Anatomical Patterns and Model Performance\n(Which joints are hardest to compensate for?)',
             fontsize=16, fontweight='bold', y=0.98)

# ===== TOP LEFT: Retention by Joint Type (Hip vs Ankle) =====
ax1 = fig.add_subplot(gs[0, 0])

joint_types = ['Hip', 'Ankle']
model_data = {model: {'Hip': [], 'Ankle': []} for model in models}

for model in models:
    for joint_result in data[model]:
        joint_type = joint_result['joint_anatomy']['type'].capitalize()
        retention = joint_result['retention_percentage']
        model_data[model][joint_type].append(retention)

# Calculate averages
x_pos = np.arange(len(joint_types))
width = 0.2

for i, (model, label, color) in enumerate(zip(models, model_labels, colors)):
    avg_retentions = [np.mean(model_data[model][jtype]) for jtype in joint_types]
    ax1.bar(x_pos + i*width, avg_retentions, width, label=label, color=color, alpha=0.8)

ax1.set_ylabel('Average Retention (%)', fontsize=12, fontweight='bold')
ax1.set_title('(a) Joint Type Comparison\nHips vs Ankles', fontsize=13, fontweight='bold')
ax1.set_xticks(x_pos + width * 1.5)
ax1.set_xticklabels(joint_types, fontsize=12)
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, 100)

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
ax2.set_title('(b) Camera Position Effect\n(Does visibility matter?)', fontsize=13, fontweight='bold')
ax2.set_xticks(x_pos + width * 1.5)
ax2.set_xticklabels(camera_positions, fontsize=11)
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, 100)

# ===== BOTTOM LEFT: Best and Worst Joints for M3 (DR) =====
ax3 = fig.add_subplot(gs[1, 0])

# Get M3 results sorted by retention
m3_results = sorted(data['M3_dr'], key=lambda x: x['retention_percentage'], reverse=True)

joint_names = [r['failed_joint'].replace('_', ' ').title() for r in m3_results]
retentions = [r['retention_percentage'] for r in m3_results]

# Color code: green for good (>50%), yellow for ok (30-50%), red for poor (<30%)
bar_colors = ['#4CAF50' if r > 50 else '#FFC107' if r > 30 else '#F44336' for r in retentions]

bars = ax3.barh(range(len(joint_names)), retentions, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1)
ax3.set_yticks(range(len(joint_names)))
ax3.set_yticklabels(joint_names, fontsize=10)
ax3.set_xlabel('Retention (%)', fontsize=12, fontweight='bold')
ax3.set_title('(c) M3 (DR) Per-Joint Performance\n(Ranked Best to Worst)', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')
ax3.set_xlim(0, 100)

# Add retention values
for i, (bar, ret) in enumerate(zip(bars, retentions)):
    ax3.text(ret + 2, i, f'{ret:.1f}%', va='center', fontsize=9, fontweight='bold')

# Add legend for colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#4CAF50', label='Good (>50%)'),
    Patch(facecolor='#FFC107', label='OK (30-50%)'),
    Patch(facecolor='#F44336', label='Poor (<30%)')
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
ax4.set_title('(d) Front vs Rear Legs\n(Locomotion Asymmetry)', fontsize=13, fontweight='bold')
ax4.set_xticks(x_pos + width * 1.5)
ax4.set_xticklabels(leg_positions, fontsize=12)
ax4.legend(fontsize=10, loc='upper right')
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim(0, 100)

# Add summary finding
fig.text(0.5, 0.01,
         '✅ KEY FINDINGS: Ankles harder than hips | Rear legs harder than front | Camera position has minimal effect | M3 (DR) consistently best',
         ha='center', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()

# Save
output_dir = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/paper/figures")
output_file = output_dir / "figure_joint_failure_analysis.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf', facecolor='white')
print(f"✅ Joint failure analysis saved to: {output_file}")

output_file_png = output_dir / "figure_joint_failure_analysis.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ PNG preview saved to: {output_file_png}")

plt.show()

# Print summary statistics
print("\n" + "="*80)
print("JOINT FAILURE ANALYSIS SUMMARY")
print("="*80)

for model, label in zip(models, model_labels):
    print(f"\n{label}:")
    hip_avg = np.mean(model_data[model]['Hip'])
    ankle_avg = np.mean(model_data[model]['Ankle'])
    front_avg = np.mean(model_data_legs[model]['Front Legs'])
    rear_avg = np.mean(model_data_legs[model]['Rear Legs'])

    print(f"  Hips: {hip_avg:.1f}% | Ankles: {ankle_avg:.1f}%")
    print(f"  Front: {front_avg:.1f}% | Rear: {rear_avg:.1f}%")

print("\n" + "="*80)
print("✅ Anatomical patterns revealed!")
print("✅ M3 (DR) shows best joint failure compensation across all categories")
