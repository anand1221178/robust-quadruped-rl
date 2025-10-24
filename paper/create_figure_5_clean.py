#!/usr/bin/env python3
"""
Create Figure 5: Joint-Noise Interaction Analysis (Experiment 7) - CLEAN VERSION
Shows whether joint failures + sensor noise have synergistic or independent effects
NO TEXT ANNOTATIONS on bars/heatmap for publication quality
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load Experiment 7 data
exp7_file = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/evaluations/evaluations/experiment_7_joint_noise_ablation/data/joint_noise_ablation_20251020_153154.json")

with open(exp7_file, 'r') as f:
    data = json.load(f)

# Models to analyze
models = ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo']
model_labels = ['M1 (Baseline)', 'M2 (SR2L)', 'M3 (DR)', 'M4 (Combined)']
colors = ['#2E86AB', '#9C27B0', '#FF6F00', '#D32F2F']

# Create figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

# NO FIGURE-LEVEL SUPTITLE for cleaner look

# ===== TOP LEFT: Interaction Matrix for M3 (DR) =====
ax1 = fig.add_subplot(gs[0, :])

# Get M3 data and organize into matrix
m3_data = data['M3_dr']
baseline = m3_data['baseline_distance']

# Organize by joint and noise level
joints = ['hip_1', 'ankle_1', 'hip_2', 'ankle_2', 'hip_3', 'ankle_3', 'hip_4', 'ankle_4']
# Use actual noise levels from experiment 7 data
noise_levels = [0.0, 0.0044, 0.10, 1.0]

# Create matrix: rows = joints, columns = noise levels
matrix_data = np.zeros((len(joints), len(noise_levels)))

for i, joint in enumerate(joints):
    for j, noise in enumerate(noise_levels):
        # Find matching result
        matches = [r for r in m3_data['ablation_results']
                  if r['failed_joint'] == joint and r['noise_level'] == noise]
        if matches:
            distance = matches[0]['distance']['mean']
            retention = (distance / baseline) * 100
            matrix_data[i, j] = retention

# Plot heatmap
im = ax1.imshow(matrix_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

# Set labels
joint_labels = [j.replace('_', ' ').title() for j in joints]
noise_labels = [f'σ={n:.2f}' for n in noise_levels]

ax1.set_xticks(np.arange(len(noise_levels)))
ax1.set_yticks(np.arange(len(joints)))
ax1.set_xticklabels(noise_labels, fontsize=11)
ax1.set_yticklabels(joint_labels, fontsize=11)

# Add percentage values to heatmap cells
for i in range(len(joints)):
    for j in range(len(noise_levels)):
        value = matrix_data[i, j]
        # All text in black for consistency
        ax1.text(j, i, f'{value:.0f}%',
                ha='center', va='center', color='black',
                fontsize=9, fontweight='bold')

ax1.set_xlabel('Sensor Noise Level', fontsize=13, fontweight='bold')
ax1.set_ylabel('Failed Joint', fontsize=13, fontweight='bold')
ax1.set_title('M3 (DR) Retention: Joint Failure × Sensor Noise',
              fontsize=13, fontweight='bold', pad=15)

# Add colorbar
cbar = plt.colorbar(im, ax=ax1, orientation='vertical', pad=0.02)
cbar.set_label('Retention (%)', fontsize=11, fontweight='bold')

# ===== BOTTOM LEFT: Additivity Test =====
ax2 = fig.add_subplot(gs[1, 0])

# For each model, test if combined effect = sum of individual effects
# Theory: Retention(joint + noise) ≈ Retention(joint only) × Retention(noise only) / 100

additivity_scores = []
model_names_plot = []

for model, label in zip(models, model_labels):
    model_data = data[model]
    baseline = model_data['baseline_distance']

    # Get examples: hip_1 + noise=0.10
    joint_only = [r for r in model_data['ablation_results']
                  if r['failed_joint'] == 'hip_1' and r['noise_level'] == 0.0]
    noise_only = [r for r in model_data['ablation_results']
                  if r['failed_joint'] is None and r['noise_level'] == 0.10] if any('failed_joint' in r and r['failed_joint'] is None for r in model_data['ablation_results']) else []
    combined = [r for r in model_data['ablation_results']
               if r['failed_joint'] == 'hip_1' and r['noise_level'] == 0.10]

    if not noise_only:
        # Use average across all joints at noise=0.10 as proxy
        noise_results = [r for r in model_data['ablation_results'] if r['noise_level'] == 0.10]
        if noise_results:
            avg_noise_dist = np.mean([r['distance']['mean'] for r in noise_results])
            noise_retention = (avg_noise_dist / baseline) * 100
        else:
            continue
    else:
        noise_retention = (noise_only[0]['distance']['mean'] / baseline) * 100

    if joint_only and combined:
        joint_retention = (joint_only[0]['distance']['mean'] / baseline) * 100
        combined_retention = (combined[0]['distance']['mean'] / baseline) * 100

        # Predicted if additive: R_joint × R_noise / 100
        predicted_retention = (joint_retention * noise_retention) / 100

        # Additivity score: how close is actual to predicted?
        # 100% = perfect additivity, >100% = synergy, <100% = interference
        additivity = (combined_retention / predicted_retention) * 100 if predicted_retention > 0 else 0
        additivity_scores.append(additivity)
        model_names_plot.append(label)

if additivity_scores:
    bars = ax2.bar(range(len(additivity_scores)), additivity_scores,
                   color=colors[:len(additivity_scores)], alpha=0.8, edgecolor='black', linewidth=2)
    ax2.axhline(y=100, color='black', linestyle='--', linewidth=2, label='Perfect Additivity')
    ax2.set_ylabel('Additivity Score (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Interaction Type (Hip_1 + σ=0.10)',
                  fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(model_names_plot)))
    ax2.set_xticklabels(model_names_plot, fontsize=11, rotation=15)
    # Adjust y-axis to accommodate high values
    max_score = max(additivity_scores) if additivity_scores else 120
    ax2.set_ylim(0, max(120, max_score * 1.1))
    # Add explicit y-axis ticks with numbers
    ax2.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax2.tick_params(axis='y', labelsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on top of bars for clarity
    for i, (bar, score) in enumerate(zip(bars, additivity_scores)):
        ax2.text(i, score + max_score*0.02, f'{score:.0f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# ===== BOTTOM RIGHT: Worst-Case Scenario Comparison =====
ax3 = fig.add_subplot(gs[1, 1])

# Find worst-case scenario for each model (most degradation)
worst_case_retentions = []
worst_case_scenarios = []

for model, label in zip(models, model_labels):
    model_data = data[model]
    baseline = model_data['baseline_distance']

    # Find minimum retention across all joint+noise combinations
    min_retention = 100
    worst_scenario = ""

    for result in model_data['ablation_results']:
        if result['noise_level'] > 0:  # Only combined stress
            retention = (result['distance']['mean'] / baseline) * 100
            if retention < min_retention:
                min_retention = retention
                worst_scenario = f"{result['failed_joint']} + σ={result['noise_level']:.2f}"

    worst_case_retentions.append(min_retention)
    worst_case_scenarios.append(worst_scenario)

bars3 = ax3.bar(range(len(models)), worst_case_retentions,
               color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax3.set_ylabel('Worst-Case Retention (%)', fontsize=12, fontweight='bold')
ax3.set_title('Worst-Case Scenario Performance',
              fontsize=13, fontweight='bold')
ax3.set_xticks(range(len(models)))
ax3.set_xticklabels(model_labels, fontsize=11, rotation=15)
ax3.set_ylim(0, 60)
# Add explicit y-axis ticks with numbers
ax3.yaxis.set_major_locator(plt.MaxNLocator(6))
ax3.tick_params(axis='y', labelsize=11)
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on top of bars
for i, (bar, ret) in enumerate(zip(bars3, worst_case_retentions)):
    ax3.text(i, ret + 2, f'{ret:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# NO SUMMARY TEXT BOX at bottom for cleaner publication look

plt.tight_layout()

# Save
output_dir = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/paper/figures")
output_file = output_dir / "figure_5_joint_noise_ablation.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf', facecolor='white')
print(f"✅ CLEAN joint-noise ablation figure saved to: {output_file}")

output_file_png = output_dir / "figure_5_joint_noise_ablation.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ CLEAN PNG preview saved to: {output_file_png}")

plt.show()

# Print summary
print("\n" + "="*80)
print("CLEAN FIGURE GENERATED - NO ANNOTATIONS")
print("="*80)
if additivity_scores:
    for label, score in zip(model_names_plot, additivity_scores):
        interaction_type = 'Independent' if 95 < score < 105 else 'Interference' if score < 95 else 'Synergy'
        print(f"{label}: {score:.1f}% additivity ({interaction_type})")

print("\nWorst-Case Scenarios:")
for label, ret, scenario in zip(model_labels, worst_case_retentions, worst_case_scenarios):
    print(f"  {label}: {ret:.1f}% - {scenario}")

print("\n" + "="*80)
print("✅ Clean publication-quality figure ready!")
print("="*80)
