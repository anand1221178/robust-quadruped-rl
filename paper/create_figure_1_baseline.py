#!/usr/bin/env python3
"""
Create Figure 1: Baseline Performance Comparison
IEEE conference paper quality
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load the latest baseline results (UPDATED: October 27, 2025 - 32M retrained models)
data_file = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/evaluations/evaluations/experiment_1_baseline/data/baseline_results_20251027_195743.json")

with open(data_file, 'r') as f:
    data = json.load(f)

# Extract data for all models
models = ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo']
model_labels = ['M1\n(Baseline)', 'M2\n(SR2L)', 'M3\n(DR)', 'M4\n(Combined)']

distances = [data[m]['distance']['mean'] for m in models]
distance_stds = [data[m]['distance']['std'] for m in models]
success_rates = [data[m]['success_rate'] * 100 for m in models]  # Convert to percentage

# Create figure with publication quality
fig, ax1 = plt.subplots(figsize=(8, 5))

# Set up x positions
x = np.arange(len(models))
width = 0.6

# Plot distance bars
bars = ax1.bar(x, distances, width, yerr=distance_stds,
               capsize=5, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
               edgecolor='black', linewidth=1.2, alpha=0.85)

# Add value labels on bars
for i, (bar, dist, success) in enumerate(zip(bars, distances, success_rates)):
    height = bar.get_height()
    # Distance value
    ax1.text(bar.get_x() + bar.get_width()/2., height + distance_stds[i] + 0.3,
             f'{dist:.2f}m',
             ha='center', va='bottom', fontweight='bold', fontsize=11)
    # Success rate below
    ax1.text(bar.get_x() + bar.get_width()/2., 0.5,
             f'{success:.0f}%',
             ha='center', va='bottom', fontsize=9, color='white', fontweight='bold')

# Styling
ax1.set_xlabel('Model Configuration', fontsize=13, fontweight='bold')
ax1.set_ylabel('Distance Traveled (m)', fontsize=13, fontweight='bold')
ax1.set_title('Baseline Performance Comparison\n(Clean Environment, No Failures)',
              fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(model_labels, fontsize=11)
ax1.set_ylim(0, max(distances) + max(distance_stds) + 2)
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax1.set_axisbelow(True)

# Tight layout (annotations removed - incorrect with new data where M3 is best)
plt.tight_layout()

# Save figure
output_dir = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/paper/figures")
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "figure_1_baseline_performance.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"✅ Figure 1 saved to: {output_file}")

# Also save PNG for preview
output_file_png = output_dir / "figure_1_baseline_performance.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
print(f"✅ PNG preview saved to: {output_file_png}")

# Show the figure
# plt.show()  # Commented out to avoid hanging in automated runs

# Print summary statistics
print("\n" + "="*60)
print("BASELINE PERFORMANCE SUMMARY")
print("="*60)
for i, (label, dist, std, success) in enumerate(zip(model_labels, distances, distance_stds, success_rates)):
    print(f"{label.replace(chr(10), ' '):<20} | {dist:6.2f}m ± {std:4.2f} | Success: {success:5.1f}%")

print("\n" + "="*60)
print("PERFORMANCE COMPARISON (M3 is BEST)")
print("="*60)
m3_dist = distances[2]  # M3 (DR) - best performer
m1_dist = distances[0]  # M1 (Baseline) - overfit
m2_dist = distances[1]  # M2 (SR2L)
m4_dist = distances[3]  # M4 (Combined)

print(f"M3 (DR) is BEST: {m3_dist:.2f}m")
print(f"M1 underperforms M3 by: {((m3_dist - m1_dist)/m3_dist)*100:.1f}% (overfitting)")
print(f"M2 underperforms M3 by: {((m3_dist - m2_dist)/m3_dist)*100:.1f}%")
print(f"M4 underperforms M3 by: {((m3_dist - m4_dist)/m3_dist)*100:.1f}%")
print("="*60)
