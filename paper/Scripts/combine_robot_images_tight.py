#!/usr/bin/env python3
"""
Combine robot simulation and joint diagram with absolutely minimal spacing
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np

figures_dir = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/paper/figures")

# Load images
robot_img = mpimg.imread(figures_dir / "realant_simulation_final.png")
diagram_img = mpimg.imread(figures_dir / "joint_numbering_diagram_ultra_hd.png")

# Crop robot image to remove its title (top 15%)
robot_img_cropped = robot_img[int(robot_img.shape[0]*0.15):, :, :]

# Create figure with specific size
fig = plt.figure(figsize=(12, 5))

# Add axes manually with exact positions [left, bottom, width, height]
# Make them touch by calculating positions - NO GAP
ax1 = fig.add_axes([0.00, 0.15, 0.50, 0.7])  # Left image takes exactly half
ax2 = fig.add_axes([0.50, 0.15, 0.50, 0.7])  # Right image starts at 0.50 - TOUCHING!

# Display images
ax1.imshow(robot_img_cropped, interpolation='none')
ax1.axis('off')
ax1.set_title('(a) Simulated Robot', fontsize=12, pad=5)

ax2.imshow(diagram_img, interpolation='none')
ax2.axis('off')
ax2.set_title('(b) Joint Labeling Diagram', fontsize=12, pad=5)

# Add overall title
fig.suptitle('RealAnt Quadruped Robot', fontsize=14, fontweight='bold')

# Save with high quality - NO bbox_inches to preserve exact layout
output_file = figures_dir / "realant_combined_figure.png"
plt.savefig(output_file, dpi=600, facecolor='white')
print(f" Combined figure saved to: {output_file}")

output_file_pdf = figures_dir / "realant_combined_figure.pdf"
plt.savefig(output_file_pdf, dpi=600, format='pdf', facecolor='white')
print(f" PDF version saved to: {output_file_pdf}")

plt.close()
print("\n SUCCESS: Tight spacing achieved with manual positioning")