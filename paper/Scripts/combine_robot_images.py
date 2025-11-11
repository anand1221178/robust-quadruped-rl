#!/usr/bin/env python3
"""
Combine robot simulation and joint diagram into single side-by-side image
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import numpy as np
import subprocess

# Convert PDF to PNG using ImageMagick or similar
figures_dir = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/paper/figures")

# First, convert the PDF to PNG if needed
pdf_path = figures_dir / "joint_numbering_diagram.pdf"
png_path = figures_dir / "joint_numbering_diagram_temp.png"

# Use matplotlib to display PDF content directly
# We'll read both images and display them side by side

# Load the robot simulation image
robot_img = mpimg.imread(figures_dir / "realant_simulation_final.png")

# Use the ULTRA HIGH RESOLUTION converted PDF diagram
diagram_img = mpimg.imread(figures_dir / "joint_numbering_diagram_ultra_hd.png")

# Create figure with two subplots side by side with ZERO spacing
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'wspace': 0.0})

# Display images with no interpolation for maximum sharpness
# Crop robot image to remove its title (assuming title is in top ~15% of image)
robot_img_cropped = robot_img[int(robot_img.shape[0]*0.15):, :, :]
ax1.imshow(robot_img_cropped, interpolation='none')
ax1.axis('off')
ax1.set_title('(a) Simulated Robot', fontsize=12, pad=5)

ax2.imshow(diagram_img, interpolation='none')
ax2.axis('off')
ax2.set_title('(b) Joint Labeling Diagram', fontsize=12, pad=5)

# Add overall title
fig.suptitle('RealAnt Quadruped Robot', fontsize=14, fontweight='bold', y=0.98)

# Adjust spacing with ZERO padding for minimum space
plt.tight_layout(rect=[0, 0, 1, 0.95], w_pad=0.0)

# Save combined figure at ULTRA HIGH DPI
output_file = figures_dir / "realant_combined_figure.png"
plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
print(f" Combined figure saved to: {output_file}")

# Also save as PDF with high quality
output_file_pdf = figures_dir / "realant_combined_figure.pdf"
plt.savefig(output_file_pdf, dpi=600, bbox_inches='tight', format='pdf', facecolor='white')
print(f" PDF version saved to: {output_file_pdf}")

plt.close()

print("\n SUCCESS: Combined figure created with proper spacing")