#!/usr/bin/env python3
"""
Create joint numbering diagram for RealAnt quadruped
Shows anatomical positions and joint naming convention
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Robot walks left→right (positive X direction)
# Camera view is from below/behind looking at rear

# Torso (center body)
torso_size = 0.8
torso = Circle((0, 0), torso_size/2,
               facecolor='#FFE5B4', edgecolor='black', linewidth=2.5, zorder=10)
ax.add_patch(torso)

# Add torso label
ax.text(0, 0, 'Torso', ha='center', va='center', fontsize=14, fontweight='bold', zorder=11)

# Leg positions (from XML coordinates, scaled for visualization)
# Each leg has: hip (proximal) and ankle (distal) joints
legs = {
    # Name: (x_pos, y_pos, hip_label, ankle_label, color)
    'front_left': {
        'pos': (0.8, 0.8),
        'hip': 'Hip 1',
        'ankle': 'Ankle 1',
        'color': '#4A90E2',  # Blue
        'number': '1',
        'side': 'Front-Left'
    },
    'front_right': {
        'pos': (-0.8, 0.8),
        'hip': 'Hip 2',
        'ankle': 'Ankle 2',
        'color': '#E24A4A',  # Red
        'number': '2',
        'side': 'Front-Right\n(Camera-Facing)'
    },
    'rear_left': {
        'pos': (-0.8, -0.8),
        'hip': 'Hip 3',
        'ankle': 'Ankle 3',
        'color': '#50C878',  # Green
        'number': '3',
        'side': 'Rear-Left'
    },
    'rear_right': {
        'pos': (0.8, -0.8),
        'hip': 'Hip 4',
        'ankle': 'Ankle 4',
        'color': '#FF8C00',  # Orange
        'number': '4',
        'side': 'Rear-Right\n(Camera-Facing)'
    }
}

# Draw legs
for leg_name, leg_data in legs.items():
    x, y = leg_data['pos']
    color = leg_data['color']

    # Hip joint (closer to torso)
    hip_pos = (x * 0.4, y * 0.4)
    hip = Circle(hip_pos, 0.15, facecolor=color, edgecolor='black',
                 linewidth=2, alpha=0.8, zorder=5)
    ax.add_patch(hip)

    # Ankle joint (farther from torso)
    ankle_pos = (x * 0.9, y * 0.9)
    ankle = Circle(ankle_pos, 0.12, facecolor=color, edgecolor='black',
                   linewidth=2, alpha=0.6, zorder=5)
    ax.add_patch(ankle)

    # Leg segment (line connecting hip to ankle)
    ax.plot([hip_pos[0], ankle_pos[0]], [hip_pos[1], ankle_pos[1]],
            color=color, linewidth=6, alpha=0.5, zorder=3)

    # Hip label
    ax.text(hip_pos[0], hip_pos[1], leg_data['hip'],
            ha='center', va='center', fontsize=9, fontweight='bold',
            color='white', zorder=6)

    # Ankle label
    ax.text(ankle_pos[0], ankle_pos[1], leg_data['ankle'],
            ha='center', va='center', fontsize=8, fontweight='bold',
            color='white', zorder=6)

    # Leg number (large, outside)
    label_offset_x = 1.4 if x > 0 else -1.4
    label_offset_y = 1.2 if y > 0 else -1.2

    # Number circle
    num_circle = Circle((label_offset_x, label_offset_y), 0.25,
                       facecolor=color, edgecolor='black',
                       linewidth=2.5, zorder=15)
    ax.add_patch(num_circle)

    ax.text(label_offset_x, label_offset_y, leg_data['number'],
            ha='center', va='center', fontsize=20, fontweight='bold',
            color='white', zorder=16)

    # Side label
    ax.text(label_offset_x, label_offset_y - 0.45, leg_data['side'],
            ha='center', va='top', fontsize=9, fontweight='bold',
            color=color, zorder=15)

# Add coordinate system arrows
# X-axis (forward direction)
arrow_x = FancyArrowPatch((0, -1.8), (1.2, -1.8),
                         arrowstyle='->', mutation_scale=30,
                         linewidth=3, color='darkgreen', zorder=20)
ax.add_patch(arrow_x)
ax.text(0.6, -2.0, 'Forward\n(+X)', ha='center', va='top',
        fontsize=11, fontweight='bold', color='darkgreen')

# Y-axis
arrow_y = FancyArrowPatch((-1.8, 0), (-1.8, 1.2),
                         arrowstyle='->', mutation_scale=30,
                         linewidth=3, color='darkblue', zorder=20)
ax.add_patch(arrow_y)
ax.text(-2.05, 0.6, '+Y', ha='right', va='center',
        fontsize=11, fontweight='bold', color='darkblue')

# Add camera position indicator
camera_x = 0
camera_y = -2.3
camera = patches.FancyBboxPatch((camera_x - 0.4, camera_y - 0.3), 0.8, 0.6,
                               boxstyle="round,pad=0.05",
                               facecolor='lightgray', edgecolor='black',
                               linewidth=2, zorder=20)
ax.add_patch(camera)
ax.text(camera_x, camera_y, '📷', ha='center', va='center',
        fontsize=24, zorder=21)
ax.text(camera_x, camera_y - 0.55, 'Camera View\n(from below/behind)',
        ha='center', va='top', fontsize=10, fontweight='bold')

# Title and annotations
ax.text(0, 2.2, 'RealAnt Joint Numbering Convention',
        ha='center', va='center', fontsize=18, fontweight='bold')

ax.text(0, 1.9, 'Top-Down View (8 Actuated Joints: 4 Hips + 4 Ankles)',
        ha='center', va='center', fontsize=12, style='italic')

# Add legend box
legend_text = (
    "Joint Types:\n"
    "• Hip: Proximal joint (closer to torso)\n"
    "• Ankle: Distal joint (end of leg)\n\n"
    "Observation Order: Joint positions/velocities\n"
    "indexed by joint number (1-4)"
)
ax.text(-2.5, 1.6, legend_text,
        fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Add note about camera-facing joints
note_text = (
    "Note: Joints 2 and 4 are camera-facing\n"
    "(critical for understanding worst-case\n"
    "failure patterns in Results Section 4.3)"
)
ax.text(2.5, 1.6, note_text,
        fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8,
                 edgecolor='orange', linewidth=2))

# Set axis properties
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 2.5)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()

# Save
output_dir = "/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/paper/figures"
output_file_pdf = f"{output_dir}/joint_numbering_diagram.pdf"
output_file_png = f"{output_dir}/joint_numbering_diagram.png"

plt.savefig(output_file_pdf, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Joint numbering diagram saved to: {output_file_pdf}")

plt.savefig(output_file_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ PNG preview saved to: {output_file_png}")

plt.show()
