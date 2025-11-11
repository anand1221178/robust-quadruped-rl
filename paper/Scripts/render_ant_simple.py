#!/usr/bin/env python3
"""
Simple script to render ant robot
Uses standard Gymnasium Ant environment with default view, zoomed out
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Create standard Ant environment with rendering (zoomed out by increasing camera distance)
env = gym.make('Ant-v4', render_mode='rgb_array', width=800, height=800,
               camera_name=None)  # Use default camera

# Reset environment
obs, info = env.reset()

# Take a few steps to let robot settle into natural pose
for _ in range(5):
    action = env.action_space.sample() * 0  # Zero action (robot stays still)
    obs, reward, terminated, truncated, info = env.step(action)

# Render with default camera
img = env.render()

# Create figure
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img)
ax.axis('off')
ax.set_title('Ant Quadruped Robot (Simulation View)',
             fontsize=14, fontweight='bold', pad=10)

# Save
output_dir = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/paper/figures")
output_file = output_dir / "realant_simulation.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f" Ant simulation view saved to: {output_file}")

# Also save as PDF
output_file_pdf = output_dir / "realant_simulation.pdf"
plt.savefig(output_file_pdf, dpi=300, bbox_inches='tight', format='pdf', facecolor='white')
print(f" PDF version saved to: {output_file_pdf}")

plt.close()
env.close()

print(f"\n SUCCESS: Robot rendered")
print(f"Image shape: {img.shape}")
