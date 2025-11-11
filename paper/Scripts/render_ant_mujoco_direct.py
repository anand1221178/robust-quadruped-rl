#!/usr/bin/env python3
"""
Render ant robot using MuJoCo directly with camera control
"""

import mujoco
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Create Ant environment to get the model
import gymnasium as gym
env = gym.make('Ant-v4')
model = env.unwrapped.model
data = env.unwrapped.data

# Reset to initial state
mujoco.mj_resetData(model, data)

# Set joints to a natural standing pose with bent legs
# Ant has 8 joints (hip and ankle for each of 4 legs)
# Negative angles bend legs downward
data.qpos[7:15] = [-0.5, -0.8, -0.5, -0.8, -0.5, -0.8, -0.5, -0.8]  # Hip and ankle bent down

mujoco.mj_forward(model, data)

# Create renderer (use smaller size due to framebuffer limitations)
renderer = mujoco.Renderer(model, height=480, width=480)

# Configure camera for angled top-down view (not directly above)
camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_FREE
camera.lookat = np.array([0.0, 0.0, 0.4])  # Look at center of robot
camera.distance = 2.8   # Distance from lookat point
camera.azimuth = 135    # Rotation around vertical axis (45° offset for better view)
camera.elevation = -50  # Angle from horizontal (50° down - less steep)

# Update scene with camera
renderer.update_scene(data, camera=camera)

# Render
img = renderer.render()

# Create figure
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img)
ax.axis('off')
ax.set_title('Ant Quadruped Robot (Top-Down View)',
             fontsize=14, fontweight='bold', pad=10)

# Save
output_dir = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/paper/figures")
output_file = output_dir / "realant_simulation.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f" Ant top-down view saved to: {output_file}")

# Also save as PDF
output_file_pdf = output_dir / "realant_simulation.pdf"
plt.savefig(output_file_pdf, dpi=300, bbox_inches='tight', format='pdf', facecolor='white')
print(f" PDF version saved to: {output_file_pdf}")

plt.close()
env.close()

print(f"\n SUCCESS: Robot rendered from top-down view")
print(f"Camera settings: distance={camera.distance}, elevation={camera.elevation}°, azimuth={camera.azimuth}°")
print(f"Image shape: {img.shape}")
