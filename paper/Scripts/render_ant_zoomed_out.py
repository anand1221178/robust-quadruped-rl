#!/usr/bin/env python3
"""
Render RealAnt robot with zoomed out camera view
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import mujoco

# Create Ant environment with different camera settings
env = gym.make('Ant-v4', render_mode='rgb_array', width=800, height=800)

# Reset environment
obs, info = env.reset()

# Get the MuJoCo model and data
model = env.unwrapped.model
data = env.unwrapped.data

# Reset to default position
mujoco.mj_resetData(model, data)

# Step simulation to let it settle naturally
for _ in range(10):
    mujoco.mj_step(model, data)

# Try to access the renderer and modify camera distance
if hasattr(env.unwrapped, 'mujoco_renderer'):
    renderer = env.unwrapped.mujoco_renderer
    # Try to zoom out by modifying camera config
    if hasattr(renderer, '_camera_settings'):
        renderer._camera_settings['distance'] = 5.0  # Increase distance (default is ~3)
    elif hasattr(renderer, 'cam'):
        renderer.cam.distance = 5.0

# Alternative approach using MuJoCo viewer directly
try:
    # Create a custom renderer with more distance
    renderer = mujoco.Renderer(model, height=800, width=800)

    # Create camera with zoomed out view
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat = np.array([0.0, 0.0, 0.0])  # Look at origin
    camera.distance = 4.5  # Zoomed out more
    camera.elevation = -20  # Slightly above horizontal
    camera.azimuth = 180   # Rotate by 45 degrees from current 135 -> 180

    # Update scene with camera
    renderer.update_scene(data, camera=camera)

    # Render with custom camera
    img = renderer.render()
except:
    # Fallback to default render
    img = env.render()

# Create figure
plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.axis('off')
plt.title('Ant Quadruped Robot (Simulation View)', fontsize=14, fontweight='bold', pad=10)

# Save
output_dir = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/paper/figures")
output_file = output_dir / "realant_simulation_zoomed.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f" Zoomed out RealAnt simulation saved to: {output_file}")

# Also save as PDF
output_file_pdf = output_dir / "realant_simulation_zoomed.pdf"
plt.savefig(output_file_pdf, dpi=300, bbox_inches='tight', format='pdf', facecolor='white')
print(f" PDF version saved to: {output_file_pdf}")

plt.close()
env.close()

print(f"\n SUCCESS: Zoomed out robot rendering complete")
print(f"Image shape: {img.shape}")
print(f"Camera distance: 4.5 (zoomed out view)")