#!/usr/bin/env python3
"""
Render RealAnt robot from top-down view for paper figure
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, '/Users/anandpatel/Documents/4th Year/robust-quadruped-rl')

try:
    # Create RealAnt environment
    env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')

    # Reset to get initial state
    obs, info = env.reset()

    # Set camera to top-down view
    # Access MuJoCo viewer/camera settings
    if hasattr(env.unwrapped, 'mujoco_renderer'):
        renderer = env.unwrapped.mujoco_renderer
        # Set camera distance, elevation, and azimuth
        renderer.viewer._camera_settings['distance'] = 2.5  # Distance from robot
        renderer.viewer._camera_settings['elevation'] = -90  # Top-down (90 degrees down)
        renderer.viewer._camera_settings['azimuth'] = 0      # Front-facing orientation
        renderer.viewer._camera_settings['lookat'] = np.array([0, 0, 0.5])  # Look at robot center

    # Render the scene
    img = env.render()

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title('RealAnt Quadruped Robot (Top-Down View)',
                 fontsize=14, fontweight='bold', pad=10)

    # Save
    output_dir = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/paper/figures")
    output_file = output_dir / "realant_topdown.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f" RealAnt top-down view saved to: {output_file}")

    # Also save as PDF
    output_file_pdf = output_dir / "realant_topdown.pdf"
    plt.savefig(output_file_pdf, dpi=300, bbox_inches='tight', format='pdf', facecolor='white')
    print(f" PDF version saved to: {output_file_pdf}")

    plt.close()
    env.close()

    print("\n SUCCESS: RealAnt robot rendered from top-down view")
    print(f"Image size: {img.shape}")

except Exception as e:
    print(f"❌ ERROR: {e}")
    print("\nTrying alternative approach with manual camera setup...")

    # Alternative: Use MuJoCo directly
    try:
        import mujoco

        # Load the model
        model_path = "/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/src/envs/assets/realant.xml"
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)

        # Create renderer
        renderer = mujoco.Renderer(model, height=800, width=800)

        # Set camera for top-down view
        renderer.update_scene(data, camera="topdown")

        # If no topdown camera, set manually
        camera = mujoco.MjvCamera()
        camera.distance = 2.5
        camera.elevation = -90  # Top-down
        camera.azimuth = 0
        camera.lookat = np.array([0, 0, 0.5])

        # Render
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        img = renderer.render()

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title('RealAnt Quadruped Robot (Top-Down View)',
                     fontsize=14, fontweight='bold', pad=10)

        # Save
        output_dir = Path("/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/paper/figures")
        output_file = output_dir / "realant_topdown.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f" RealAnt top-down view saved to: {output_file}")

        output_file_pdf = output_dir / "realant_topdown.pdf"
        plt.savefig(output_file_pdf, dpi=300, bbox_inches='tight', format='pdf', facecolor='white')
        print(f" PDF version saved to: {output_file_pdf}")

        plt.close()

        print("\n SUCCESS: Alternative rendering approach worked!")

    except Exception as e2:
        print(f"❌ Alternative approach also failed: {e2}")
        print("\nPlease run the environment manually and take a screenshot.")
