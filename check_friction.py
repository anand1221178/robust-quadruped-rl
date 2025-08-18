#!/usr/bin/env python3
"""
Check and potentially modify RealAnt friction settings
"""
import sys
sys.path.append('src')

import gymnasium as gym
import realant_sim
import numpy as np

def check_friction():
    """Check RealAnt friction coefficients"""
    env = gym.make('RealAntMujoco-v0')
    
    print("RealAnt Friction Analysis")
    print("=" * 50)
    
    # Access MuJoCo model
    model = env.unwrapped.model
    
    # Check geom friction values
    print("\nGeometry Friction Coefficients:")
    print(f"Number of geoms: {model.ngeom}")
    
    for i in range(min(model.ngeom, 20)):  # Check first 20 geoms
        geom_name = model.geom(i).name
        friction = model.geom_friction[i]
        print(f"  {geom_name}: {friction}")
    
    # Check floor/ground friction
    print("\nFloor Contact Properties:")
    if hasattr(model, 'geom_friction'):
        floor_friction = model.geom_friction[0]  # Usually first geom is floor
        print(f"  Floor friction: {floor_friction}")
        print(f"  Recommended for walking: [1.0, 0.005, 0.0001]")
        print(f"  Current looks {'slippery' if floor_friction[0] < 0.8 else 'normal'}")
    
    # Test walking with different actions
    print("\nTesting movement...")
    obs, _ = env.reset()
    
    for i in range(10):
        action = np.ones(8) * 0.5  # Moderate forward action
        obs, reward, done, truncated, info = env.step(action)
        
        if hasattr(env.unwrapped, 'data'):
            qvel = env.unwrapped.data.qvel
            x_vel = qvel[0] if len(qvel) > 0 else 0
            print(f"  Step {i+1}: x_velocity = {x_vel:.3f} m/s")
    
    env.close()
    
    print("\nIf friction is too low (<0.5), the robot will slip.")
    print("Consider modifying the XML file or using a friction wrapper.")

if __name__ == "__main__":
    check_friction()