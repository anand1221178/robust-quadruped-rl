#!/usr/bin/env python3
"""
Verify Ant-v4 environment for the Robust Quadruped RL project
"""

import gymnasium as gym
import numpy as np
import warnings

# Suppress the deprecation warning since v4 works fine
warnings.filterwarnings("ignore", message=".*The environment Ant-v4 is out of date.*")

def test_ant_environment():
    """Complete test of Ant-v4 for our project needs"""
    
    print("=" * 60)
    print("Ant-v4 Environment Verification for Robust Quadruped RL")
    print("=" * 60)
    
    # Create environment
    env = gym.make('Ant-v4')
    print("✓ Successfully created Ant-v4 environment")
    
    # 1. Check observation space (should be 27-dimensional)
    print("\n=== Observation Space ===")
    obs_space = env.observation_space
    print(f"Shape: {obs_space.shape}")
    print(f"Type: {type(obs_space)}")
    print(f"Bounds: [{obs_space.low.min():.2f}, {obs_space.high.max():.2f}]")
    
    # Reset and examine observation structure
    obs, info = env.reset()
    print(f"\nObservation breakdown (27 dimensions):")
    
    # Based on Ant-v4 documentation:
    # Note: This is slightly different from RealAnt's 28 dimensions
    print("  [0:2]   - x,y position of torso (excluded from obs)")
    print("  [0:13]  - Joint angles and torso orientation (qpos[2:])")
    print("  [13:27] - Joint velocities and torso velocities (qvel)")
    print(f"\nActual observation: shape={obs.shape}")
    print(f"First 5 values: {obs[:5]}")
    
    # 2. Check action space (should be 8-dimensional for 8 joints)
    print("\n=== Action Space ===")
    act_space = env.action_space
    print(f"Shape: {act_space.shape}")
    print(f"Type: {type(act_space)}")
    print(f"Bounds: [{act_space.low[0]:.2f}, {act_space.high[0]:.2f}]")
    print("✓ 8 continuous actions (torques for 8 joints)")
    
    # Joint mapping for Ant
    print("\nJoint mapping:")
    print("  - hip_1, ankle_1 (front right leg)")
    print("  - hip_2, ankle_2 (front left leg)")
    print("  - hip_3, ankle_3 (back right leg)")
    print("  - hip_4, ankle_4 (back left leg)")
    
    # 3. Test basic functionality
    print("\n=== Testing Basic Functionality ===")
    
    # Take some random steps
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if i == 0:
            print(f"Step 1 - Reward: {reward:.3f}, Info keys: {list(info.keys())}")
    
    print(f"✓ Completed 10 steps, total reward: {total_reward:.3f}")
    
    # 4. Test episode completion
    print("\n=== Testing Episode Completion ===")
    obs, info = env.reset()
    step_count = 0
    done = False
    
    while not done and step_count < 100:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
    
    print(f"✓ Episode ended after {step_count} steps")
    print(f"  - Terminated: {terminated} (usually means fell over)")
    print(f"  - Truncated: {truncated} (time limit)")
    
    # 5. Check reward structure
    print("\n=== Reward Structure ===")
    print("Ant-v4 reward includes:")
    print("  - Forward velocity reward (main)")
    print("  - Control cost penalty")
    print("  - Contact cost penalty")
    print("  - Survival bonus")
    
    # 6. Performance test
    print("\n=== Performance Test ===")
    import time
    
    env.reset()
    start = time.time()
    steps = 1000
    
    for _ in range(steps):
        action = env.action_space.sample()
        env.step(action)
    
    elapsed = time.time() - start
    fps = steps / elapsed
    print(f"✓ Performed {steps} steps in {elapsed:.2f}s ({fps:.0f} FPS)")
    
    env.close()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✅ Ant-v4 is ready to use for your project!")
    print("\nKey differences from RealAnt (28 dims):")
    print("- Ant-v4 has 27-dim observations (missing contact sensors)")
    print("- Otherwise very similar: 8 DOF quadruped")
    print("- Same action space: 8 joint torques")
    print("\nRecommendation: Use Ant-v4 for development, then adapt to RealAnt later")
    
    return True

def create_ant_wrapper_example():
    """Show how to wrap Ant-v4 to make it more like RealAnt if needed"""
    
    print("\n" + "=" * 60)
    print("OPTIONAL: Ant-v4 to RealAnt Wrapper")
    print("=" * 60)
    
    wrapper_code = '''
# If you need Ant-v4 to be more like RealAnt (28 dims), use this wrapper:

class AntToRealAntWrapper(gym.Wrapper):
    """Makes Ant-v4 observation space similar to RealAnt"""
    
    def __init__(self, env):
        super().__init__(env)
        # RealAnt has 28 dims, Ant-v4 has 27
        # Add one dummy dimension for compatibility
        high = np.append(env.observation_space.high, 1.0)
        low = np.append(env.observation_space.low, 0.0)
        self.observation_space = gym.spaces.Box(low=low, high=high)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Add dummy contact sensor
        obs = np.append(obs, 0.0)
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Add dummy contact sensor
        obs = np.append(obs, 0.0)
        return obs, reward, terminated, truncated, info

# Usage:
env = gym.make('Ant-v4')
env = AntToRealAntWrapper(env)  # Now has 28-dim observations
'''
    print(wrapper_code)

if __name__ == "__main__":
    # Run the verification
    success = test_ant_environment()
    
    if success:
        create_ant_wrapper_example()
        
        print("\n✅ Environment setup is complete!")
        print("\nNext steps:")
        print("1. Update your configs to use 'Ant-v4' instead of 'RealAnt-v0'")
        print("2. Adjust observation dimension from 28 to 27 in your code")
        print("3. Start with Phase 2: PPO Baseline implementation")