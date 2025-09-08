#!/usr/bin/env python3
"""
BULLETPROOF verification that robot actually walks properly
No bullshit calculations - just raw position tracking
"""
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.target_walking_wrapper import TargetWalkingWrapper
import realant_sim
import time

def verify_walking_performance():
    """Verify robot actually walks at expected speed with rock-solid measurements"""
    
    print("🔍 BULLETPROOF WALKING VERIFICATION")
    print("No fancy calculations - just raw position tracking")
    print("=" * 60)
    
    # Model paths
    model_path = 'done/ppo_baseline_ueqbjf2x/best_model/best_model.zip'
    norm_path = 'done/ppo_baseline_ueqbjf2x/vec_normalize.pkl'
    
    # Create environment
    def make_env():
        env = gym.make('RealAntMujoco-v0')
        env = TargetWalkingWrapper(env, target_distance=5.0)
        return env
    
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(norm_path, env)
    env.training = False
    
    model = PPO.load(model_path)
    
    print("✅ Model and environment loaded")
    
    # Test 1: Raw position tracking
    print(f"\n🎯 TEST 1: RAW POSITION TRACKING")
    print("-" * 40)
    
    obs = env.reset()
    
    # Get initial position
    initial_x = env.envs[0].unwrapped.data.qpos[0]
    print(f"Initial position: {initial_x:.6f}m")
    
    # Track positions over time
    positions = [initial_x]
    timestamps = [0]
    
    # Run for exactly 500 steps (25 seconds at 50Hz)
    print("Running for 500 steps (25 seconds)...")
    
    for step in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        current_x = env.envs[0].unwrapped.data.qpos[0]
        positions.append(current_x)
        timestamps.append((step + 1) * 0.05)  # Each step = 0.05s
        
        # Print every 100 steps
        if (step + 1) % 100 == 0:
            distance_so_far = current_x - initial_x
            time_elapsed = (step + 1) * 0.05
            velocity_so_far = distance_so_far / time_elapsed
            print(f"  Step {step+1:3d}: x={current_x:.3f}m, dist={distance_so_far:.3f}m, vel={velocity_so_far:.3f}m/s")
        
        if done[0]:
            print(f"Episode ended early at step {step + 1}")
            break
    
    # Final calculations
    final_x = positions[-1]
    total_distance = final_x - initial_x
    total_time = len(positions) * 0.05
    average_velocity = total_distance / total_time
    
    print(f"\n📊 RAW MEASUREMENTS:")
    print(f"  Initial position: {initial_x:.6f}m")
    print(f"  Final position: {final_x:.6f}m")
    print(f"  Total distance: {total_distance:.6f}m")
    print(f"  Total time: {total_time:.2f}s ({len(positions)} steps)")
    print(f"  Average velocity: {average_velocity:.6f} m/s")
    
    # Test 2: Compare to baseline expectation
    print(f"\n⚖️  COMPARISON TO BASELINE:")
    print("-" * 40)
    baseline_velocity = 0.214  # Known baseline performance
    print(f"  Expected baseline velocity: {baseline_velocity:.3f} m/s")
    print(f"  Measured velocity: {average_velocity:.3f} m/s")
    
    velocity_ratio = average_velocity / baseline_velocity
    print(f"  Performance ratio: {velocity_ratio:.3f} ({velocity_ratio*100:.1f}%)")
    
    if velocity_ratio > 0.9:
        print("  ✅ EXCELLENT: Within 10% of baseline!")
    elif velocity_ratio > 0.7:
        print("  ✅ GOOD: Within 30% of baseline")
    elif velocity_ratio > 0.5:
        print("  ⚠️  MODERATE: Significant slowdown")
    else:
        print("  ❌ POOR: Major performance loss")
    
    # Test 3: Instantaneous velocity check
    print(f"\n📈 INSTANTANEOUS VELOCITY ANALYSIS:")
    print("-" * 40)
    
    # Calculate velocities between consecutive steps
    instantaneous_velocities = []
    for i in range(1, len(positions)):
        dt = timestamps[i] - timestamps[i-1]
        dx = positions[i] - positions[i-1]
        vel = dx / dt
        instantaneous_velocities.append(vel)
    
    avg_inst_vel = np.mean(instantaneous_velocities)
    std_inst_vel = np.std(instantaneous_velocities)
    max_inst_vel = np.max(instantaneous_velocities)
    min_inst_vel = np.min(instantaneous_velocities)
    
    print(f"  Average instantaneous velocity: {avg_inst_vel:.6f} m/s")
    print(f"  Velocity std deviation: {std_inst_vel:.6f} m/s")
    print(f"  Max instantaneous velocity: {max_inst_vel:.6f} m/s")
    print(f"  Min instantaneous velocity: {min_inst_vel:.6f} m/s")
    
    # Test 4: Check for target reaching behavior
    print(f"\n🎯 TARGET BEHAVIOR VERIFICATION:")
    print("-" * 40)
    
    obs = env.reset()
    targets_reached = 0
    target_steps = []
    
    print("Checking target reaching in fresh episode...")
    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        if info[0] and 'success_bonus' in info[0] and info[0]['success_bonus'] > 0:
            targets_reached += 1
            target_steps.append(step)
            current_x = env.envs[0].unwrapped.data.qpos[0]
            print(f"  🎯 Target {targets_reached} reached at step {step} (x={current_x:.1f}m)")
            
            if targets_reached >= 3:  # Stop after 3 targets
                break
        
        if done[0]:
            break
    
    print(f"\n📊 TARGET PERFORMANCE:")
    print(f"  Targets reached: {targets_reached}")
    if target_steps:
        avg_steps_per_target = np.mean(np.diff([0] + target_steps))
        print(f"  Average steps per target: {avg_steps_per_target:.1f}")
        print(f"  Average time per target: {avg_steps_per_target * 0.05:.1f}s")
        
    # Final assessment
    print(f"\n🏆 FINAL ASSESSMENT:")
    print("=" * 60)
    
    walking_good = average_velocity > 0.15
    targets_good = targets_reached >= 2
    consistent_good = std_inst_vel < 0.1
    
    print(f"Walking Performance: {'✅ GOOD' if walking_good else '❌ POOR'} ({average_velocity:.3f} m/s)")
    print(f"Target Behavior: {'✅ GOOD' if targets_good else '❌ POOR'} ({targets_reached} targets)")
    print(f"Consistency: {'✅ GOOD' if consistent_good else '⚠️  VARIABLE'} (std: {std_inst_vel:.3f})")
    
    if walking_good and targets_good:
        print(f"\n🎉 VERIFICATION PASSED!")
        print(f"✅ Robot walks properly at {average_velocity:.3f} m/s")
        print(f"✅ Robot reaches targets consistently")
        print(f"🚀 Ready to proceed with robustness training!")
    else:
        print(f"\n❌ VERIFICATION FAILED!")
        print(f"Robot performance is not good enough for robustness training")
        print(f"Need to debug further before proceeding")
    
    env.close()
    
    return {
        'velocity': average_velocity,
        'targets': targets_reached,
        'std_velocity': std_inst_vel,
        'total_distance': total_distance,
        'passed': walking_good and targets_good
    }

if __name__ == "__main__":
    results = verify_walking_performance()
    
    if results['passed']:
        print(f"\n✅ VERIFICATION COMPLETE - ROBOT IS GOOD!")
    else:
        print(f"\n❌ ROBOT NEEDS MORE WORK BEFORE PROCEEDING")