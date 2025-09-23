#!/usr/bin/env python3
"""
Test V7.7e Ultra Speed with EXTENDED episodes to see if rotation completes
Removes the 1000 step limit to allow full rotation behavior
"""
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
import realant_sim

def test_ankle4_rotation():
    """Test if V7.7e completes rotation with ankle_4 failure given more time"""

    # Load model
    model_path = "/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/done/dr/Curr best/v7_7e_ultra_speed_jtfwl2qf/final_model.zip"
    vec_path = "/Users/anandpatel/Documents/4th Year/robust-quadruped-rl/done/dr/Curr best/v7_7e_ultra_speed_jtfwl2qf/vec_normalize.pkl"

    # Create environment WITHOUT time limit wrapper
    def make_env():
        env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')
        # Remove time limit wrapper if it exists
        if hasattr(env, '_max_episode_steps'):
            env._max_episode_steps = 3000  # Set to 3000 steps (60 seconds)
        if hasattr(env, 'spec') and hasattr(env.spec, 'max_episode_steps'):
            env.spec.max_episode_steps = 3000
        env = SuccessRewardWrapper(env)
        return env

    env = DummyVecEnv([make_env])

    # Load normalization
    try:
        env = VecNormalize.load(vec_path, env)
        env.training = False
        env.norm_reward = False
        print("VecNormalize loaded")
    except:
        print("No VecNormalize found")

    model = PPO.load(model_path)
    print("Model loaded successfully")

    # Test with ankle_4 locked
    print("\n" + "="*60)
    print("TESTING ANKLE_4 FAILURE WITH EXTENDED TIME")
    print("="*60)

    obs = env.reset()
    positions = []
    rotations = []  # Track yaw angle

    ankle_4_idx = 7  # ankle_4 action index

    for step in range(3000):  # 60 seconds at 50Hz
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)

        # Lock ankle_4
        action[0][ankle_4_idx] = 0.0

        # Step environment
        obs, reward, done, info = env.step(action)

        # Track position and rotation
        x_pos = env.envs[0].unwrapped.data.qpos[0]
        positions.append(x_pos)

        # Get robot orientation (yaw angle)
        if hasattr(env.envs[0].unwrapped.data, 'qpos'):
            # Orientation quaternion is typically at indices 3-6
            # But for simplicity, let's track x velocity direction change
            if len(positions) > 10:
                recent_movement = positions[-1] - positions[-10]
                rotations.append(recent_movement)

        # Print progress every 500 steps (10 seconds)
        if step % 500 == 0 and step > 0:
            distance = positions[-1] - positions[0]
            velocity = distance / (step * 0.02)  # 50Hz timestep
            print(f"  Step {step}: Distance={distance:.2f}m, Velocity={velocity:.3f} m/s")

            # Check for rotation behavior
            if len(rotations) > 20:
                rotation_change = np.std(rotations[-20:])
                if rotation_change > 0.1:
                    print(f"    🔄 ROTATION DETECTED! Variation={rotation_change:.3f}")

        if done[0]:
            print(f"Episode ended at step {step}")
            break

    # Final analysis
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    total_distance = positions[-1] - positions[0]
    total_time = len(positions) * 0.02
    avg_velocity = total_distance / total_time

    print(f"Total Distance: {total_distance:.2f}m")
    print(f"Total Time: {total_time:.1f}s")
    print(f"Average Velocity: {avg_velocity:.3f} m/s")

    # Check if performance improved over time
    if len(positions) > 1000:
        early_performance = (positions[500] - positions[0]) / 10.0
        late_performance = (positions[-1] - positions[-500]) / 10.0

        print(f"\nPerformance Analysis:")
        print(f"  First 10 seconds: {early_performance:.3f} m/s")
        print(f"  Last 10 seconds: {late_performance:.3f} m/s")

        if late_performance > early_performance * 1.2:
            print("  ✅ PERFORMANCE IMPROVED! Rotation likely helped!")
        elif late_performance < early_performance * 0.8:
            print("  ❌ Performance degraded over time")
        else:
            print("  ⚠️ Performance remained similar")

    env.close()

if __name__ == "__main__":
    test_ankle4_rotation()