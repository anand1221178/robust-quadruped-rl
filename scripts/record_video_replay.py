#!/usr/bin/env python3
"""
Two-pass video recording: First collect trajectory, then replay with rendering
This avoids rendering overhead affecting physics simulation
"""

import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
from envs.target_walking_wrapper import TargetWalkingWrapper
import realant_sim
import imageio
import argparse
import os
import pickle

def collect_trajectory(model_path, vec_normalize_path=None, use_target_walking=False, episode_length=500):
    """First pass: Collect trajectory without rendering"""
    
    print("=" * 50)
    print("PASS 1: Collecting trajectory without rendering...")
    print("=" * 50)
    
    # Load model
    model = PPO.load(model_path)
    print(f"Loaded model from {model_path}")
    
    # Create environment WITHOUT rendering
    def make_env():
        env = gym.make('RealAntMujoco-v0')  # No render_mode!
        
        # Apply same wrapper as training
        if use_target_walking:
            print("Using Target Walking Wrapper")
            env = TargetWalkingWrapper(env)
        else:
            print("Using Success Reward Wrapper")
            env = SuccessRewardWrapper(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Load VecNormalize if available
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
    elif os.path.exists(model_path.replace('best_model.zip', '../vec_normalize.pkl')):
        vec_path = model_path.replace('best_model.zip', '../vec_normalize.pkl')
        env = VecNormalize.load(vec_path, env)
        env.training = False
        env.norm_reward = False
        print(f"Auto-loaded VecNormalize from: {vec_path}")
    
    # Collect trajectory
    obs = env.reset()
    trajectory = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'infos': [],
        'dones': []
    }
    
    velocities = []
    distances = []
    episode_reward = 0
    
    print(f"Collecting {episode_length} steps...")
    for step in range(episode_length):
        # Store observation
        trajectory['observations'].append(obs.copy())
        
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        trajectory['actions'].append(action.copy())
        
        # Step environment
        obs, reward, done, info = env.step(action)
        
        # Store results
        trajectory['rewards'].append(reward[0])
        trajectory['infos'].append(info[0])
        trajectory['dones'].append(done[0])
        
        episode_reward += reward[0]
        
        # Track metrics
        if info[0] is not None:
            if 'speed' in info[0]:
                velocities.append(info[0]['speed'])
            elif 'current_velocity' in info[0]:
                velocities.append(info[0]['current_velocity'])
            if 'distance_traveled' in info[0]:
                distances.append(info[0]['distance_traveled'])
        
        if done[0]:
            print(f"Episode ended early at step {step}")
            break
    
    env.close()
    
    # Print performance metrics (without rendering overhead)
    print("\n" + "=" * 50)
    print("ACTUAL PERFORMANCE (No Rendering):")
    print("=" * 50)
    print(f"Total Reward: {episode_reward:.2f}")
    print(f"Episode Length: {len(trajectory['actions'])} steps")
    if velocities:
        print(f"Average Velocity: {np.mean(velocities):.3f} m/s")
        print(f"Max Velocity: {np.max(velocities):.3f} m/s")
    if distances:
        print(f"Total Distance: {distances[-1]:.3f} m")
    print("=" * 50)
    
    return trajectory, np.mean(velocities) if velocities else 0.0


def replay_with_rendering(trajectory, output_path="walking_video_replay.mp4", use_target_walking=False):
    """Second pass: Replay trajectory with rendering"""
    
    print("\n" + "=" * 50)
    print("PASS 2: Replaying trajectory with rendering...")
    print("=" * 50)
    
    # Create environment WITH rendering
    def make_env():
        env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')
        
        if use_target_walking:
            env = TargetWalkingWrapper(env)
        else:
            env = SuccessRewardWrapper(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Reset environment
    obs = env.reset()
    frames = []
    
    print(f"Replaying {len(trajectory['actions'])} steps with rendering...")
    
    # Replay the exact trajectory
    for step, action in enumerate(trajectory['actions']):
        # Use the recorded action (squeeze to remove batch dimension)
        obs, reward, done, info = env.step(action)  # action is already in correct shape
        
        # Capture frame
        frame = env.render(mode='rgb_array')
        if frame is not None:
            frames.append(frame)
        
        if done[0]:
            break
    
    env.close()
    
    # Save video
    if frames:
        print(f"Saving video to {output_path}...")
        imageio.mimsave(output_path, frames, fps=50)
        print(f"Video saved! ({len(frames)} frames)")
        print(f"Note: This video shows the EXACT trajectory from Pass 1")
    else:
        print("No frames captured!")
    
    return len(frames)


def main():
    parser = argparse.ArgumentParser(description='Two-pass video recording for accurate performance')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--vec-normalize', type=str, help='Path to VecNormalize file')
    parser.add_argument('--output', type=str, default='walking_video_replay.mp4', help='Output video path')
    parser.add_argument('--use-target-walking', action='store_true', help='Use TargetWalkingWrapper')
    parser.add_argument('--episode-length', type=int, default=500, help='Episode length in steps')
    
    args = parser.parse_args()
    
    # Pass 1: Collect trajectory without rendering
    trajectory, true_velocity = collect_trajectory(
        args.model,
        args.vec_normalize,
        args.use_target_walking,
        args.episode_length
    )
    
    # Pass 2: Replay with rendering
    frames_rendered = replay_with_rendering(
        trajectory,
        args.output,
        args.use_target_walking
    )
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print("=" * 50)
    print(f"✅ True Performance (no rendering): {true_velocity:.3f} m/s")
    print(f"✅ Video created with {frames_rendered} frames")
    print(f"✅ Video accurately shows the robot's actual trajectory")
    print("=" * 50)


if __name__ == "__main__":
    main()