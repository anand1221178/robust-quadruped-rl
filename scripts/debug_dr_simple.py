#!/usr/bin/env python3
"""
Simple DR debug - just shows what joints are being dropped each episode
"""

import os
import sys
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import argparse

# Import RealAnt environments
import realant_sim

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from envs.domain_randomization_wrapper import DomainRandomizationWrapper
from envs.target_walking_wrapper import TargetWalkingWrapper

# Joint names for RealAnt
JOINT_NAMES = [
    'front_left_hip', 'front_left_knee',
    'front_right_hip', 'front_right_knee', 
    'back_left_hip', 'back_left_knee',
    'back_right_hip', 'back_right_knee'
]

def test_dr_episodes(num_episodes=10):
    """Test what DR does across multiple episodes"""
    
    print("\n" + "="*60)
    print("DOMAIN RANDOMIZATION EPISODE SAMPLING TEST")
    print("="*60)
    
    # Phase 2 config (what your model is training with now)
    dr_config = {
        'joint_dropout_prob': 0.2,  # 20% chance
        'max_dropped_joints': 1,
        'min_dropped_joints': 1,
        'sensor_noise_std': 0.02,
        'noise_joints_only': True
    }
    
    print(f"\nConfiguration:")
    print(f"  Joint dropout probability: {dr_config['joint_dropout_prob']*100}%")
    print(f"  Max joints to drop: {dr_config['max_dropped_joints']}")
    print(f"  Sensor noise std: {dr_config['sensor_noise_std']}")
    print(f"\nSampling {num_episodes} episodes to see joint dropout pattern:\n")
    
    # Create env with DR
    env = gym.make('RealAntMujoco-v0')
    env = TargetWalkingWrapper(env, target_distance=5.0)
    env = DomainRandomizationWrapper(env, dr_config)
    
    episode_stats = {
        'no_dropout': 0,
        'single_dropout': 0,
        'joint_counts': {}
    }
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        
        if env.dropped_joints:
            joint_names = [JOINT_NAMES[j] for j in env.dropped_joints]
            print(f"Episode {episode+1:2d}: 🔴 DROPPED {joint_names} (indices: {env.dropped_joints})")
            episode_stats['single_dropout'] += 1
            
            for joint in env.dropped_joints:
                joint_name = JOINT_NAMES[joint]
                episode_stats['joint_counts'][joint_name] = episode_stats['joint_counts'].get(joint_name, 0) + 1
        else:
            print(f"Episode {episode+1:2d}: 🟢 No joints dropped")
            episode_stats['no_dropout'] += 1
    
    # Summary statistics
    print("\n" + "-"*60)
    print("STATISTICS:")
    print(f"  Episodes with NO dropout: {episode_stats['no_dropout']}/{num_episodes} ({episode_stats['no_dropout']/num_episodes*100:.1f}%)")
    print(f"  Episodes with dropout: {episode_stats['single_dropout']}/{num_episodes} ({episode_stats['single_dropout']/num_episodes*100:.1f}%)")
    
    if episode_stats['joint_counts']:
        print(f"\nJoint dropout frequency:")
        for joint_name, count in sorted(episode_stats['joint_counts'].items()):
            print(f"    {joint_name}: {count} times")
    
    print("="*60)
    
    env.close()

def test_model_performance(model_path):
    """Test how model performs with and without DR"""
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE TEST")
    print("="*60)
    
    # Load model
    model = PPO.load(model_path)
    
    # Find vec_normalize
    exp_dir = os.path.dirname(os.path.dirname(model_path))
    vec_norm_path = os.path.join(exp_dir, 'vec_normalize.pkl')
    
    configs = [
        {'name': 'No DR', 'use_dr': False, 'config': None},
        {'name': 'With DR (Phase 2)', 'use_dr': True, 'config': {
            'joint_dropout_prob': 0.2,
            'max_dropped_joints': 1,
            'min_dropped_joints': 1,
            'sensor_noise_std': 0.02,
            'noise_joints_only': True
        }}
    ]
    
    for test_config in configs:
        print(f"\nTesting: {test_config['name']}")
        print("-"*40)
        
        # Create env
        env = gym.make('RealAntMujoco-v0')
        env = TargetWalkingWrapper(env, target_distance=5.0)
        
        if test_config['use_dr']:
            env = DomainRandomizationWrapper(env, test_config['config'])
        
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        
        if os.path.exists(vec_norm_path):
            env = VecNormalize.load(vec_norm_path, env)
            env.training = False
            env.norm_reward = False
        
        # Run 5 episodes
        rewards = []
        for episode in range(5):
            obs = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            # Show DR status
            if test_config['use_dr'] and hasattr(env.envs[0].env.env, 'dropped_joints'):
                dropped = env.envs[0].env.env.dropped_joints
                if dropped:
                    joint_names = [JOINT_NAMES[j] for j in dropped]
                    print(f"  Episode {episode+1}: Dropped {joint_names}")
                else:
                    print(f"  Episode {episode+1}: No dropout")
            
            while not done and steps < 1000:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                total_reward += reward[0]
                steps += 1
            
            rewards.append(total_reward)
            print(f"    → Reward: {total_reward:.2f}")
        
        print(f"  Average: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        
        env.close()
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Debug Domain Randomization')
    parser.add_argument('--model', type=str, default=None, help='Path to model to test')
    parser.add_argument('--episodes', type=int, default=20, help='Number of episodes to sample')
    
    args = parser.parse_args()
    
    # First show DR sampling pattern
    test_dr_episodes(args.episodes)
    
    # Then test model if provided
    if args.model:
        test_model_performance(args.model)

if __name__ == "__main__":
    main()