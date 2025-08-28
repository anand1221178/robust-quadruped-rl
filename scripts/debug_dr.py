#!/usr/bin/env python3
"""
Debug script for Domain Randomization - shows exactly what failures are being injected
"""

import os
import sys
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import argparse
import time

# Import RealAnt environments to register them
import realant_sim

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from envs.domain_randomization_wrapper import DomainRandomizationWrapper, CurriculumDRWrapper
from envs.target_walking_wrapper import TargetWalkingWrapper

# Joint names for RealAnt
JOINT_NAMES = [
    'front_left_hip', 'front_left_knee',
    'front_right_hip', 'front_right_knee',
    'back_left_hip', 'back_left_knee',
    'back_right_hip', 'back_right_knee'
]

def create_debug_env(use_dr=True, dr_config=None):
    """Create environment with optional DR wrapper for debugging"""
    # Don't use render_mode here to avoid compatibility issues
    env = gym.make('RealAntMujoco-v0')
    env = TargetWalkingWrapper(env, target_distance=5.0)
    
    if use_dr and dr_config:
        print("\n" + "="*60)
        print("DOMAIN RANDOMIZATION CONFIGURATION")
        print("="*60)
        print(f"Joint dropout probability: {dr_config.get('joint_dropout_prob', 0)}%")
        print(f"Max dropped joints: {dr_config.get('max_dropped_joints', 0)}")
        print(f"Sensor noise std: {dr_config.get('sensor_noise_std', 0)}")
        print("="*60 + "\n")
        env = DomainRandomizationWrapper(env, dr_config)
    
    env = Monitor(env)
    return env

def test_model_with_dr(model_path, num_episodes=5):
    """Test model with detailed DR information display"""
    
    print(f"\nTesting model: {model_path}")
    
    # Load model
    model = PPO.load(model_path)
    
    # Try to find vec_normalize
    exp_dir = os.path.dirname(os.path.dirname(model_path))
    vec_norm_path = os.path.join(exp_dir, 'vec_normalize.pkl')
    
    # Create environments with different DR settings
    test_configs = [
        {
            'name': 'NO DOMAIN RANDOMIZATION (baseline)',
            'use_dr': False,
            'config': None
        },
        {
            'name': 'PHASE 2 DR (Single joint + mild noise)',
            'use_dr': True,
            'config': {
                'joint_dropout_prob': 0.2,
                'max_dropped_joints': 1,
                'min_dropped_joints': 1,
                'sensor_noise_std': 0.02,
                'noise_joints_only': True
            }
        },
        {
            'name': 'PHASE 3 DR (Multiple joints + high noise)',
            'use_dr': True,
            'config': {
                'joint_dropout_prob': 0.4,
                'max_dropped_joints': 3,
                'min_dropped_joints': 1,
                'sensor_noise_std': 0.05,
                'noise_joints_only': True
            }
        }
    ]
    
    for test_config in test_configs:
        print("\n" + "="*80)
        print(f"TESTING WITH: {test_config['name']}")
        print("="*80)
        
        # Create environment
        env = create_debug_env(
            use_dr=test_config['use_dr'],
            dr_config=test_config['config']
        )
        
        # Wrap with VecNormalize if available
        env = DummyVecEnv([lambda: env])
        if os.path.exists(vec_norm_path):
            env = VecNormalize.load(vec_norm_path, env)
            env.training = False
            env.norm_reward = False
            print("✅ Loaded normalization stats")
        
        # Run episodes
        episode_results = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            done = False
            total_reward = 0
            steps = 0
            velocities = []
            
            print(f"\n--- Episode {episode + 1} ---")
            
            # Get DR info if available
            if test_config['use_dr'] and hasattr(env.envs[0].env.env, 'dropped_joints'):
                dropped_joints = env.envs[0].env.env.dropped_joints
                if dropped_joints:
                    joint_names = [JOINT_NAMES[j] for j in dropped_joints]
                    print(f"🔴 DROPPED JOINTS: {joint_names} (indices: {dropped_joints})")
                else:
                    print("🟢 NO JOINTS DROPPED THIS EPISODE")
                
                sensor_noise = env.envs[0].env.env.sensor_noise_std
                print(f"📊 Sensor noise level: {sensor_noise:.3f}")
            
            while not done and steps < 1000:
                action, _ = model.predict(obs, deterministic=True)
                
                # Show action modifications if DR is active
                if test_config['use_dr'] and hasattr(env.envs[0].env.env, 'dropped_joints'):
                    if env.envs[0].env.env.dropped_joints and steps % 100 == 0:
                        print(f"  Step {steps}: Actions being zeroed for joints {env.envs[0].env.env.dropped_joints}")
                
                obs, reward, done, info = env.step(action)
                
                # Extract velocity info - try multiple possible keys
                if 'x_velocity' in info[0]:
                    velocities.append(info[0]['x_velocity'])
                elif 'velocity' in info[0]:
                    velocities.append(info[0]['velocity'])
                elif hasattr(env.envs[0], 'env') and hasattr(env.envs[0].env, 'env'):
                    # Try to get velocity from the underlying environment
                    try:
                        base_env = env.envs[0].env
                        while hasattr(base_env, 'env'):
                            base_env = base_env.env
                        if hasattr(base_env, 'sim'):
                            vel = base_env.sim.data.qvel[0]  # x-velocity
                            velocities.append(vel)
                    except:
                        pass
                
                total_reward += reward[0]
                steps += 1
                
                # Skip render to avoid compatibility issues
                # env.envs[0].env.render()
                # time.sleep(0.01)
            
            # Episode summary
            avg_velocity = np.mean(velocities) if velocities else 0
            distance = avg_velocity * steps * 0.05  # 0.05 is dt
            
            print(f"\nEpisode {episode + 1} Results:")
            print(f"  Reward: {total_reward:.2f}")
            print(f"  Steps: {steps}")
            print(f"  Avg velocity: {avg_velocity:.3f} m/s")
            print(f"  Distance: {distance:.2f} m")
            
            if 'success' in info[0]:
                print(f"  Success: {'✅' if info[0]['success'] else '❌'}")
            
            episode_results.append({
                'reward': total_reward,
                'velocity': avg_velocity,
                'distance': distance,
                'steps': steps
            })
        
        # Summary for this DR configuration
        print(f"\n{test_config['name']} SUMMARY:")
        print(f"  Avg velocity: {np.mean([r['velocity'] for r in episode_results]):.3f} ± {np.std([r['velocity'] for r in episode_results]):.3f} m/s")
        print(f"  Avg distance: {np.mean([r['distance'] for r in episode_results]):.2f} m")
        print(f"  Avg reward: {np.mean([r['reward'] for r in episode_results]):.2f}")
        
        env.close()
    
    print("\n" + "="*80)
    print("DOMAIN RANDOMIZATION DEBUG COMPLETE")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Debug Domain Randomization')
    parser.add_argument('model_path', type=str, help='Path to model')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes per config')
    
    args = parser.parse_args()
    
    test_model_with_dr(args.model_path, args.episodes)

if __name__ == "__main__":
    main()