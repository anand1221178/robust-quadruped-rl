#!/usr/bin/env python3
"""
Test DR models with PROPER curriculum phases they were trained on!
Bug fix: Don't reset curriculum wrapper - test at the phases they experienced
"""
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
from envs.domain_randomization_wrapper import CurriculumDRWrapper
import realant_sim
import os
import argparse
import yaml

def test_dr_model_proper_phases(model_folder, episodes_per_phase=3):
    """Test DR model at the actual phases it was trained on"""
    
    model_name = os.path.basename(model_folder)
    print(f"🔬 PROPER PHASE TESTING: {model_name.upper()}")
    print("🎯 Testing at phases the model actually experienced during training")
    print("=" * 80)
    
    # Load paths
    model_path = os.path.join(model_folder, 'final_model.zip')
    vec_normalize_path = os.path.join(model_folder, 'vec_normalize.pkl')
    config_path = os.path.join(model_folder, 'config.yaml')
    
    if not all(os.path.exists(p) for p in [model_path, vec_normalize_path, config_path]):
        print("❌ Missing required files!")
        return
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dr_config = config.get('domain_randomization', {})
    total_steps = config.get('total_timesteps', 25000000)
    
    # Extract phase info
    phase_1_steps = dr_config.get('phase_1_steps', 8000000)
    phase_2_steps = dr_config.get('phase_2_steps', 8000000)
    
    phase_1_config = dr_config.get('phase_1_config', {})
    phase_2_config = dr_config.get('phase_2_config', {})
    phase_3_config = dr_config.get('phase_3_config', {})
    
    print(f"📊 MODEL TRAINING HISTORY:")
    print(f"  Total training: {total_steps:,} steps")
    print(f"  Phase 1 (0-{phase_1_steps:,}): {phase_1_config}")
    print(f"  Phase 2 ({phase_1_steps:,}-{phase_1_steps + phase_2_steps:,}): {phase_2_config}")
    print(f"  Phase 3 ({phase_1_steps + phase_2_steps:,}+): {phase_3_config}")
    
    # Load model
    model = PPO.load(model_path)
    print("✅ Model loaded")
    
    # Test each phase the model experienced
    phases_to_test = []
    
    if total_steps > 0:
        phases_to_test.append(("Phase 1 (Clean)", phase_1_config, "🧹"))
    if total_steps > phase_1_steps:
        phases_to_test.append(("Phase 2 (Mild Failures)", phase_2_config, "⚡"))
    if total_steps > phase_1_steps + phase_2_steps:
        phases_to_test.append(("Phase 3 (Hard Failures)", phase_3_config, "🔥"))
    
    all_results = {}
    
    for phase_name, phase_config, emoji in phases_to_test:
        print(f"\n{emoji} TESTING {phase_name.upper()}")
        print(f"Config: {phase_config}")
        print("-" * 60)
        
        # Create environment for this phase
        env = gym.make('RealAntMujoco-v0')
        env = SuccessRewardWrapper(env)
        
        # Apply the specific phase config manually
        if phase_config.get('joint_dropout_prob', 0) > 0:
            # Create a simple DR wrapper with this phase's settings
            from envs.domain_randomization_wrapper import DomainRandomizationWrapper
            env = DomainRandomizationWrapper(env, phase_config)
            print(f"✅ DR applied: {phase_config.get('joint_dropout_prob', 0)*100:.1f}% failure rate")
        else:
            print("✅ Clean environment (no failures)")
        
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
        
        # Test episodes
        episode_results = []
        joint_failure_counts = []
        
        for episode in range(episodes_per_phase):
            obs = env.reset()
            total_reward = 0
            positions = []
            joint_failures_this_episode = 0
            
            # Get initial position
            if hasattr(env, 'get_original_obs'):
                initial_pos = env.get_original_obs()[0][0]
            else:
                initial_pos = 0.0
            positions.append(initial_pos)
            
            for step in range(1000):  # 50 seconds
                # Check for joint failures
                if hasattr(env.envs[0], 'current_dropped_joints'):
                    if env.envs[0].current_dropped_joints:
                        joint_failures_this_episode += 1
                
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_reward += reward[0]
                
                # Track position
                if hasattr(env, 'get_original_obs'):
                    current_pos = env.get_original_obs()[0][0]
                    positions.append(current_pos)
                
                if done.any():
                    break
            
            # Calculate metrics
            distance = positions[-1] - positions[0] if len(positions) > 1 else 0
            time_taken = len(positions) * 0.05
            velocity = distance / time_taken if time_taken > 0 else 0
            
            episode_results.append({
                'velocity': velocity,
                'distance': distance,
                'reward': total_reward,
                'steps': len(positions)
            })
            joint_failure_counts.append(joint_failures_this_episode)
            
            print(f"  Episode {episode+1}: {velocity:.3f} m/s, {total_reward:.0f} reward, {joint_failures_this_episode} failures")
        
        # Phase summary
        avg_velocity = np.mean([r['velocity'] for r in episode_results])
        avg_reward = np.mean([r['reward'] for r in episode_results])
        avg_failures = np.mean(joint_failure_counts)
        
        all_results[phase_name] = {
            'velocity': avg_velocity,
            'reward': avg_reward,
            'failures': avg_failures,
            'config': phase_config
        }
        
        print(f"\n📊 {phase_name} SUMMARY:")
        print(f"  Average velocity: {avg_velocity:.3f} m/s")
        print(f"  Average reward: {avg_reward:.0f}")
        print(f"  Average failures per episode: {avg_failures:.1f}")
        
        # Status assessment
        if avg_velocity > 0.15:
            status = f"{emoji} EXCELLENT"
        elif avg_velocity > 0.10:
            status = f"{emoji} GOOD"
        elif avg_velocity > 0.05:
            status = f"{emoji} MODERATE"
        else:
            status = f"{emoji} POOR"
        
        print(f"  Status: {status}")
        
        env.close()
    
    # Final comparison
    print(f"\n🏆 COMPLETE MODEL ANALYSIS: {model_name}")
    print("=" * 80)
    
    baseline_velocity = 0.224  # Known baseline
    
    for phase_name, results in all_results.items():
        retention = (results['velocity'] / baseline_velocity) * 100
        print(f"{phase_name:20}: {results['velocity']:.3f} m/s ({retention:.1f}% retention)")
    
    # Overall assessment
    if len(all_results) >= 2:
        clean_vel = list(all_results.values())[0]['velocity']
        failure_vel = list(all_results.values())[-1]['velocity']
        
        if failure_vel > 0.15:
            print(f"\n🎉 MODEL ASSESSMENT: EXCELLENT ROBUSTNESS!")
            print(f"   Maintains {failure_vel:.3f} m/s even with failures!")
        elif failure_vel > 0.10:
            print(f"\n✅ MODEL ASSESSMENT: GOOD ROBUSTNESS")
            print(f"   Decent performance under failures: {failure_vel:.3f} m/s")
        elif failure_vel > 0.05:
            print(f"\n⚠️  MODEL ASSESSMENT: MODERATE ROBUSTNESS")
            print(f"   Some degradation under failures: {failure_vel:.3f} m/s")
        else:
            print(f"\n❌ MODEL ASSESSMENT: POOR ROBUSTNESS")
            print(f"   Significant degradation: {failure_vel:.3f} m/s")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description='Test DR models with proper phases')
    parser.add_argument('model_folder', help='Path to DR model folder')
    parser.add_argument('--episodes', type=int, default=3, help='Episodes per phase (default: 3)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_folder):
        print(f"❌ Model folder not found: {args.model_folder}")
        return
    
    results = test_dr_model_proper_phases(args.model_folder, args.episodes)
    
    if results:
        print(f"\n✅ Proper phase testing complete!")
        print(f"🔍 This reveals the model's TRUE robustness capabilities!")
    else:
        print(f"❌ Testing failed")

if __name__ == "__main__":
    main()