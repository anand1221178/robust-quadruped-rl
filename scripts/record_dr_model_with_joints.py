#!/usr/bin/env python3
"""
Record video of DR model showing joint failures
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
import cv2
import os
from datetime import datetime
import argparse

def record_dr_model_with_joints(model_folder, steps=500):
    """Record DR model video showing joint states"""
    
    model_name = os.path.basename(model_folder)
    print(f"🎬 DR MODEL + JOINT ANALYSIS: {model_name.upper()}")
    print(f" Model: {model_folder}")  
    print(f"🎯 Steps: {steps}")
    print("=" * 70)
    
    # Model paths
    model_path = os.path.join(model_folder, 'final_model.zip')
    vec_normalize_path = os.path.join(model_folder, 'vec_normalize.pkl')
    config_path = os.path.join(model_folder, 'config.yaml')
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    if not os.path.exists(vec_normalize_path):
        print(f"❌ VecNormalize not found: {vec_normalize_path}")
        return
    
    # Load config to understand DR settings
    dr_config = {}
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            dr_config = config.get('domain_randomization', {})
        print(" Config loaded - DR settings found")
    
    print("📁 PASS 1: Collecting trajectory with joint failure analysis")
    print("-" * 50)
    
    # Create environment with DR
    env = gym.make('RealAntMujoco-v0')
    env = SuccessRewardWrapper(env)
    
    # Add DR wrapper if curriculum detected
    has_curriculum = any(key.startswith('phase_') for key in dr_config.keys())
    if has_curriculum:
        print("🔄 Curriculum DR detected - using CurriculumDRWrapper")
        env = CurriculumDRWrapper(env, dr_config)
    
    print(f"📏 Observation space: {env.observation_space}")
    
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(vec_normalize_path, env)
    env.training = False
    env.norm_reward = False
    print(" VecNormalize loaded")
    
    # Load model
    model = PPO.load(model_path)
    print(" Model loaded")
    
    print(f"\n🤖 Recording with joint failure tracking...")
    
    # Collect trajectory data
    trajectory = {'obs': [], 'actions': [], 'rewards': [], 'joint_states': []}
    obs = env.reset()
    trajectory['obs'].append(obs)
    
    total_reward = 0
    positions = []
    velocities = []
    joint_failure_log = []
    
    # Joint names for RealAnt
    joint_names = [
        "hip_4", "ankle_4", "hip_1", "ankle_1", 
        "hip_2", "ankle_2", "hip_3", "ankle_3"
    ]
    
    for step in range(steps):
        # Get current state
        if hasattr(env, 'get_original_obs'):
            original_obs = env.get_original_obs()[0]
            pos_x = original_obs[0]
        else:
            pos_x = 0.0
            
        positions.append(pos_x)
        
        # Check for joint failures in wrapped environment
        joint_failures = []
        if hasattr(env.envs[0], 'current_dropped_joints'):
            dropped_joints = env.envs[0].current_dropped_joints
            if dropped_joints:
                joint_failures = [joint_names[i] if i < len(joint_names) else f"joint_{i}" for i in dropped_joints]
        elif hasattr(env.envs[0].env, 'current_dropped_joints'):
            dropped_joints = env.envs[0].env.current_dropped_joints
            if dropped_joints:
                joint_failures = [joint_names[i] if i < len(joint_names) else f"joint_{i}" for i in dropped_joints]
        
        joint_failure_log.append(joint_failures)
        
        # Calculate velocity
        if len(positions) >= 10:
            recent_positions = positions[-10:]
            distance = recent_positions[-1] - recent_positions[0]
            time_taken = 9 * 0.05
            current_velocity = distance / time_taken if time_taken > 0 else 0
        else:
            current_velocity = 0
            
        velocities.append(current_velocity)
        
        # Progress output with joint info
        if step % 50 == 0:
            joint_info = f", joints_failed: {joint_failures}" if joint_failures else ", joints_ok"
            print(f"  Step {step:3d}: x={pos_x:5.2f}m, vel={current_velocity:.3f}m/s{joint_info}")
        
        # Predict and step
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        trajectory['actions'].append(action)
        trajectory['rewards'].append(reward)
        trajectory['obs'].append(obs)
        trajectory['joint_states'].append(joint_failures.copy())
        
        total_reward += reward[0] if hasattr(reward, '__len__') else reward
        
        if done.any() if hasattr(done, 'any') else done:
            print(f"Episode ended at step {step}")
            break
    
    # Calculate metrics
    final_distance = positions[-1] - positions[0] if len(positions) > 1 else 0
    total_time = len(positions) * 0.05
    avg_velocity = final_distance / total_time if total_time > 0 else 0
    
    # Analyze joint failures
    failure_count = sum(1 for failures in joint_failure_log if failures)
    failure_percentage = (failure_count / len(joint_failure_log)) * 100 if joint_failure_log else 0
    
    # Count specific joint failures
    joint_failure_counts = {}
    for failures in joint_failure_log:
        for joint in failures:
            joint_failure_counts[joint] = joint_failure_counts.get(joint, 0) + 1
    
    print(f"\n📊 TRAJECTORY ANALYSIS:")
    print(f"  Steps: {len(trajectory['obs'])}")
    print(f"  Distance: {final_distance:.3f}m")
    print(f"  Velocity: {avg_velocity:.3f} m/s")
    print(f"  Total reward: {total_reward:.0f}")
    print(f"\n🦾 JOINT FAILURE ANALYSIS:")
    print(f"  Episodes with failures: {failure_count}/{len(joint_failure_log)} ({failure_percentage:.1f}%)")
    if joint_failure_counts:
        print(f"  Joint failure counts:")
        for joint, count in sorted(joint_failure_counts.items()):
            print(f"    {joint}: {count} times")
    else:
        print(f"  No joint failures detected")
    
    # Status
    if avg_velocity < 0:
        print("🚨 ROBOT WALKS BACKWARDS!")
    elif avg_velocity < 0.05:
        print("⚠️  ROBOT BARELY MOVES!")  
    elif avg_velocity > 0.15:
        print(" Good locomotion despite failures")
    
    # Create video with joint failure overlay
    print(f"\n🎥 PASS 2: Rendering video with joint failure info")
    print("-" * 50)
    
    render_env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')
    render_env = SuccessRewardWrapper(render_env)
    
    print("🎬 Rendering video...")
    frames = []
    obs = render_env.reset()[0] if hasattr(render_env.reset(), '__len__') else render_env.reset()
    
    for i, action in enumerate(trajectory['actions']):
        if i % 100 == 0:
            print(f"  Rendered {i+1}/{len(trajectory['actions'])} frames...")
        
        # Step and render
        step_result = render_env.step(action[0] if hasattr(action, '__len__') else action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result
            
        frame = render_env.render()
        
        if frame is not None:
            frame_with_text = frame.copy()
            
            # Get current data
            step_vel = velocities[min(i, len(velocities)-1)]
            failed_joints = trajectory['joint_states'][i] if i < len(trajectory['joint_states']) else []
            
            # Add text overlay
            cv2.putText(frame_with_text, f'Step: {i+1}/{len(trajectory["actions"])}', 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame_with_text, f'Velocity: {step_vel:.3f} m/s', 
                       (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame_with_text, f'Model: {model_name}', 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show joint failures
            if failed_joints:
                joint_text = f'Failed: {", ".join(failed_joints)}'
                cv2.putText(frame_with_text, joint_text, 
                           (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
            else:
                cv2.putText(frame_with_text, 'All joints OK', 
                           (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
            
            frames.append(frame_with_text)
        
        if done:
            break
    
    render_env.close()
    
    if frames:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{model_name}_joints_video_{timestamp}.mp4'
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
        
        for frame in frames:
            out.write(frame)
        out.release()
        
        print(f"\n VIDEO WITH JOINT ANALYSIS COMPLETED!")
        print(f"📁 Saved as: {filename}")
        print(f"📊 Performance: {avg_velocity:.3f} m/s")
        print(f"🦾 Joint failures: {failure_percentage:.1f}% of episodes")
        
        return filename, {
            'velocity': avg_velocity,
            'joint_failure_rate': failure_percentage,
            'joint_failure_counts': joint_failure_counts
        }
    else:
        print("❌ No frames captured!")
        return None, None

def main():
    parser = argparse.ArgumentParser(description='Record DR model with joint analysis')
    parser.add_argument('model_folder', help='Path to model folder')
    parser.add_argument('--steps', type=int, default=500, help='Number of steps (default: 500)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_folder):
        print(f"❌ Model folder not found: {args.model_folder}")
        sys.exit(1)
    
    filename, analysis = record_dr_model_with_joints(args.model_folder, args.steps)
    
    if filename:
        print(f"\n🎬 Video ready: {filename}")
        if analysis:
            print(f"📈 Analysis: {analysis['velocity']:.3f} m/s with {analysis['joint_failure_rate']:.1f}% failure rate")
    else:
        print(f"\n❌ Video creation failed")

if __name__ == "__main__":
    main()