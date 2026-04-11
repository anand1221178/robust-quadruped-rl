#!/usr/bin/env python3
"""
Test DR models WITH ACTUAL JOINT FAILURES INJECTED!
Show how they handle the robustness they were trained for!
"""
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
from envs.domain_randomization_wrapper import DomainRandomizationWrapper
import realant_sim
import cv2
import os
from datetime import datetime
import argparse

def test_dr_with_failures(model_folder, failure_rate=0.2, steps=600):
    """Test DR model WITH joint failures to see robustness!"""
    
    model_name = os.path.basename(model_folder)
    print(f"🦾 DR ROBUSTNESS TEST: {model_name.upper()}")
    print(f"🎯 Joint failure rate: {failure_rate*100:.0f}%")
    print(f"🔥 Testing ACTUAL robustness capabilities!")
    print("=" * 70)
    
    # Model paths
    model_path = os.path.join(model_folder, 'final_model.zip')
    vec_normalize_path = os.path.join(model_folder, 'vec_normalize.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(vec_normalize_path):
        print("❌ Missing model files!")
        return
    
    print("📁 PASS 1: Collecting trajectory WITH JOINT FAILURES")
    print("-" * 50)
    
    # Create environment with DR failures
    env = gym.make('RealAntMujoco-v0')
    env = SuccessRewardWrapper(env)
    
    # Add DR with specified failure rate
    dr_config = {
        'joint_dropout_prob': failure_rate,
        'max_dropped_joints': 2,
        'min_dropped_joints': 1,  # Force at least 1 failure when triggered
        'sensor_noise_std': 0.02
    }
    
    env = DomainRandomizationWrapper(env, dr_config)
    print(f"🦾 DR configured: {failure_rate*100:.0f}% failure rate, 1-2 joints")
    
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(vec_normalize_path, env)
    env.training = False
    env.norm_reward = False
    print("✅ VecNormalize loaded")
    
    # Load model
    model = PPO.load(model_path)
    print("✅ Model loaded")
    
    print(f"\n🤖 Recording with {failure_rate*100:.0f}% joint failures...")
    
    # Collect trajectory with failure tracking
    trajectory = {'obs': [], 'actions': [], 'rewards': [], 'joint_states': []}
    obs = env.reset()
    trajectory['obs'].append(obs)
    
    total_reward = 0
    positions = []
    velocities = []
    joint_failure_log = []
    failure_episodes = 0
    
    # Joint names for display
    joint_names = ["hip_4", "ankle_4", "hip_1", "ankle_1", "hip_2", "ankle_2", "hip_3", "ankle_3"]
    
    for step in range(steps):
        # Get position
        if hasattr(env, 'get_original_obs'):
            original_obs = env.get_original_obs()[0]
            pos_x = original_obs[0]
        else:
            pos_x = 0.0
            
        positions.append(pos_x)
        
        # Check for joint failures
        joint_failures = []
        if hasattr(env.envs[0], 'current_dropped_joints'):
            dropped_joints = env.envs[0].current_dropped_joints
            if dropped_joints:
                joint_failures = [joint_names[i] if i < len(joint_names) else f"joint_{i}" for i in dropped_joints]
                failure_episodes = 1  # Mark that failures happened
        
        joint_failure_log.append(joint_failures.copy())
        
        # Calculate velocity
        if len(positions) >= 20:
            recent_positions = positions[-20:]
            distance = recent_positions[-1] - recent_positions[0]
            time_taken = 19 * 0.05
            current_velocity = distance / time_taken if time_taken > 0 else 0
        else:
            current_velocity = 0
            
        velocities.append(current_velocity)
        
        # Progress with failure info
        if step % 100 == 0:
            joint_info = f", 🦾 FAILING: {joint_failures}" if joint_failures else ", 🟢 OK"
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
    if len(positions) > 1:
        total_distance = positions[-1] - positions[0]
        total_time = len(positions) * 0.05
        avg_velocity = total_distance / total_time if total_time > 0 else 0
    else:
        avg_velocity = 0
        total_distance = 0
    
    # Analyze failures
    failure_count = sum(1 for failures in joint_failure_log if failures)
    failure_percentage = (failure_count / len(joint_failure_log)) * 100 if joint_failure_log else 0
    
    print(f"\n📊 ROBUSTNESS ANALYSIS:")
    print(f"  Total distance: {total_distance:.3f}m")
    print(f"  Average velocity: {avg_velocity:.3f} m/s")
    print(f"  Total reward: {total_reward:.0f}")
    print(f"  Steps with failures: {failure_count}/{len(joint_failure_log)} ({failure_percentage:.1f}%)")
    
    if failure_count > 0:
        if avg_velocity > 0.15:
            print("🔥 EXCELLENT ROBUSTNESS - Walks well despite failures!")
        elif avg_velocity > 0.10:
            print("✅ GOOD ROBUSTNESS - Maintains locomotion with failures")
        elif avg_velocity > 0.05:
            print("⚠️ MODERATE ROBUSTNESS - Some degradation with failures")
        else:
            print("❌ POOR ROBUSTNESS - Significant impact from failures")
    else:
        print("⚠️ NO FAILURES OCCURRED - Need to test robustness!")
    
    # Create video with failure visualization
    print(f"\n🎥 PASS 2: Rendering video with failure visualization")
    print("-" * 50)
    
    render_env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')
    render_env = SuccessRewardWrapper(render_env)
    render_env = DomainRandomizationWrapper(render_env, dr_config)
    
    print("🎬 Rendering with failure indicators...")
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
            
            # Add performance overlay
            cv2.putText(frame_with_text, f'Step: {i+1}/{len(trajectory["actions"])}', 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_with_text, f'Velocity: {step_vel:.3f} m/s', 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_with_text, f'Failure Rate: {failure_rate*100:.0f}%', 
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show current joint failures
            if failed_joints:
                joint_text = f'FAILING: {", ".join(failed_joints)}'
                cv2.putText(frame_with_text, joint_text, 
                           (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2)
                # Add warning indicator
                cv2.rectangle(frame_with_text, (10, 10), (50, 50), (0, 0, 255), 3)
                cv2.putText(frame_with_text, '!', (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            else:
                cv2.putText(frame_with_text, 'All joints OK', 
                           (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)
                cv2.rectangle(frame_with_text, (10, 10), (50, 50), (0, 255, 0), 3)
                cv2.putText(frame_with_text, '✓', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            frames.append(frame_with_text)
        
        if done:
            break
    
    render_env.close()
    
    if frames:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{model_name}_robustness_{failure_rate*100:.0f}percent_{timestamp}.mp4'
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
        
        for frame in frames:
            out.write(frame)
        out.release()
        
        print(f"\n🎉 ROBUSTNESS VIDEO COMPLETED!")
        print(f"📁 Saved as: {filename}")
        print(f"🦾 Shows performance with {failure_rate*100:.0f}% joint failures")
        print(f"📊 Average velocity: {avg_velocity:.3f} m/s")
        print(f"🔥 Failure rate achieved: {failure_percentage:.1f}%")
        
        return filename, avg_velocity, failure_percentage
    else:
        print("❌ No frames captured!")
        return None, 0, 0

def main():
    parser = argparse.ArgumentParser(description='Test DR models with joint failures')
    parser.add_argument('model_folder', help='Path to DR model folder')
    parser.add_argument('--failure_rate', type=float, default=0.2, help='Joint failure rate (default: 0.2 = 20%)')
    parser.add_argument('--steps', type=int, default=600, help='Number of steps (default: 600)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_folder):
        print(f"❌ Model folder not found: {args.model_folder}")
        return
    
    filename, velocity, failure_rate = test_dr_with_failures(args.model_folder, args.failure_rate, args.steps)
    
    if filename:
        print(f"\n🎬 Robustness test complete: {filename}")
        print(f"🎯 Result: {velocity:.3f} m/s with {failure_rate:.1f}% failures")
        
        if velocity > 0.15:
            print("🔥 EXCELLENT robustness demonstrated!")
        elif velocity > 0.10:
            print("✅ GOOD robustness shown")
        else:
            print("⚠️ Robustness needs improvement")
    else:
        print("❌ Test failed")

if __name__ == "__main__":
    main()