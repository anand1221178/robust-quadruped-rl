#!/usr/bin/env python3
"""
Test DR model with SPECIFIC joint failure to see robustness!
Simple and direct - force a specific joint to fail and see what happens
"""
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
import realant_sim
import cv2
import os
from datetime import datetime
import argparse

class SimpleJointFailureWrapper(gym.Wrapper):
    """Simple wrapper that fails specific joints"""
    
    def __init__(self, env, failed_joints):
        super().__init__(env)
        self.failed_joints = failed_joints
        self.joint_names = ["hip_4", "ankle_4", "hip_1", "ankle_1", "hip_2", "ankle_2", "hip_3", "ankle_3"]
        print(f"🦾 FORCING JOINT FAILURES: {[self.joint_names[i] for i in failed_joints]}")
    
    def step(self, action):
        # Force failed joints to 0 (locked)
        modified_action = action.copy()
        for joint_idx in self.failed_joints:
            modified_action[joint_idx] = 0.0
        
        return self.env.step(modified_action)

def test_specific_joint_failure(model_folder, failed_joints, steps=600):
    """Test model with specific joints failed"""
    
    model_name = os.path.basename(model_folder)
    joint_names = ["hip_4", "ankle_4", "hip_1", "ankle_1", "hip_2", "ankle_2", "hip_3", "ankle_3"]
    failed_joint_names = [joint_names[i] for i in failed_joints]
    
    print(f"🦾 JOINT FAILURE TEST: {model_name.upper()}")
    print(f"❌ FAILING JOINTS: {failed_joint_names}")
    print(f"🎯 Steps: {steps}")
    print("=" * 70)
    
    # Model paths
    model_path = os.path.join(model_folder, 'final_model.zip')
    vec_normalize_path = os.path.join(model_folder, 'vec_normalize.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(vec_normalize_path):
        print("❌ Missing model files!")
        return
    
    print("📁 PASS 1: Testing with joint failures")
    print("-" * 50)
    
    # Create environment
    env = gym.make('RealAntMujoco-v0')
    env = SuccessRewardWrapper(env)
    env = SimpleJointFailureWrapper(env, failed_joints)
    
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(vec_normalize_path, env)
    env.training = False
    env.norm_reward = False
    print("✅ Environment ready")
    
    # Load model
    model = PPO.load(model_path)
    print("✅ Model loaded")
    
    print(f"\n🤖 Testing robustness with failed joints...")
    
    # Test episode
    obs = env.reset()
    total_reward = 0
    positions = []
    
    for step in range(steps):
        # Get position
        if hasattr(env, 'get_original_obs'):
            original_obs = env.get_original_obs()[0]
            pos_x = original_obs[0]
        else:
            pos_x = 0.0
            
        positions.append(pos_x)
        
        # Progress
        if step % 100 == 0:
            distance = pos_x - positions[0] if positions else 0
            print(f"  Step {step:3d}: x={pos_x:5.2f}m, distance={distance:.3f}m")
        
        # Step
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0] if hasattr(reward, '__len__') else reward
        
        if done.any() if hasattr(done, 'any') else done:
            print(f"Episode ended at step {step}")
            break
    
    # Calculate results
    if len(positions) > 1:
        total_distance = positions[-1] - positions[0]
        total_time = len(positions) * 0.05
        avg_velocity = total_distance / total_time if total_time > 0 else 0
    else:
        avg_velocity = 0
        total_distance = 0
    
    print(f"\n📊 ROBUSTNESS RESULTS:")
    print(f"  Failed joints: {failed_joint_names}")
    print(f"  Distance: {total_distance:.3f}m")
    print(f"  Velocity: {avg_velocity:.3f} m/s")
    print(f"  Reward: {total_reward:.0f}")
    
    if avg_velocity > 0.15:
        print("🔥 EXCELLENT - Robot handles joint failure well!")
    elif avg_velocity > 0.10:
        print("✅ GOOD - Robot adapts to joint failure")
    elif avg_velocity > 0.05:
        print("⚠️ MODERATE - Some adaptation to failure")
    else:
        print("❌ POOR - Severely impacted by joint failure")
    
    # Create video
    print(f"\n🎥 PASS 2: Creating video with failure visualization")
    print("-" * 50)
    
    render_env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')
    render_env = SuccessRewardWrapper(render_env)
    render_env = SimpleJointFailureWrapper(render_env, failed_joints)
    
    print("🎬 Rendering video...")
    frames = []
    obs = render_env.reset()[0] if hasattr(render_env.reset(), '__len__') else render_env.reset()
    current_pos = 0
    
    for step in range(min(steps, 800)):  # Limit for video size
        if step % 100 == 0:
            print(f"  Rendered {step+1}/{min(steps, 800)} frames...")
        
        # Step and render
        action, _ = model.predict(np.array([obs]), deterministic=True)
        step_result = render_env.step(action[0] if hasattr(action, '__len__') else action)
        
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result
            
        frame = render_env.render()
        
        if frame is not None:
            frame_with_text = frame.copy()
            
            # Calculate current metrics
            if hasattr(render_env, 'unwrapped') and hasattr(render_env.unwrapped, 'get_body_com'):
                try:
                    current_pos = render_env.unwrapped.get_body_com("torso")[0]
                except:
                    current_pos += 0.001  # Rough estimate
            
            distance = current_pos - (positions[0] if positions else 0)
            current_time = step * 0.05
            current_vel = distance / current_time if current_time > 0 else 0
            
            # Add overlay
            cv2.putText(frame_with_text, f'Step: {step+1}', 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame_with_text, f'Velocity: {current_vel:.3f} m/s', 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame_with_text, f'Distance: {distance:.2f}m', 
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Show failed joints
            failed_text = f'FAILED: {", ".join(failed_joint_names)}'
            cv2.putText(frame_with_text, failed_text, 
                       (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
            
            # Add failure indicator
            cv2.rectangle(frame_with_text, (10, 10), (60, 60), (0, 0, 255), 4)
            cv2.putText(frame_with_text, '!', (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            frames.append(frame_with_text)
        
        if done:
            break
    
    render_env.close()
    
    if frames:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        joint_str = "_".join([f"j{i}" for i in failed_joints])
        filename = f'{model_name}_joint_failure_{joint_str}_{timestamp}.mp4'
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
        
        for frame in frames:
            out.write(frame)
        out.release()
        
        print(f"\n🎉 JOINT FAILURE VIDEO COMPLETED!")
        print(f"📁 Saved as: {filename}")
        print(f"❌ Failed joints: {failed_joint_names}")
        print(f"📊 Performance: {avg_velocity:.3f} m/s")
        
        return filename, avg_velocity
    else:
        print("❌ No frames captured!")
        return None, 0

def main():
    parser = argparse.ArgumentParser(description='Test model with specific joint failures')
    parser.add_argument('model_folder', help='Path to model folder')
    parser.add_argument('--joints', type=int, nargs='+', default=[2], help='Joint indices to fail (default: [2])')
    parser.add_argument('--steps', type=int, default=600, help='Number of steps (default: 600)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_folder):
        print(f"❌ Model folder not found: {args.model_folder}")
        return
    
    filename, velocity = test_specific_joint_failure(args.model_folder, args.joints, args.steps)
    
    if filename:
        print(f"\n🎬 Test complete: {filename}")
        print(f"🎯 Result: {velocity:.3f} m/s with joints {args.joints} failed")
    else:
        print("❌ Test failed")

if __name__ == "__main__":
    main()