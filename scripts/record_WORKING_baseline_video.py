#!/usr/bin/env python3
"""
Record video of the ACTUALLY WORKING combination:
- Model: done/ppo_baseline_ueqbjf2x (working baseline trained with SuccessRewardWrapper)
- Wrapper: TargetWalkingWrapper (goal-directed behavior)
- Expected: 0.220 m/s with target reaching

TWO-PASS approach as documented in CLAUDE.md
"""
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.target_walking_wrapper import TargetWalkingWrapper
import realant_sim
import cv2
import os
from datetime import datetime

def record_working_baseline_video():
    """Two-pass video of WORKING baseline + TargetWalkingWrapper combo"""
    
    print("🎬 TWO-PASS VIDEO: WORKING BASELINE + TARGET WALKING")
    print(" Model: done/ppo_baseline_ueqbjf2x (speed-trained baseline)")  
    print(" Wrapper: TargetWalkingWrapper (goal-directed)")
    print(" Expected: 0.220 m/s with A-to-B locomotion")
    print("=" * 70)
    
    # WORKING model paths
    model_path = 'done/ppo_baseline_ueqbjf2x/best_model/best_model.zip'
    norm_path = 'done/ppo_baseline_ueqbjf2x/vec_normalize.pkl'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
        
    # Output settings
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    video_path = f'WORKING_baseline_target_walking_{timestamp}.mp4'
    
    print("📁 PASS 1: Collecting trajectory (TRUE performance)")
    print("-" * 50)
    
    # === PASS 1: Collect trajectory WITHOUT rendering ===
    def make_env():
        env = gym.make('RealAntMujoco-v0')  # No render_mode for pass 1
        env = TargetWalkingWrapper(env, target_distance=5.0)
        return env
    
    env = DummyVecEnv([make_env])
    
    print(f"📏 Observation space: {env.observation_space}")
    
    # Load VecNormalize - this might fail due to wrapper mismatch, but let's try
    try:
        env = VecNormalize.load(norm_path, env)
        env.training = False
        env.norm_reward = False
        print(" VecNormalize loaded (spaces matched!)")
    except Exception as e:
        print(f"⚠️  VecNormalize failed: {e}")
        print("   Will test without normalization")
        # Create fresh env without VecNormalize
        env.close()
        env = DummyVecEnv([make_env])
    
    # Load model
    model = PPO.load(model_path)
    print(" Model loaded")
    
    # Collect trajectory
    print("\n🤖 Recording robot performance...")
    trajectory = []
    obs = env.reset()
    
    # Track metrics
    positions = []
    targets_reached = 0
    episode_reward = 0
    
    initial_x = env.envs[0].unwrapped.data.qpos[0] if hasattr(env.envs[0], 'unwrapped') else 0
    print(f"🏃 Starting position: {initial_x:.3f}m")
    
    for step in range(1000):  # 50 seconds worth
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # Store trajectory for replay
        trajectory.append({
            'action': action.copy(),
            'obs': obs.copy(), 
            'reward': reward[0],
            'info': dict(info[0]) if info[0] else {}
        })
        
        episode_reward += reward[0]
        
        # Track position
        if hasattr(env.envs[0], 'unwrapped'):
            current_x = env.envs[0].unwrapped.data.qpos[0]
            positions.append(current_x)
        else:
            positions.append(0)
        
        # Check for target reaches
        if info[0] and info[0].get('target_reached', False):
            targets_reached += 1
            print(f"  🎯 Target {targets_reached} reached at step {step} (x={positions[-1]:.1f}m)")
        
        # Print progress
        if step % 100 == 0:
            current_pos = positions[-1] if positions else 0
            velocity = (current_pos - initial_x) / ((step + 1) * 0.05) if step > 0 else 0
            target_dist = info[0].get('distance_to_target', 0) if info[0] else 0
            print(f"  Step {step:3d}: x={current_pos:5.1f}m, vel={velocity:.3f}m/s, target_dist={target_dist:.1f}m")
        
        if done[0]:
            print(f"Episode ended at step {step}")
            break
    
    env.close()
    
    # Calculate metrics
    final_x = positions[-1] if positions else initial_x
    total_distance = final_x - initial_x
    avg_velocity = total_distance / (len(trajectory) * 0.05) if trajectory else 0
    
    print(f"\n📊 TRAJECTORY PERFORMANCE:")
    print(f"  Steps: {len(trajectory)}")
    print(f"  Total reward: {episode_reward:.0f}")
    print(f"  Targets reached: {targets_reached}")
    print(f"  Distance traveled: {total_distance:.2f}m")
    print(f"  Average velocity: {avg_velocity:.3f} m/s")
    
    if avg_velocity > 0.15:
        print(" Good velocity performance!")
    else:
        print("⚠️  Velocity lower than expected")
    
    if targets_reached > 0:
        print(" Goal-directed behavior confirmed!")
    else:
        print("⚠️  No targets reached - check wrapper")
    
    print(f"\n🎥 PASS 2: Rendering video from trajectory")
    print("-" * 50)
    
    # === PASS 2: Replay trajectory WITH rendering ===
    def make_render_env():
        env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')
        env = TargetWalkingWrapper(env, target_distance=5.0)
        return env
    
    render_env = DummyVecEnv([make_render_env])
    
    # Try to load normalization again for rendering
    try:
        render_env = VecNormalize.load(norm_path, render_env)
        render_env.training = False
        render_env.norm_reward = False
    except:
        pass  # Continue without normalization
    
    # Set up video writer
    frame_width, frame_height = 1920, 1080
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (frame_width, frame_height))
    
    print("🎬 Rendering video...")
    
    # Reset and replay
    obs = render_env.reset()
    targets_in_video = 0
    
    for i, step_data in enumerate(trajectory):
        # Use recorded action
        obs, _, _, info = render_env.step(step_data['action'])
        
        # Count targets in video
        if info[0] and info[0].get('target_reached', False):
            targets_in_video += 1
        
        # Render frame
        try:
            frame = render_env.envs[0].render()
            
            if frame is not None:
                # Resize frame
                frame = cv2.resize(frame, (frame_width, frame_height))
                
                # Add overlay text
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_color = (255, 255, 255)
                
                # Performance info
                current_pos = positions[i] if i < len(positions) else 0
                step_velocity = (positions[i] - positions[0]) / ((i + 1) * 0.05) if i > 0 and positions else 0
                
                # Get target info from wrapper if available
                target_pos = "?"
                target_dist = "?"
                if hasattr(render_env.envs[0], 'target_x'):
                    target_pos = f"{render_env.envs[0].target_x:.1f}m"
                    target_dist = f"{abs(render_env.envs[0].target_x - current_pos):.1f}m"
                elif info[0] and 'distance_to_target' in info[0]:
                    target_dist = f"{info[0]['distance_to_target']:.1f}m"
                
                cv2.putText(frame, f"Robot Position: {current_pos:.2f}m", 
                           (50, 50), font, 1.0, text_color, 2)
                cv2.putText(frame, f"Velocity: {step_velocity:.3f} m/s", 
                           (50, 90), font, 1.0, text_color, 2)
                cv2.putText(frame, f"Target Position: {target_pos}", 
                           (50, 130), font, 1.0, (0, 255, 255), 2)  # Cyan for target
                cv2.putText(frame, f"Distance to Target: {target_dist}", 
                           (50, 170), font, 1.0, (0, 255, 255), 2)
                cv2.putText(frame, f"Targets Reached: {targets_in_video}/{targets_reached}", 
                           (50, 210), font, 1.0, (0, 255, 0), 2)
                
                # Draw target direction indicator
                if hasattr(render_env.envs[0], 'target_x'):
                    target_x = render_env.envs[0].target_x
                    # Draw arrow pointing to target
                    center_x, center_y = frame_width // 2, frame_height // 2
                    if target_x > current_pos:
                        # Arrow pointing right
                        cv2.arrowedLine(frame, (center_x + 100, center_y - 100), 
                                      (center_x + 200, center_y - 100), 
                                      (0, 255, 255), 5, tipLength=0.3)
                        cv2.putText(frame, f"TARGET →", 
                                   (center_x + 80, center_y - 110), font, 1.0, (0, 255, 255), 2)
                    else:
                        # Arrow pointing left  
                        cv2.arrowedLine(frame, (center_x - 100, center_y - 100), 
                                      (center_x - 200, center_y - 100), 
                                      (0, 255, 255), 5, tipLength=0.3)
                        cv2.putText(frame, f"← TARGET", 
                                   (center_x - 180, center_y - 110), font, 1.0, (0, 255, 255), 2)
                
                # Show goal-directed behavior status
                if targets_in_video > 0:
                    cv2.putText(frame, "🎯 GOAL-DIRECTED LOCOMOTION ACTIVE", 
                               (frame_width//2 - 300, frame_height - 50), 
                               font, 1.2, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "WALKING TO TARGET...", 
                               (frame_width//2 - 200, frame_height - 50), 
                               font, 1.2, (255, 255, 0), 2)
                
                # Convert RGB to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)
            
        except Exception as e:
            print(f"Warning: Frame {i} render failed: {e}")
            continue
        
        if (i + 1) % 100 == 0:
            print(f"  Rendered {i+1}/{len(trajectory)} frames...")
    
    video_writer.release()
    render_env.close()
    
    print(f"\n VIDEO COMPLETED!")
    print(f"📁 Saved as: {video_path}")
    print(f"📊 Performance: {avg_velocity:.3f} m/s, {targets_reached} targets")
    print(f"🎯 Shows: Working baseline + TargetWalkingWrapper combo")
    
    return video_path, {
        'velocity': avg_velocity,
        'targets': targets_reached,
        'distance': total_distance,
        'reward': episode_reward
    }

if __name__ == "__main__":
    video_path, metrics = record_working_baseline_video()
    print(f"\n🎬 Video ready: {video_path}")
    print(f"📈 Final: {metrics['velocity']:.3f} m/s, {metrics['targets']} targets")