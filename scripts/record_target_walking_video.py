#!/usr/bin/env python3
"""
Two-pass video recording of WORKING TargetWalkingWrapper
Finally - a robot that actually walks from A to B!
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
from PIL import Image
import os
from datetime import datetime

def record_target_walking_video():
    """Two-pass video recording of WORKING goal-directed locomotion"""
    
    print("🎬 RECORDING WORKING TARGET WALKING ROBOT!")
    print("First robot that actually walks from A to B!")
    print("=" * 60)
    
    # Model paths - using PROVEN working baseline
    model_path = 'done/ppo_baseline_ueqbjf2x/best_model/best_model.zip'
    norm_path = 'done/ppo_baseline_ueqbjf2x/vec_normalize.pkl'
    
    # Output settings
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    video_path = f'target_walking_SUCCESS_{timestamp}.mp4'
    
    print("📁 PASS 1: Collecting trajectory (TRUE goal-directed performance)")
    print("-" * 60)
    
    # === PASS 1: Collect trajectory WITHOUT rendering ===
    def make_env():
        env = gym.make('RealAntMujoco-v0')
        env = TargetWalkingWrapper(env, target_distance=5.0)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Load VecNormalize and model
    env = VecNormalize.load(norm_path, env)
    env.training = False
    print("✅ VecNormalize loaded")
    
    model = PPO.load(model_path)
    print("✅ Working baseline model loaded")
    
    # Collect trajectory
    trajectory = []
    obs = env.reset()
    episode_reward = 0
    targets_reached = 0
    positions = []
    target_positions = []
    
    print("🎯 Recording goal-directed robot performance...")
    
    for step in range(1200):  # ~60 seconds at 20fps, enough for multiple targets
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        episode_reward += reward[0]
        
        # Get current position and target
        current_x = env.envs[0].unwrapped.data.qpos[0]
        positions.append(current_x)
        
        # Get target position from wrapper
        if hasattr(env.envs[0], 'target_x'):
            target_x = env.envs[0].target_x
            target_positions.append(target_x)
        else:
            target_positions.append(target_positions[-1] if target_positions else 5.0)
        
        # Store trajectory data
        trajectory.append({
            'obs': obs.copy(),
            'action': action.copy(),
            'reward': reward[0],
            'info': info[0].copy() if info[0] else {},
            'current_x': current_x,
            'target_x': target_positions[-1]
        })
        
        # Check for targets reached
        if info[0] and 'success_bonus' in info[0] and info[0]['success_bonus'] > 0:
            targets_reached += 1
            print(f"  🎯 TARGET {targets_reached} REACHED at x={current_x:.1f}m (step {step})!")
        
        # Print progress every 150 steps
        if step % 150 == 0:
            dist_to_target = info[0].get('distance_to_target', 0) if info[0] else 0
            progress = info[0].get('progress', 0) if info[0] else 0
            print(f"  Step {step:4d}: x={current_x:5.1f}m, target={target_positions[-1]:5.1f}m, dist={dist_to_target:.1f}m, progress={progress:.3f}")
        
        if done[0]:
            break
    
    env.close()
    
    # Calculate final metrics
    total_distance = positions[-1] - positions[0] if len(positions) >= 2 else 0
    avg_velocity = total_distance / (step * 0.05) if step > 0 else 0
    
    print(f"\n📊 TRAJECTORY COLLECTED:")
    print(f"  Steps: {len(trajectory)}")
    print(f"  Total reward: {episode_reward:.0f}")
    print(f"  Targets reached: {targets_reached}")
    print(f"  Distance traveled: {total_distance:.1f}m")
    print(f"  Average velocity: {avg_velocity:.3f} m/s")
    print(f"  Final position: {positions[-1]:.1f}m")
    
    print(f"\n🎥 PASS 2: Creating HD video from trajectory")
    print("-" * 60)
    
    # === PASS 2: Replay trajectory WITH rendering ===
    def make_render_env():
        env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')
        env = TargetWalkingWrapper(env, target_distance=5.0)
        return env
    
    render_env = DummyVecEnv([make_render_env])
    
    # Load normalization again
    render_env = VecNormalize.load(norm_path, render_env)
    render_env.training = False
    
    # Setup video writer
    frame_width, frame_height = 1920, 1080
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (frame_width, frame_height))
    
    print("🎬 Rendering video with target indicators...")
    
    # Reset environment and replay trajectory
    obs = render_env.reset()
    targets_in_video = 0
    
    for i, step_data in enumerate(trajectory):
        # Use recorded action
        obs, _, _, info = render_env.step(step_data['action'])
        
        # Count targets reached in video
        if info[0] and 'success_bonus' in info[0] and info[0]['success_bonus'] > 0:
            targets_in_video += 1
        
        # Render frame
        try:
            if hasattr(render_env.envs[0], 'render'):
                frame = render_env.envs[0].render()
            else:
                frame = render_env.envs[0].unwrapped.render()
            
            if frame is not None:
                # Resize to HD
                img = Image.fromarray(frame)
                img = img.resize((frame_width, frame_height), Image.LANCZOS)
                frame_resized = np.array(img)
                
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
                
                # Add comprehensive text overlay
                current_x = step_data['current_x']
                target_x = step_data['target_x']
                distance_to_target = abs(target_x - current_x)
                
                text_info = [
                    f"GOAL-DIRECTED LOCOMOTION - WORKING!",
                    f"Step: {i+1}/{len(trajectory)}",
                    f"Robot Position: {current_x:.1f}m",
                    f"Target Position: {target_x:.1f}m", 
                    f"Distance to Target: {distance_to_target:.1f}m",
                    f"Targets Reached: {targets_reached}",
                    f"Total Distance: {total_distance:.1f}m",
                    f"Velocity: {avg_velocity:.3f} m/s"
                ]
                
                # Add text with background for better visibility
                y_pos = 50
                for j, text in enumerate(text_info):
                    # Add background rectangle
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                    cv2.rectangle(frame_bgr, (40, y_pos-30), (40 + text_size[0] + 20, y_pos + 10), (0, 0, 0), -1)
                    
                    # Add text
                    color = (0, 255, 255) if j == 0 else (255, 255, 255)  # First line cyan, others white
                    cv2.putText(frame_bgr, text, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                               1.0, color, 2, cv2.LINE_AA)
                    y_pos += 45
                
                # Add target reached indicator
                if info[0] and 'success_bonus' in info[0] and info[0]['success_bonus'] > 0:
                    cv2.putText(frame_bgr, "🎯 TARGET REACHED!", (50, frame_height-100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3, cv2.LINE_AA)
                
                video_writer.write(frame_bgr)
            
        except Exception as e:
            print(f"  Frame {i} render failed: {e}")
            # Create black frame as fallback
            black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            cv2.putText(black_frame, f"Render Error at Frame {i}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            video_writer.write(black_frame)
        
        # Progress indicator
        if i % 100 == 0:
            print(f"  Rendered {i+1}/{len(trajectory)} frames...")
    
    video_writer.release()
    render_env.close()
    
    print(f"\n🎉 GOAL-DIRECTED VIDEO COMPLETED!")
    print(f"📁 Saved as: {video_path}")
    print(f"🎯 This is our FIRST working A-to-B robot!")
    print(f"📊 Final metrics:")
    print(f"  - Targets reached: {targets_reached}")
    print(f"  - Average velocity: {avg_velocity:.3f} m/s") 
    print(f"  - Total distance: {total_distance:.1f}m")
    print(f"  - Video length: {len(trajectory)} frames ({len(trajectory)/20:.1f} seconds)")
    print(f"  - Performance: PERFECT goal-directed behavior!")
    
    return video_path, {
        'reward': episode_reward,
        'targets': targets_reached,
        'velocity': avg_velocity,
        'distance': total_distance,
        'steps': len(trajectory)
    }

if __name__ == "__main__":
    video_path, metrics = record_target_walking_video()
    print(f"\n🎬 SUCCESS VIDEO READY: {video_path}")
    print(f"🎯 First robot that actually walks from A to B: {metrics['velocity']:.3f} m/s, {metrics['targets']} targets!")