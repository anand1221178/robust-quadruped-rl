#!/usr/bin/env python3
"""
Two-pass video recording for SmoothTargetWrapper baseline
Pass 1: Collect trajectory with TRUE performance metrics
Pass 2: Replay with rendering for video
"""
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.smooth_target_wrapper import SmoothTargetWrapper
import realant_sim
import cv2
from PIL import Image
import os
from datetime import datetime

def record_smooth_baseline_video():
    """Two-pass video recording with accurate metrics"""
    
    print("🎬 TWO-PASS VIDEO RECORDING: SMOOTH BASELINE")
    print("=" * 60)
    
    # Model paths
    model_path = 'experiments/ppo_smooth_baseline_rohl32fn/best_model.zip'
    norm_path = 'experiments/ppo_smooth_baseline_rohl32fn/vec_normalize.pkl'
    
    # Output settings
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    video_path = f'smooth_baseline_goal_directed_{timestamp}.mp4'
    
    print("📁 PASS 1: Collecting trajectory (TRUE performance)")
    print("-" * 50)
    
    # === PASS 1: Collect trajectory WITHOUT rendering ===
    def make_env():
        env = gym.make('RealAntMujoco-v0')
        env = SmoothTargetWrapper(env, target_distance=5.0)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Load VecNormalize and model
    try:
        env = VecNormalize.load(norm_path, env)
        env.training = False
        print("✅ VecNormalize loaded")
    except Exception as e:
        print(f"⚠️  VecNormalize failed: {e}")
    
    model = PPO.load(model_path)
    print("✅ Model loaded")
    
    # Collect trajectory
    trajectory = []
    obs = env.reset()
    episode_reward = 0
    targets_reached = 0
    positions = []
    
    print("🤖 Recording robot performance...")
    
    for step in range(800):  # ~40 seconds at 20fps
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        episode_reward += reward[0]
        
        # Store trajectory data
        trajectory.append({
            'obs': obs.copy(),
            'action': action.copy(),
            'reward': reward[0],
            'info': info[0].copy() if info[0] else {}
        })
        
        # Track performance metrics
        if hasattr(env.envs[0], 'unwrapped'):
            current_x = env.envs[0].unwrapped.data.qpos[0]
            positions.append(current_x)
        
        if info[0] and 'target_reached' in info[0] and info[0]['target_reached']:
            targets_reached += 1
            print(f"  🎯 Target {targets_reached} reached at step {step}")
        
        # Print progress every 100 steps
        if step % 100 == 0:
            current_pos = positions[-1] if positions else 0
            vel = info[0].get('velocity', 0) if info[0] else 0
            dist_to_target = info[0].get('distance_to_target', 0) if info[0] else 0
            print(f"  Step {step:3d}: x={current_pos:5.1f}m, vel={vel:.3f}m/s, target_dist={dist_to_target:.1f}m")
        
        if done[0]:
            break
    
    env.close()
    
    # Calculate final metrics
    total_distance = positions[-1] - positions[0] if len(positions) >= 2 else 0
    avg_velocity = total_distance / (step * 0.05) if step > 0 else 0  # dt = 0.05s
    
    print(f"\n📊 TRAJECTORY COLLECTED:")
    print(f"  Steps: {len(trajectory)}")
    print(f"  Total reward: {episode_reward:.0f}")
    print(f"  Targets reached: {targets_reached}")
    print(f"  Distance traveled: {total_distance:.1f}m")
    print(f"  Average velocity: {avg_velocity:.3f} m/s")
    
    print(f"\n🎥 PASS 2: Rendering video from trajectory")
    print("-" * 50)
    
    # === PASS 2: Replay trajectory WITH rendering ===
    def make_render_env():
        env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')
        env = SmoothTargetWrapper(env, target_distance=5.0)
        return env
    
    render_env = DummyVecEnv([make_render_env])
    
    # Load normalization again
    try:
        render_env = VecNormalize.load(norm_path, render_env)
        render_env.training = False
    except:
        pass
    
    # Setup video writer
    frame_width, frame_height = 1920, 1080
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (frame_width, frame_height))
    
    print("🎬 Rendering video...")
    
    # Reset environment and replay trajectory
    obs = render_env.reset()
    targets_in_video = 0
    
    for i, step_data in enumerate(trajectory):
        # Use recorded action
        obs, _, _, info = render_env.step(step_data['action'])
        
        # Count targets reached in video
        if info[0] and 'target_reached' in info[0] and info[0]['target_reached']:
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
                
                # Add text overlay with metrics
                text_info = [
                    f"Step: {i+1}/{len(trajectory)}",
                    f"Reward: {episode_reward:.0f}",
                    f"Targets: {targets_reached}",
                    f"Velocity: {avg_velocity:.3f} m/s",
                    f"Distance: {total_distance:.1f}m"
                ]
                
                y_pos = 50
                for text in text_info:
                    cv2.putText(frame_bgr, text, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                               1.2, (255, 255, 255), 2, cv2.LINE_AA)
                    y_pos += 40
                
                video_writer.write(frame_bgr)
            
        except Exception as e:
            print(f"  Frame {i} render failed: {e}")
            # Create black frame as fallback
            black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            video_writer.write(black_frame)
        
        # Progress indicator
        if i % 50 == 0:
            print(f"  Rendered {i+1}/{len(trajectory)} frames...")
    
    video_writer.release()
    render_env.close()
    
    print(f"\n✅ VIDEO COMPLETED!")
    print(f"📁 Saved as: {video_path}")
    print(f"📊 Final metrics:")
    print(f"  - Episode reward: {episode_reward:.0f}")
    print(f"  - Targets reached: {targets_reached}")
    print(f"  - Average velocity: {avg_velocity:.3f} m/s") 
    print(f"  - Total distance: {total_distance:.1f}m")
    print(f"  - Video length: {len(trajectory)} frames")
    
    return video_path, {
        'reward': episode_reward,
        'targets': targets_reached,
        'velocity': avg_velocity,
        'distance': total_distance
    }

if __name__ == "__main__":
    video_path, metrics = record_smooth_baseline_video()
    print(f"\n🎬 Video ready: {video_path}")
    print(f"📈 Performance: {metrics['velocity']:.3f} m/s, {metrics['targets']} targets")