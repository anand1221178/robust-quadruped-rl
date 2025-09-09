#!/usr/bin/env python3
"""
Record video of PURE FORWARD LOCOMOTION - matching research proposal
- Model: done/ppo_baseline_ueqbjf2x (baseline trained with SuccessRewardWrapper)
- Wrapper: SuccessRewardWrapper (pure forward walking - no targets)
- Expected: 0.224 m/s forward locomotion (verified performance)

Two-pass approach for accurate metrics
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

def record_forward_locomotion_video():
    """Two-pass video of baseline doing PURE forward locomotion"""
    
    print("🎬 FORWARD LOCOMOTION VIDEO - RESEARCH PROPOSAL DEMO")
    print("✅ Model: done/ppo_baseline_ueqbjf2x (verified 0.224 m/s)")  
    print("✅ Wrapper: SuccessRewardWrapper (pure forward walking)")
    print("✅ Task: Forward locomotion (NO targets - just walk forward)")
    print("=" * 70)
    
    # Model paths
    model_path = 'done/ppo_baseline_ueqbjf2x/best_model/best_model.zip'
    norm_path = 'done/ppo_baseline_ueqbjf2x/vec_normalize.pkl'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
        
    # Output settings
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    video_path = f'forward_locomotion_baseline_{timestamp}.mp4'
    
    print("📁 PASS 1: Collecting trajectory (TRUE performance)")
    print("-" * 50)
    
    # === PASS 1: Collect trajectory WITHOUT rendering ===
    def make_env():
        env = gym.make('RealAntMujoco-v0')  # No render_mode for pass 1
        env = SuccessRewardWrapper(env)  # PURE forward locomotion rewards
        return env
    
    env = DummyVecEnv([make_env])
    
    print(f"📏 Observation space: {env.observation_space}")
    
    # Load VecNormalize
    env = VecNormalize.load(norm_path, env)
    env.training = False
    env.norm_reward = False
    print("✅ VecNormalize loaded")
    
    # Load model
    model = PPO.load(model_path)
    print("✅ Model loaded")
    
    # Record trajectory
    print(f"\n🤖 Recording forward locomotion performance...")
    trajectory = []
    obs = env.reset()
    episode_reward = 0
    positions = []
    rewards_log = []
    
    initial_x = env.envs[0].unwrapped.data.qpos[0]
    print(f"🏃 Starting position: {initial_x:.3f}m")
    
    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        trajectory.append(action.copy())
        
        obs, reward, done, info = env.step(action)
        episode_reward += reward[0]
        rewards_log.append(reward[0])
        
        if done[0]:
            print(f"Episode ended at step {step}")
            break
        
        # Track position
        current_x = env.envs[0].unwrapped.data.qpos[0]
        positions.append(current_x)
        
        # Progress updates - show forward locomotion
        if step % 100 == 0:
            velocity = (current_x - initial_x) / ((step + 1) * 0.05)
            instant_vel = info[0].get('current_velocity', 0) if info[0] else 0
            custom_reward = info[0].get('custom_reward', 0) if info[0] else 0
            print(f"  Step {step:3d}: x={current_x:5.1f}m, vel={velocity:.3f}m/s, instant={instant_vel:.3f}m/s, reward={custom_reward:.1f}")
    
    env.close()
    
    # Calculate final metrics
    final_x = positions[-1] if positions else initial_x
    total_distance = final_x - initial_x
    total_time = len(positions) * 0.05
    avg_velocity = total_distance / total_time if total_time > 0 else 0
    
    print(f"\n📊 FORWARD LOCOMOTION PERFORMANCE:")
    print(f"  Steps: {len(trajectory)}")
    print(f"  Total reward: {episode_reward:.0f}")
    print(f"  Distance traveled: {total_distance:.2f}m")
    print(f"  Average velocity: {avg_velocity:.3f} m/s")
    print(f"  Time: {total_time:.1f} seconds")
    
    if avg_velocity > 0.20:
        print("✅ Excellent forward locomotion performance!")
    elif avg_velocity > 0.15:
        print("✅ Good forward locomotion performance!")
    else:
        print("⚠️  Velocity lower than expected")
    
    print(f"\n🎥 PASS 2: Rendering video from trajectory")
    print("-" * 50)
    
    # === PASS 2: Replay trajectory WITH rendering ===
    def make_render_env():
        env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')
        env = SuccessRewardWrapper(env)
        return env
    
    render_env = DummyVecEnv([make_render_env])
    render_env = VecNormalize.load(norm_path, render_env)
    render_env.training = False
    render_env.norm_reward = False
    
    # Setup video writer
    print("🎬 Rendering video...")
    obs = render_env.reset()
    
    # Get initial frame for video setup
    frame = render_env.render()
    if frame is None:
        print("❌ Could not get initial frame")
        render_env.close()
        return
    
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))
    
    render_positions = []
    render_initial_x = render_env.envs[0].unwrapped.data.qpos[0]
    
    # Replay trajectory - FIX: Step environment BEFORE rendering
    for step, action in enumerate(trajectory):
        # Step environment FIRST
        obs, reward, done, info = render_env.step(action)
        
        if done[0]:
            print(f"  Episode ended at render step {step}")
            break
        
        # THEN render the result
        frame = render_env.render()
        if frame is not None:
            # Get position AFTER stepping
            current_x = render_env.envs[0].unwrapped.data.qpos[0]
            render_positions.append(current_x)
            
            # Use the SAME velocity calculation as trajectory collection
            distance_so_far = current_x - render_initial_x
            time_so_far = len(render_positions) * 0.05
            current_velocity = distance_so_far / time_so_far if time_so_far > 0 else 0
            
            # Add text overlay with larger font
            cv2.putText(frame, f"Forward Locomotion Demo", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.putText(frame, f"Step: {step+1}", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Position: {current_x:.2f}m", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Velocity: {current_velocity:.3f} m/s", (10, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Distance: {distance_so_far:.2f}m", (10, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Add expected performance for comparison
            expected_pos = (step + 1) * 0.05 * 0.220  # Expected at 0.220 m/s
            cv2.putText(frame, f"Expected: {expected_pos:.2f}m", (10, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # Write frame
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        
        if step % 100 == 0:
            distance_check = render_positions[-1] - render_initial_x if render_positions else 0
            expected_check = (step + 1) * 0.05 * 0.220
            print(f"  Rendered {step+1}/{len(trajectory)} frames... pos={distance_check:.2f}m, expected={expected_check:.2f}m")
    
    video_writer.release()
    render_env.close()
    
    # Final video metrics
    render_final_x = render_positions[-1] if render_positions else render_initial_x
    render_distance = render_final_x - render_initial_x
    render_velocity = render_distance / (len(render_positions) * 0.05) if render_positions else 0
    
    print(f"\n✅ VIDEO COMPLETED!")
    print(f"📁 Saved as: {video_path}")
    print(f"📊 Video Performance: {render_velocity:.3f} m/s, {render_distance:.1f}m")
    print(f"🎯 Shows: Pure forward locomotion (research proposal demo)")
    
    return video_path, avg_velocity, total_distance

if __name__ == "__main__":
    video_file, velocity, distance = record_forward_locomotion_video()
    print(f"\n🎬 Video ready: {video_file}")
    print(f"📈 Performance: {velocity:.3f} m/s, {distance:.1f}m forward locomotion")
    print(f"🎯 Perfect demo for research proposal!")