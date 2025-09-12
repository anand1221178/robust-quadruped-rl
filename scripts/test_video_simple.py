#!/usr/bin/env python3
"""Simple video test to isolate the issue"""
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

def test_simple_video():
    print("🔍 TESTING SIMPLE VIDEO RECORDING")
    
    # Create simple test environment (no DR wrapper complications)
    def make_env():
        env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')
        env = SuccessRewardWrapper(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Load systematic curriculum model
    model_path = 'experiments/ppo_systematic_curriculum_54M_v9kog7p1/final_model.zip'
    vec_path = 'experiments/ppo_systematic_curriculum_54M_v9kog7p1/vec_normalize.pkl'
    
    try:
        env = VecNormalize.load(vec_path, env)
        env.training = False
        env.norm_reward = False
        print("✅ VecNormalize loaded")
    except:
        print("⚠️  No VecNormalize")
    
    model = PPO.load(model_path)
    print("✅ Model loaded")
    
    # Get a single frame to test
    obs = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
    frame = env.envs[0].render()
    print(f"Frame shape: {frame.shape}")
    print(f"Frame dtype: {frame.dtype}")
    
    env.close()
    
    if frame is not None:
        print("✅ Got frame from environment")
        
        # Try to save a single frame as image first
        cv2.imwrite('test_frame.png', frame)
        print("✅ Saved test frame as PNG")
        
        # Now try video with exact same settings as working SR2L
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'test_video_{timestamp}.mp4'
        
        frame_size = (frame.shape[1], frame.shape[0])  # (width, height)
        fps = 60
        
        print(f"Trying video: {output_path}")
        print(f"Frame size: {frame_size}")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        
        if video_writer.isOpened():
            print("✅ VideoWriter opened successfully")
            
            # Write the same frame 60 times (1 second of video)
            for i in range(60):
                success = video_writer.write(frame)
                if not success:
                    print(f"❌ Frame write failed at frame {i}")
                    break
                elif i % 10 == 0:
                    print(f"  Wrote frame {i}")
            
            video_writer.release()
            
            # Check if video was created
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                print(f"✅ Video created successfully: {output_path}")
                print(f"File size: {os.path.getsize(output_path)} bytes")
            else:
                print("❌ Video creation failed or file too small")
        else:
            print("❌ VideoWriter failed to open")
    else:
        print("❌ No frame from environment")

if __name__ == "__main__":
    test_simple_video()