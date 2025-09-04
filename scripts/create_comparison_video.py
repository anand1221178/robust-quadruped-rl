#!/usr/bin/env python3
"""
Create side-by-side comparison video of Baseline vs DR under joint failures
Perfect for research presentations!
"""

import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
import realant_sim
import imageio
import os
from PIL import Image, ImageDraw, ImageFont

def create_comparison_video():
    print("=" * 60)
    print("🎬 CREATING BASELINE vs DR COMPARISON VIDEO")
    print("=" * 60)
    
    models = {
        'Baseline (No DR)': 'done/ppo_baseline_ueqbjf2x/best_model/best_model.zip',
        'With DR Training': 'experiments/ppo_dr_gentle_v2_wptws01u/best_model/best_model.zip'
    }
    
    failure_rates = [0.0, 0.15, 0.30]  # Show degradation
    
    for failure_rate in failure_rates:
        print(f"\n📹 Recording at {failure_rate*100:.0f}% failure rate...")
        
        frames_dict = {}
        
        for model_name, model_path in models.items():
            print(f"  Recording {model_name}...")
            
            # Load model
            model = PPO.load(model_path)
            
            # Create environment with rendering
            def make_env():
                env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')
                env = SuccessRewardWrapper(env)
                return env
            
            env = DummyVecEnv([make_env])
            
            # Load VecNormalize
            vec_path = model_path.replace('best_model.zip', '../vec_normalize.pkl')
            if os.path.exists(vec_path):
                env = VecNormalize.load(vec_path, env)
                env.training = False
                env.norm_reward = False
            
            # Collect frames
            frames = []
            obs = env.reset()
            
            for step in range(200):  # Shorter video
                action, _ = model.predict(obs, deterministic=True)
                
                # Apply failures
                if failure_rate > 0:
                    for i in range(len(action[0])):
                        if np.random.random() < failure_rate:
                            action[0][i] = 0.0  # Lock joint
                
                obs, _, done, _ = env.step(action)
                
                # Capture frame
                frame = env.render(mode='rgb_array')
                if frame is not None:
                    # Add label to frame
                    img = Image.fromarray(frame)
                    draw = ImageDraw.Draw(img)
                    
                    # Add text label
                    text = f"{model_name}"
                    draw.rectangle([(0, 0), (300, 30)], fill=(0, 0, 0, 180))
                    draw.text((10, 5), text, fill=(255, 255, 255))
                    
                    if failure_rate > 0:
                        failure_text = f"{failure_rate*100:.0f}% Joint Failures"
                        draw.rectangle([(0, 30), (200, 55)], fill=(255, 0, 0, 180))
                        draw.text((10, 35), failure_text, fill=(255, 255, 255))
                    
                    frames.append(np.array(img))
                
                if done[0]:
                    break
            
            env.close()
            frames_dict[model_name] = frames
        
        # Combine side by side
        combined_frames = []
        min_len = min(len(frames_dict[name]) for name in frames_dict)
        
        for i in range(min_len):
            frame1 = frames_dict['Baseline (No DR)'][i]
            frame2 = frames_dict['With DR Training'][i]
            
            # Resize if needed
            h = min(frame1.shape[0], frame2.shape[0])
            w = min(frame1.shape[1], frame2.shape[1])
            
            frame1 = frame1[:h, :w]
            frame2 = frame2[:h, :w]
            
            # Combine horizontally
            combined = np.hstack([frame1, frame2])
            combined_frames.append(combined)
        
        # Save video
        output_name = f"comparison_{int(failure_rate*100)}pct_failures.mp4"
        imageio.mimsave(output_name, combined_frames, fps=30)
        print(f"  ✅ Saved: {output_name}")
    
    print("\n" + "=" * 60)
    print("🎉 COMPARISON VIDEOS CREATED!")
    print("=" * 60)
    print("Files created:")
    print("  • comparison_0pct_failures.mp4 - Normal operation")
    print("  • comparison_15pct_failures.mp4 - Moderate failures")
    print("  • comparison_30pct_failures.mp4 - Extreme failures")
    print("\nUse these for presentations to show DR superiority!")

if __name__ == "__main__":
    create_comparison_video()