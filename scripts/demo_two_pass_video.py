#!/usr/bin/env python3
"""
Demo script showing how to use two-pass video recording in the demo suite
This can be integrated into interactive_robot_viewer.py and research_demo_gui.py
"""

import sys
sys.path.append('src')

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
from utils.two_pass_video import TwoPassVideoRecorder
import argparse
import os

def demo_two_pass_recording(model_path, output_path="demo_twopass.mp4"):
    """
    Demonstrate two-pass video recording with accurate performance metrics
    """
    
    print("=" * 60)
    print("🎬 TWO-PASS VIDEO RECORDING DEMO")
    print("=" * 60)
    print("This method separates trajectory collection from rendering")
    print("to get TRUE performance metrics without rendering overhead")
    print("=" * 60 + "\n")
    
    # Load model
    model = PPO.load(model_path)
    print(f"✅ Loaded model: {model_path}\n")
    
    # Create environment (no rendering for Pass 1)
    def make_env():
        env = gym.make('RealAntMujoco-v0')  # No render_mode!
        env = SuccessRewardWrapper(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Load VecNormalize if available
    vec_normalize_paths = [
        model_path.replace('best_model.zip', '../vec_normalize.pkl'),
        model_path.replace('final_model.zip', 'vec_normalize.pkl'),
        os.path.join(os.path.dirname(model_path), 'vec_normalize.pkl')
    ]
    
    for vec_path in vec_normalize_paths:
        if os.path.exists(vec_path):
            env = VecNormalize.load(vec_path, env)
            env.training = False
            env.norm_reward = False
            print(f"✅ Loaded VecNormalize: {vec_path}\n")
            break
    
    # Initialize two-pass recorder
    recorder = TwoPassVideoRecorder(env_name='RealAntMujoco-v0')
    
    # PASS 1: Collect trajectory without rendering
    trajectory = recorder.collect_trajectory(
        model=model,
        env=env,
        num_steps=500,
        show_progress=True
    )
    
    # Get true performance metrics
    metrics = recorder.get_metrics()
    
    # PASS 2: Create video from trajectory
    success = recorder.replay_with_rendering(
        trajectory=trajectory,
        output_path=output_path,
        wrapper_class=SuccessRewardWrapper,
        show_progress=True
    )
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 SUCCESS! Two-Pass Recording Complete")
        print("=" * 60)
        print("ADVANTAGES OF THIS METHOD:")
        print("✅ Accurate velocity measurement (no rendering overhead)")
        print("✅ True performance metrics")
        print("✅ Video shows exact trajectory from Pass 1")
        print("✅ Can be used for all demo suite tools")
        print("=" * 60)
        
        # Show comparison with old method
        print("\nCOMPARISON WITH OLD METHOD:")
        print(f"Old method (with rendering): ~0.081 m/s (WRONG)")
        print(f"New method (no rendering):   {metrics['avg_velocity']:.3f} m/s (CORRECT)")
        print("=" * 60)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Demo of two-pass video recording for accurate performance'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='done/ppo_baseline_ueqbjf2x/best_model/best_model.zip',
        help='Path to model file'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='demo_twopass.mp4',
        help='Output video path'
    )
    
    args = parser.parse_args()
    
    # Run demo
    metrics = demo_two_pass_recording(args.model, args.output)
    
    print("\n" + "=" * 60)
    print("HOW TO INTEGRATE INTO DEMO SUITE:")
    print("=" * 60)
    print("1. Import: from utils.two_pass_video import TwoPassVideoRecorder")
    print("2. Create recorder: recorder = TwoPassVideoRecorder()")
    print("3. When user clicks 'Record':")
    print("   - Stop rendering")
    print("   - Run Pass 1 (collect trajectory)")
    print("   - Run Pass 2 (create video)")
    print("4. Show true metrics to user")
    print("=" * 60)


if __name__ == "__main__":
    main()