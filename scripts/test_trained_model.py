#!/usr/bin/env python3
"""
Test your trained model and create videos
"""

import os
import glob
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import cv2
import warnings

# Add path to import your metrics
import sys
sys.path.append('tests')
sys.path.append('src')  # Add src to path for RealAnt environments
from define_success_metrics import MetricsEvaluator
import realant_sim  # Import RealAnt environments

warnings.filterwarnings("ignore")

def find_latest_model():
    """Find the most recent experiment"""
    experiment_dirs = glob.glob("experiments/ppo_*/")
    if not experiment_dirs:
        print("No experiments found!")
        return None
    
    # Get the most recent one
    latest = max(experiment_dirs, key=os.path.getmtime)
    return latest

def test_model(model_path, vec_normalize_path=None, render=False, num_episodes=5):
    """Test a trained model"""
    
    print(f"\nTesting model: {model_path}")
    
    # Load the model
    model = PPO.load(model_path)
    
    # Create environment based on whether we have normalization
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        # Create VecEnv for normalized environment
        def make_env():
            # Check if model was trained with RealAnt or Ant-v4
            # You can detect this from config or just use RealAnt by default now
            return gym.make('RealAntMujoco-v0', render_mode='human' if render else 'rgb_array')
        
        env = DummyVecEnv([make_env])
        
        # Try to load VecNormalize, but handle obs space mismatch
        try:
            env = VecNormalize.load(vec_normalize_path, env)
            env.training = False  # Don't update stats during testing
            env.norm_reward = False  # Don't normalize rewards during testing
            print("✅ Loaded normalization stats")
            use_vecenv = True
        except AssertionError as e:
            if "spaces must have the same shape" in str(e):
                print("⚠️  VecNormalize shape mismatch - model trained on different environment")
                print("   Testing without normalization (results may be inaccurate)")
                # Use regular env without normalization
                env.close()
                env = gym.make('RealAntMujoco-v0', render_mode='human' if render else 'rgb_array')
                use_vecenv = False
            else:
                raise e
    else:
        # Regular Gym environment
        env = gym.make('RealAntMujoco-v0', render_mode='human' if render else 'rgb_array')
        use_vecenv = False
    
    # Test metrics
    all_rewards = []
    all_lengths = []
    
    for episode in range(num_episodes):
        # Reset based on environment type
        if use_vecenv:
            obs = env.reset()  # VecEnv API: returns only obs
        else:
            obs, info = env.reset()  # Gym API: returns (obs, info)
        
        total_reward = 0
        steps = 0
        
        while True:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            if use_vecenv:
                obs, reward, done, info = env.step(action)
                # Extract scalar values from arrays
                reward = reward[0] if isinstance(reward, np.ndarray) else reward
                done = done[0] if isinstance(done, np.ndarray) else done
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
            total_reward += reward
            steps += 1
            
            # Check if episode ended
            if done:
                break
            
            if steps > 1000:  # Safety limit
                break
        
        all_rewards.append(total_reward)
        all_lengths.append(steps)
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")
    
    print(f"\nAverage reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Average length: {np.mean(all_lengths):.0f} ± {np.std(all_lengths):.0f}")
    
    env.close()

def create_video(model_path, vec_normalize_path=None, video_name="robot_walking.mp4"):
    """Create a video of the trained agent"""
    
    print(f"\nCreating video...")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment based on whether we have normalization
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        # Create VecEnv for video recording
        def make_env():
            return gym.make('RealAntMujoco-v0', render_mode='rgb_array')
        
        env = DummyVecEnv([make_env])
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
        use_vecenv = True
    else:
        env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')
        use_vecenv = False
    
    # Video settings
    fps = 30
    frames = []
    
    # Reset
    if use_vecenv:
        obs = env.reset()
    else:
        obs, info = env.reset()
    
    # Record one episode
    for _ in range(1000):  # Max 1000 steps (10 seconds at 100Hz)
        action, _ = model.predict(obs, deterministic=True)
        
        if use_vecenv:
            obs, reward, done, info = env.step(action)
            done = done[0] if isinstance(done, np.ndarray) else done
        else:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        # Render and save frame  
        frame = env.render()
        if isinstance(frame, np.ndarray):
            frames.append(frame)
        
        if done:
            break
    
    env.close()
    
    # Save video
    if frames:
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Video saved as: {video_name}")
        print(f"Duration: {len(frames)/fps:.1f} seconds")
    else:
        print("No frames captured for video!")

def evaluate_with_metrics(model_path, vec_normalize_path=None):
    """Evaluate using your success metrics"""
    
    print("\n" + "="*60)
    print("Evaluating with Success Metrics")
    print("="*60)
    
    # Load model
    model = PPO.load(model_path)
    
    # For metrics evaluation, we'll use the raw environment
    # since MetricsEvaluator expects a regular Gym env
    # Note: MetricsEvaluator needs to be updated if it expects Ant-v4 specifically
    evaluator = MetricsEvaluator()
    
    # Define policy function that handles normalization if needed
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        # Load normalization stats for manual normalization
        dummy_env = DummyVecEnv([lambda: gym.make('RealAntMujoco-v0')])
        vec_norm = VecNormalize.load(vec_normalize_path, dummy_env)
        vec_norm.training = False
        
        def policy_fn(obs):
            # Normalize observation
            norm_obs = vec_norm.normalize_obs(obs)
            action, _ = model.predict(norm_obs, deterministic=True)
            return action
    else:
        def policy_fn(obs):
            action, _ = model.predict(obs, deterministic=True)
            return action
    
    # Evaluate with reasonable number of episodes
    results = evaluator.evaluate_batch(n_episodes=100, policy=policy_fn)
    
    print("\nSuccess Metrics:")
    print(f"  Success rate: {results['success_rate']*100:.1f}%")
    print(f"  Avg distance: {results['avg_distance']:.2f} m")
    print(f"  Avg velocity: {results['avg_velocity']:.2f} m/s")
    print(f"  Failure rate: {results['failure_rate']*100:.1f}%")
    
    # Compare with random baseline
    print("\nComparison with random policy:")
    print("  Random success rate: ~0%")
    print(f"  Your success rate: {results['success_rate']*100:.1f}%")
    
    if results['success_rate'] > 0.8:
        print("\n✅ Great! Your robot learned to walk!")
    elif results['success_rate'] > 0.5:
        print("\n📈 Good progress! Train longer for better results.")
    else:
        print("\n🔄 Needs more training. 10M steps is good, but might need more!")

def main():
    # Allow specifying experiment directory
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, help='Specific experiment directory')
    args = parser.parse_args()
    
    if args.exp:
        exp_dir = args.exp
    else:
        exp_dir = find_latest_model()
    
    if not exp_dir:
        return
    
    print(f"Found experiment: {exp_dir}")
    
    # Check what models exist - be flexible about what's available
    model_candidates = [
        os.path.join(exp_dir, "best_model", "best_model.zip"),
        os.path.join(exp_dir, "final_model.zip"),
        os.path.join(exp_dir, "checkpoints", "model_10000000_steps.zip"),
    ]
    
    # Also check for any checkpoint
    checkpoint_files = glob.glob(os.path.join(exp_dir, "checkpoints", "*.zip"))
    if checkpoint_files:
        # Get the latest checkpoint by step number
        try:
            checkpoint_files.sort(key=lambda x: int(x.split('_')[-2]))
            model_candidates.extend(checkpoint_files[-3:])  # Add last 3 checkpoints
        except:
            pass  # If filename format is different
    
    # Find first existing model
    model_to_test = None
    for candidate in model_candidates:
        if os.path.exists(candidate):
            model_to_test = candidate
            break
    
    if not model_to_test:
        print("No model found! Checked:")
        for c in model_candidates:
            print(f"  - {c}")
        return
    
    vec_normalize = os.path.join(exp_dir, "vec_normalize.pkl")
    
    # Run tests
    print("\n1. Basic Performance Test")
    test_model(model_to_test, vec_normalize, render=False, num_episodes=5)
    
    print("\n2. Success Metrics Evaluation") 
    evaluate_with_metrics(model_to_test, vec_normalize)
    
    print("\n3. Creating Video")
    video_path = os.path.join(exp_dir, "robot_demo.mp4")
    create_video(model_to_test, vec_normalize, video_path)
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"✓ Model tested: {model_to_test}")
    print(f"✓ Video saved: {video_path}")
    print(f"✓ Check W&B for training curves")
    
    # Check actual training steps from model
    try:
        model = PPO.load(model_to_test)
        if hasattr(model, 'num_timesteps'):
            print(f"\nModel trained for: {model.num_timesteps:,} steps")
    except:
        pass

if __name__ == "__main__":
    main()