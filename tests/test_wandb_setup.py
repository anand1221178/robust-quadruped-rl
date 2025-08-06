#!/usr/bin/env python3
"""
Test Weights & Biases setup
"""

import wandb
import numpy as np
import gymnasium as gym
import time

def test_basic_wandb():
    """Test basic W&B functionality"""
    print("=== Testing Basic W&B ===")
    
    # Initialize a test run
    run = wandb.init(
        project="robust-quadruped-test",  # Test project
        name="test-run",
        config={
            "learning_rate": 0.001,
            "batch_size": 32,
            "architecture": "MLP",
        }
    )
    
    # Log some test metrics
    for i in range(10):
        wandb.log({
            "loss": np.random.random() * np.exp(-i * 0.1),
            "accuracy": np.random.random() * i / 10,
            "step": i
        })
        time.sleep(0.1)
    
    # Finish run
    wandb.finish()
    
    print("✓ Basic W&B logging works!")
    print(f"View your run at: {run.url}")

def test_gym_with_wandb():
    """Test W&B with Gym environment"""
    print("\n=== Testing W&B with Gym ===")
    
    # Initialize run with more config
    config = {
        "env_name": "Ant-v4",
        "num_episodes": 3,
        "max_steps": 100,
    }
    
    run = wandb.init(
        project="robust-quadruped-test",
        name="ant-env-test",
        config=config,
        tags=["test", "ant-v4"],
    )
    
    # Create environment
    env = gym.make('Ant-v4')
    
    # Run a few episodes
    for episode in range(config["num_episodes"]):
        obs, info = env.reset()
        episode_reward = 0
        
        for step in range(config["max_steps"]):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Log step metrics
            wandb.log({
                "step_reward": reward,
                "episode": episode,
                "global_step": episode * config["max_steps"] + step,
            })
            
            if terminated or truncated:
                break
        
        # Log episode metrics
        wandb.log({
            "episode_reward": episode_reward,
            "episode_length": step + 1,
            "episode": episode,
        })
        
        print(f"Episode {episode}: Reward = {episode_reward:.2f}")
    
    env.close()
    wandb.finish()
    
    print("✓ W&B with Gym environment works!")
    print(f"View your run at: {run.url}")

def test_wandb_video():
    """Test logging videos to W&B"""
    print("\n=== Testing W&B Video Logging ===")
    
    try:
        import cv2
        
        run = wandb.init(
            project="robust-quadruped-test",
            name="video-test",
        )
        
        # Create environment with rendering
        env = gym.make('Ant-v4', render_mode='rgb_array')
        
        # Record a short clip
        frames = []
        obs, info = env.reset()
        
        for _ in range(100):  # 100 frames
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            frame = env.render()
            frames.append(frame)
            
            if terminated or truncated:
                break
        
        # Log video to W&B
        wandb.log({
            "random_policy_video": wandb.Video(
                np.array(frames), 
                fps=30, 
                format="mp4"
            )
        })
        
        env.close()
        wandb.finish()
        
        print("✓ Video logging works!")
        
    except ImportError:
        print("⚠️  OpenCV not installed, skipping video test")
        print("   Install with: pip install opencv-python")

def check_wandb_config():
    """Check if W&B is properly configured"""
    print("=== W&B Configuration Check ===")
    
    # Check if logged in
    if wandb.api.api_key:
        print("✓ W&B API key found")
        print(f"✓ Logged in as: {wandb.api.viewer()['username']}")
    else:
        print("✗ Not logged in to W&B")
        print("  Run: wandb login")
        return False
    
    # Check project settings
    print("\nYour W&B setup:")
    print(f"  Default entity: {wandb.api.default_entity}")
    print(f"  API key: {'*' * 30}{wandb.api.api_key[-6:]}")
    
    return True

def main():
    print("=" * 60)
    print("Weights & Biases Setup Test")
    print("=" * 60)
    
    # Check configuration
    if not check_wandb_config():
        print("\n⚠️  Please run 'wandb login' first!")
        return
    
    print("\nRunning tests...")
    
    # Run tests
    test_basic_wandb()
    test_gym_with_wandb()
    test_wandb_video()
    
    print("\n" + "=" * 60)
    print("✅ W&B setup complete!")
    print("\nNext steps:")
    print("1. Go to https://wandb.ai/your-username/robust-quadruped-test")
    print("2. You should see your test runs")
    print("3. Update configs/train/default.yaml with wandb: true")
    print("4. Start training with W&B logging enabled!")

if __name__ == "__main__":
    main()