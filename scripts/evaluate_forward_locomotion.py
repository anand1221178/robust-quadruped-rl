#!/usr/bin/env python3
"""
Evaluate Forward Locomotion Models
Simple evaluation for baseline, SR2L, and DR models
"""
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
from envs.domain_randomization_wrapper import DomainRandomizationWrapper
from envs.target_walking_wrapper import TargetWalkingWrapper
import realant_sim
import argparse

def evaluate_model(model_path, vec_normalize_path, model_name, use_dr=False, dr_config=None, episodes=5):
    """Evaluate a single model"""
    print(f"\n📊 EVALUATING {model_name}")
    print("=" * 50)
    
    def make_env():
        env = gym.make('RealAntMujoco-v0')
        # Use TargetWalkingWrapper to match the working video script
        env = TargetWalkingWrapper(env, target_distance=5.0)
        
        if use_dr and dr_config:
            env = DomainRandomizationWrapper(env, dr_config)
            print(f"  🎲 DR enabled: {dr_config.get('joint_failure_prob', 0.1)} failure rate")
        
        return env
    
    env = DummyVecEnv([make_env])
    
    try:
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
        print("   VecNormalize loaded")
    except Exception as e:
        print(f"  ⚠️  VecNormalize failed: {e}")
    
    model = PPO.load(model_path)
    print(f"   Model loaded")
    
    velocities = []
    distances = []
    rewards = []
    falls = []
    
    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0
        positions = []
        fell = False
        
        for step in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            
            if done[0]:
                print(f"    Episode ended at step {step} (fell: {fell})")
                break
            
            # Track position (only if episode is still running)
            x_pos = env.envs[0].unwrapped.data.qpos[0]
            positions.append(x_pos)
            
            # Debug what's happening - check more steps
            if step < 5 or step % 200 == 0:
                print(f"    Step {step}: x={x_pos:.3f}, reward={reward[0]:.1f}, done={done[0]}")
                # Also debug positions array
                if step % 200 == 0:
                    print(f"    DEBUG: positions array length={len(positions)}, last few values={positions[-3:]}")
        
        # Calculate metrics
        if len(positions) >= 2:
            total_distance = positions[-1] - positions[0]
            total_time = (len(positions) - 1) * 0.05  # Time in seconds
            avg_velocity = total_distance / total_time if total_time > 0 else 0
            
            # Debug the calculation
            print(f"  DEBUG: positions length={len(positions)}")
            print(f"  DEBUG: first 5 positions={positions[:5]}")
            print(f"  DEBUG: last 5 positions={positions[-5:]}")
            print(f"  DEBUG: start_pos={positions[0]:.3f}, end_pos={positions[-1]:.3f}, distance={total_distance:.3f}, time={total_time:.1f}s")
            
            velocities.append(avg_velocity)
            distances.append(total_distance)
            rewards.append(episode_reward)
            falls.append(fell)
        else:
            avg_velocity = 0.0
            total_distance = 0.0
        
        print(f"  Episode {episode+1}: {avg_velocity:.3f} m/s, {total_distance:.1f}m, fell: {fell}")
    
    env.close()
    
    # Summary
    avg_vel = np.mean(velocities)
    std_vel = np.std(velocities)
    avg_dist = np.mean(distances)
    avg_reward = np.mean(rewards)
    fall_rate = np.mean(falls) * 100
    
    print(f"\n📈 RESULTS:")
    print(f"  Velocity: {avg_vel:.3f} ± {std_vel:.3f} m/s")
    print(f"  Distance: {avg_dist:.1f} m/episode")
    print(f"  Reward: {avg_reward:.0f}")
    print(f"  Fall Rate: {fall_rate:.1f}%")
    
    return {
        'name': model_name,
        'velocity': avg_vel,
        'velocity_std': std_vel,
        'distance': avg_dist,
        'reward': avg_reward,
        'fall_rate': fall_rate
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, default='done/ppo_baseline_ueqbjf2x')
    parser.add_argument('--sr2l', type=str, help='Path to SR2L model')
    parser.add_argument('--dr', type=str, help='Path to DR model')
    parser.add_argument('--episodes', type=int, default=5)
    args = parser.parse_args()
    
    print("🚀 FORWARD LOCOMOTION EVALUATION")
    print("Testing models for forward walking performance")
    print("=" * 60)
    
    results = []
    
    # Evaluate baseline
    if args.baseline:
        result = evaluate_model(
            f"{args.baseline}/best_model/best_model.zip",
            f"{args.baseline}/vec_normalize.pkl",
            "Baseline PPO",
            episodes=args.episodes
        )
        results.append(result)
    
    # Evaluate SR2L
    if args.sr2l:
        result = evaluate_model(
            f"{args.sr2l}/final_model.zip",
            f"{args.sr2l}/vec_normalize.pkl", 
            "SR2L",
            episodes=args.episodes
        )
        results.append(result)
    
    # Evaluate DR
    if args.dr:
        result = evaluate_model(
            f"{args.dr}/final_model.zip",
            f"{args.dr}/vec_normalize.pkl",
            "Domain Randomization", 
            episodes=args.episodes
        )
        results.append(result)
    
    # Final comparison
    if len(results) > 1:
        print(f"\n🏆 FINAL COMPARISON:")
        print("=" * 60)
        for result in results:
            print(f"{result['name']:20}: {result['velocity']:.3f} m/s, {result['fall_rate']:.1f}% falls")
        
        baseline_vel = results[0]['velocity']
        print(f"\nRetention vs Baseline:")
        for result in results[1:]:
            retention = (result['velocity'] / baseline_vel) * 100
            print(f"  {result['name']:20}: {retention:.1f}% retention")

if __name__ == "__main__":
    main()