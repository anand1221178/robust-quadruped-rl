#!/usr/bin/env python3
"""
Side-by-side Model Comparison Tool
Compare two models running simultaneously under same conditions
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import argparse
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import environments
import realant_sim
from envs.target_walking_wrapper import TargetWalkingWrapper

class ModelComparator:
    def __init__(self, model1_path, model2_path, labels=None):
        """
        Initialize comparator with two models
        
        Args:
            model1_path: Path to first model
            model2_path: Path to second model  
            labels: Optional labels for models
        """
        self.model1_path = model1_path
        self.model2_path = model2_path
        
        # Set labels
        if labels:
            self.label1, self.label2 = labels
        else:
            self.label1 = os.path.basename(os.path.dirname(model1_path))
            self.label2 = os.path.basename(os.path.dirname(model2_path))
        
        # Load models
        print(f"Loading Model 1: {self.label1}")
        self.model1 = PPO.load(model1_path)
        
        print(f"Loading Model 2: {self.label2}")
        self.model2 = PPO.load(model2_path)
        
        # Create environments
        self.env1 = self._create_env(model1_path)
        self.env2 = self._create_env(model2_path)
        
    def _create_env(self, model_path):
        """Create environment with proper normalization"""
        env = gym.make('RealAntMujoco-v0')
        env = TargetWalkingWrapper(env, target_distance=5.0)
        env = DummyVecEnv([lambda: env])
        
        # Load normalization if exists
        exp_dir = os.path.dirname(os.path.dirname(model_path))
        vec_norm_path = os.path.join(exp_dir, 'vec_normalize.pkl')
        if os.path.exists(vec_norm_path):
            env = VecNormalize.load(vec_norm_path, env)
            env.training = False
            env.norm_reward = False
        
        return env
    
    def compare_episodes(self, num_episodes=5, max_steps=1000, add_noise=False, noise_level=0.1):
        """
        Run episodes for both models and compare
        
        Args:
            num_episodes: Number of episodes to run
            max_steps: Max steps per episode
            add_noise: Whether to add sensor noise
            noise_level: Noise standard deviation
        """
        results = {
            'model1': {'velocities': [], 'distances': [], 'rewards': [], 'successes': 0},
            'model2': {'velocities': [], 'distances': [], 'rewards': [], 'successes': 0}
        }
        
        print(f"\n{'='*70}")
        print(f"SIDE-BY-SIDE COMPARISON: {self.label1} vs {self.label2}")
        print(f"{'='*70}")
        
        if add_noise:
            print(f"🔊 Adding {noise_level*100:.0f}% sensor noise to both models")
        
        for episode in range(num_episodes):
            print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
            
            # Reset both environments
            obs1 = self.env1.reset()
            obs2 = self.env2.reset()
            
            # Episode metrics
            ep1_metrics = {'velocity': [], 'distance': 0, 'reward': 0}
            ep2_metrics = {'velocity': [], 'distance': 0, 'reward': 0}
            
            for step in range(max_steps):
                # Add noise if requested
                if add_noise:
                    noise1 = np.random.normal(0, noise_level, obs1.shape)
                    noise2 = np.random.normal(0, noise_level, obs2.shape)
                    obs1_noisy = obs1 + noise1
                    obs2_noisy = obs2 + noise2
                else:
                    obs1_noisy = obs1
                    obs2_noisy = obs2
                
                # Get actions
                action1, _ = self.model1.predict(obs1_noisy, deterministic=True)
                action2, _ = self.model2.predict(obs2_noisy, deterministic=True)
                
                # Step environments
                obs1, reward1, done1, info1 = self.env1.step(action1)
                obs2, reward2, done2, info2 = self.env2.step(action2)
                
                # Extract velocities - TargetWalkingWrapper uses 'speed'
                vel1 = info1[0].get('speed', info1[0].get('x_velocity', 0)) if info1[0] else 0
                vel2 = info2[0].get('speed', info2[0].get('x_velocity', 0)) if info2[0] else 0
                
                # Update metrics
                ep1_metrics['velocity'].append(vel1)
                ep1_metrics['distance'] += abs(vel1) * 0.05
                ep1_metrics['reward'] += reward1[0]
                
                ep2_metrics['velocity'].append(vel2)
                ep2_metrics['distance'] += abs(vel2) * 0.05
                ep2_metrics['reward'] += reward2[0]
                
                # Print progress every 200 steps
                if step % 200 == 0 and step > 0:
                    avg_vel1 = np.mean(ep1_metrics['velocity'][-100:])
                    avg_vel2 = np.mean(ep2_metrics['velocity'][-100:])
                    print(f"  Step {step}: {self.label1}={avg_vel1:.3f} m/s, "
                          f"{self.label2}={avg_vel2:.3f} m/s")
                
                # Check for success
                if info1[0] and info1[0].get('success', False):
                    results['model1']['successes'] += 1
                if info2[0] and info2[0].get('success', False):
                    results['model2']['successes'] += 1
                
                if done1 or done2:
                    break
            
            # Episode summary
            avg_vel1 = np.mean(ep1_metrics['velocity'])
            avg_vel2 = np.mean(ep2_metrics['velocity'])
            
            print(f"\nEpisode {episode + 1} Results:")
            print(f"  {self.label1:20s}: Vel={avg_vel1:.3f} m/s, "
                  f"Dist={ep1_metrics['distance']:.2f} m, "
                  f"Reward={ep1_metrics['reward']:.1f}")
            print(f"  {self.label2:20s}: Vel={avg_vel2:.3f} m/s, "
                  f"Dist={ep2_metrics['distance']:.2f} m, "
                  f"Reward={ep2_metrics['reward']:.1f}")
            
            # Determine winner
            if avg_vel1 > avg_vel2 * 1.1:
                print(f"  🏆 {self.label1} wins this episode!")
            elif avg_vel2 > avg_vel1 * 1.1:
                print(f"  🏆 {self.label2} wins this episode!")
            else:
                print(f"  🤝 Too close to call!")
            
            # Store results
            results['model1']['velocities'].append(avg_vel1)
            results['model1']['distances'].append(ep1_metrics['distance'])
            results['model1']['rewards'].append(ep1_metrics['reward'])
            
            results['model2']['velocities'].append(avg_vel2)
            results['model2']['distances'].append(ep2_metrics['distance'])
            results['model2']['rewards'].append(ep2_metrics['reward'])
        
        # Final comparison
        self._print_final_comparison(results)
        self._plot_comparison(results)
        
        return results
    
    def _print_final_comparison(self, results):
        """Print final comparison statistics"""
        print(f"\n{'='*70}")
        print(f"FINAL COMPARISON RESULTS")
        print(f"{'='*70}")
        
        # Calculate averages
        avg_vel1 = np.mean(results['model1']['velocities'])
        avg_vel2 = np.mean(results['model2']['velocities'])
        
        avg_dist1 = np.mean(results['model1']['distances'])
        avg_dist2 = np.mean(results['model2']['distances'])
        
        avg_reward1 = np.mean(results['model1']['rewards'])
        avg_reward2 = np.mean(results['model2']['rewards'])
        
        # Print comparison
        print(f"\n📊 Average Velocity:")
        print(f"  {self.label1:20s}: {avg_vel1:.3f} ± {np.std(results['model1']['velocities']):.3f} m/s")
        print(f"  {self.label2:20s}: {avg_vel2:.3f} ± {np.std(results['model2']['velocities']):.3f} m/s")
        
        print(f"\n📏 Average Distance:")
        print(f"  {self.label1:20s}: {avg_dist1:.2f} m")
        print(f"  {self.label2:20s}: {avg_dist2:.2f} m")
        
        print(f"\n🎯 Success Count:")
        print(f"  {self.label1:20s}: {results['model1']['successes']} targets reached")
        print(f"  {self.label2:20s}: {results['model2']['successes']} targets reached")
        
        print(f"\n💰 Average Reward:")
        print(f"  {self.label1:20s}: {avg_reward1:.2f}")
        print(f"  {self.label2:20s}: {avg_reward2:.2f}")
        
        # Determine overall winner
        print(f"\n{'='*70}")
        score1 = 0
        score2 = 0
        
        if avg_vel1 > avg_vel2:
            score1 += 1
        else:
            score2 += 1
            
        if avg_dist1 > avg_dist2:
            score1 += 1
        else:
            score2 += 1
            
        if results['model1']['successes'] > results['model2']['successes']:
            score1 += 1
        else:
            score2 += 1
        
        if score1 > score2:
            print(f"🏆 OVERALL WINNER: {self.label1}")
        elif score2 > score1:
            print(f"🏆 OVERALL WINNER: {self.label2}")
        else:
            print(f"🤝 IT'S A TIE!")
        print(f"{'='*70}\n")
    
    def _plot_comparison(self, results):
        """Create comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Model Comparison: {self.label1} vs {self.label2}', fontsize=14)
        
        episodes = range(1, len(results['model1']['velocities']) + 1)
        
        # Velocity comparison
        axes[0, 0].plot(episodes, results['model1']['velocities'], 'b-o', label=self.label1)
        axes[0, 0].plot(episodes, results['model2']['velocities'], 'r-s', label=self.label2)
        axes[0, 0].set_title('Average Velocity per Episode')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Velocity (m/s)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Distance comparison
        axes[0, 1].plot(episodes, results['model1']['distances'], 'b-o', label=self.label1)
        axes[0, 1].plot(episodes, results['model2']['distances'], 'r-s', label=self.label2)
        axes[0, 1].set_title('Distance Traveled per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Distance (m)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Reward comparison
        axes[1, 0].plot(episodes, results['model1']['rewards'], 'b-o', label=self.label1)
        axes[1, 0].plot(episodes, results['model2']['rewards'], 'r-s', label=self.label2)
        axes[1, 0].set_title('Total Reward per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Bar chart comparison
        metrics = ['Avg Velocity\n(m/s)', 'Avg Distance\n(m)', 'Success Rate\n(%)']
        model1_vals = [
            np.mean(results['model1']['velocities']),
            np.mean(results['model1']['distances']),
            results['model1']['successes'] / len(results['model1']['velocities']) * 100
        ]
        model2_vals = [
            np.mean(results['model2']['velocities']),
            np.mean(results['model2']['distances']),
            results['model2']['successes'] / len(results['model2']['velocities']) * 100
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, model1_vals, width, label=self.label1, color='blue', alpha=0.7)
        axes[1, 1].bar(x + width/2, model2_vals, width, label=self.label2, color='red', alpha=0.7)
        axes[1, 1].set_title('Overall Performance Comparison')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Compare two models side-by-side')
    parser.add_argument('model1', type=str, help='Path to first model')
    parser.add_argument('model2', type=str, help='Path to second model')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes')
    parser.add_argument('--steps', type=int, default=1000, help='Max steps per episode')
    parser.add_argument('--noise', action='store_true', help='Add sensor noise')
    parser.add_argument('--noise-level', type=float, default=0.1, help='Noise level (0-1)')
    parser.add_argument('--label1', type=str, help='Label for model 1')
    parser.add_argument('--label2', type=str, help='Label for model 2')
    
    args = parser.parse_args()
    
    labels = None
    if args.label1 and args.label2:
        labels = (args.label1, args.label2)
    
    comparator = ModelComparator(args.model1, args.model2, labels)
    comparator.compare_episodes(
        num_episodes=args.episodes,
        max_steps=args.steps,
        add_noise=args.noise,
        noise_level=args.noise_level
    )

if __name__ == "__main__":
    main()