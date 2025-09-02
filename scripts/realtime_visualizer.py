#!/usr/bin/env python3
"""
Real-time Performance Visualizer with Live Graphs
Shows robot walking with live performance metrics
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import time
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

class RealtimeVisualizer:
    def __init__(self, model_path, history_length=100):
        """
        Initialize real-time visualizer
        
        Args:
            model_path: Path to trained model
            history_length: How many timesteps to show in graphs
        """
        self.model_path = model_path
        self.history_length = history_length
        
        # Load model
        print(f"Loading model: {model_path}")
        self.model = PPO.load(model_path)
        
        # Create environment
        self.env = self._create_env()
        
        # Data storage for graphs
        self.timesteps = deque(maxlen=history_length)
        self.velocities = deque(maxlen=history_length)
        self.rewards = deque(maxlen=history_length)
        self.joint_angles = deque(maxlen=history_length)
        self.actions = deque(maxlen=history_length)
        self.distances = deque(maxlen=history_length)
        
        # Performance metrics
        self.total_distance = 0
        self.episode_reward = 0
        self.step_count = 0
        self.success_count = 0
        
        # Setup figure with subplots
        self.setup_plots()
        
    def _create_env(self):
        """Create environment with proper wrappers"""
        env = gym.make('RealAntMujoco-v0', render_mode='human')
        env = TargetWalkingWrapper(env, target_distance=5.0)
        env = DummyVecEnv([lambda: env])
        
        # Load vec_normalize if exists
        exp_dir = os.path.dirname(os.path.dirname(self.model_path))
        vec_norm_path = os.path.join(exp_dir, 'vec_normalize.pkl')
        if os.path.exists(vec_norm_path):
            env = VecNormalize.load(vec_norm_path, env)
            env.training = False
            env.norm_reward = False
            print("✅ Loaded normalization stats")
        
        return env
    
    def setup_plots(self):
        """Setup matplotlib figure with multiple subplots"""
        plt.style.use('dark_background')  # Cool dark theme
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.suptitle('Real-time Robot Performance Monitor', fontsize=16, color='cyan')
        
        # Create subplots
        # Top row: Velocity and Distance
        self.ax_velocity = plt.subplot(3, 3, 1)
        self.ax_distance = plt.subplot(3, 3, 2)
        self.ax_reward = plt.subplot(3, 3, 3)
        
        # Middle row: Joint angles and Actions
        self.ax_joints = plt.subplot(3, 3, 4)
        self.ax_actions = plt.subplot(3, 3, 5)
        self.ax_smoothness = plt.subplot(3, 3, 6)
        
        # Bottom row: Stats and Health
        self.ax_stats = plt.subplot(3, 1, 3)
        self.ax_stats.axis('off')
        
        # Initialize line plots
        self.line_velocity, = self.ax_velocity.plot([], [], 'g-', linewidth=2)
        self.line_distance, = self.ax_distance.plot([], [], 'b-', linewidth=2)
        self.line_reward, = self.ax_reward.plot([], [], 'y-', linewidth=2)
        
        # Joint angle lines (8 joints)
        self.joint_lines = []
        colors = plt.cm.rainbow(np.linspace(0, 1, 8))
        for i in range(8):
            line, = self.ax_joints.plot([], [], color=colors[i], alpha=0.7, label=f'J{i}')
        self.joint_lines.append(line)
        
        # Action lines
        self.action_lines = []
        for i in range(8):
            line, = self.ax_actions.plot([], [], color=colors[i], alpha=0.7)
            self.action_lines.append(line)
        
        # Smoothness line
        self.line_smoothness, = self.ax_smoothness.plot([], [], 'c-', linewidth=2)
        
        # Configure axes
        self.ax_velocity.set_title('Velocity (m/s)', color='green')
        self.ax_velocity.set_ylim(-0.5, 2.0)
        self.ax_velocity.grid(True, alpha=0.3)
        
        self.ax_distance.set_title('Distance Traveled (m)', color='blue')
        self.ax_distance.set_ylim(0, 10)
        self.ax_distance.grid(True, alpha=0.3)
        
        self.ax_reward.set_title('Instant Reward', color='yellow')
        self.ax_reward.set_ylim(-5, 10)
        self.ax_reward.grid(True, alpha=0.3)
        
        self.ax_joints.set_title('Joint Angles (rad)')
        self.ax_joints.set_ylim(-2, 2)
        self.ax_joints.grid(True, alpha=0.3)
        self.ax_joints.legend(loc='upper right', fontsize=8)
        
        self.ax_actions.set_title('Motor Commands')
        self.ax_actions.set_ylim(-1.2, 1.2)
        self.ax_actions.grid(True, alpha=0.3)
        
        self.ax_smoothness.set_title('Action Smoothness', color='cyan')
        self.ax_smoothness.set_ylim(0, 1)
        self.ax_smoothness.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    def update_plots(self, frame):
        """Update all plots with new data"""
        if not self.done:
            # Get action from model
            action, _ = self.model.predict(self.obs, deterministic=True)
            
            # Step environment
            self.obs, reward, self.done, info = self.env.step(action)
            
            # Extract data
            self.step_count += 1
            velocity = info[0].get('x_velocity', 0) if info[0] else 0
            
            # Store data
            self.timesteps.append(self.step_count)
            self.velocities.append(velocity)
            self.rewards.append(reward[0])
            self.total_distance += velocity * 0.05  # dt = 0.05
            self.distances.append(self.total_distance)
            self.episode_reward += reward[0]
            
            # Joint angles from observation (dims 13-20)
            if len(self.obs[0]) >= 21:
                joints = self.obs[0][13:21]
                self.joint_angles.append(joints)
            else:
                self.joint_angles.append(np.zeros(8))
            
            # Actions
            self.actions.append(action[0])
            
            # Calculate smoothness
            if len(self.actions) > 1:
                action_diff = np.abs(self.actions[-1] - self.actions[-2])
                smoothness = 1.0 / (1.0 + np.mean(action_diff))
            else:
                smoothness = 1.0
            
            # Update line plots
            if len(self.timesteps) > 1:
                self.line_velocity.set_data(self.timesteps, self.velocities)
                self.line_distance.set_data(self.timesteps, self.distances)
                self.line_reward.set_data(self.timesteps, self.rewards)
                
                # Update joint lines
                joint_data = np.array(self.joint_angles).T
                for i, line in enumerate(self.joint_lines):
                    if i < len(joint_data):
                        line.set_data(self.timesteps, joint_data[i])
                
                # Update action lines
                action_data = np.array(self.actions).T
                for i, line in enumerate(self.action_lines):
                    if i < len(action_data):
                        line.set_data(self.timesteps, action_data[i])
                
                # Update smoothness
                smoothness_history = [1.0 / (1.0 + np.mean(np.abs(self.actions[i] - self.actions[i-1]))) 
                                     for i in range(1, len(self.actions))]
                if smoothness_history:
                    self.line_smoothness.set_data(self.timesteps[1:], smoothness_history)
            
            # Adjust x-axis limits
            for ax in [self.ax_velocity, self.ax_distance, self.ax_reward, 
                      self.ax_joints, self.ax_actions, self.ax_smoothness]:
                ax.set_xlim(max(0, self.step_count - self.history_length), 
                           self.step_count + 10)
            
            # Update stats text
            self.ax_stats.clear()
            self.ax_stats.axis('off')
            
            stats_text = f"""
            🤖 ROBOT PERFORMANCE STATS 🤖
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━
            
            📊 Current Metrics:
            • Velocity: {velocity:.3f} m/s
            • Total Distance: {self.total_distance:.2f} m
            • Episode Reward: {self.episode_reward:.2f}
            • Steps: {self.step_count}
            • Smoothness: {smoothness:.3f}
            
            🎯 Target Status:
            • Success Count: {self.success_count}
            • Current Target: {info[0].get('target_x', 5.0):.1f} m
            
            🔧 Model: {os.path.basename(os.path.dirname(self.model_path))}
            """
            
            self.ax_stats.text(0.1, 0.5, stats_text, fontsize=12, 
                              family='monospace', color='white',
                              transform=self.ax_stats.transAxes)
            
            # Check for success
            if 'success' in info[0] and info[0]['success']:
                self.success_count += 1
        
        return [self.line_velocity, self.line_distance, self.line_reward, 
                self.line_smoothness] + self.joint_lines + self.action_lines
    
    def run(self, max_steps=2000):
        """Run the visualization"""
        # Reset environment
        self.obs = self.env.reset()
        self.done = False
        self.step_count = 0
        self.total_distance = 0
        self.episode_reward = 0
        
        # Create animation
        anim = FuncAnimation(self.fig, self.update_plots, 
                           interval=50,  # 20 FPS
                           blit=False, 
                           repeat=False,
                           frames=max_steps)
        
        plt.show()
        
        # Final summary
        print("\n" + "="*50)
        print("EPISODE COMPLETE!")
        print("="*50)
        print(f"Total Distance: {self.total_distance:.2f} m")
        print(f"Average Velocity: {np.mean(list(self.velocities)):.3f} m/s")
        print(f"Total Reward: {self.episode_reward:.2f}")
        print(f"Success Count: {self.success_count}")
        print(f"Steps Survived: {self.step_count}")
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description='Real-time Robot Performance Visualizer')
    parser.add_argument('model_path', type=str, help='Path to model')
    parser.add_argument('--steps', type=int, default=2000, help='Max steps to run')
    parser.add_argument('--history', type=int, default=100, help='Graph history length')
    
    args = parser.parse_args()
    
    visualizer = RealtimeVisualizer(args.model_path, args.history)
    visualizer.run(args.steps)

if __name__ == "__main__":
    main()