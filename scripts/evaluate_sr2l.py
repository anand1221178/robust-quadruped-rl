#!/usr/bin/env python3
"""
SR2L Evaluation Script

Tests the effectiveness of SR2L by comparing:
1. Action smoothness between PPO and PPO+SR2L
2. Robustness to sensor noise
3. Performance under different noise conditions

Usage:
    python scripts/evaluate_sr2l.py --baseline_model path/to/ppo_model --sr2l_model path/to/sr2l_model
"""

import os
import sys
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import glob
from typing import Dict, List, Tuple, Optional
import argparse
from pathlib import Path

# Add src to path
sys.path.append('src')
import realant_sim
from stable_baselines3 import PPO
from agents.ppo_sr2l import PPO_SR2L
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.target_walking_wrapper import TargetWalkingWrapper
from envs.success_reward_wrapper import SuccessRewardWrapper


def create_correct_env(render_mode=None, use_target_walking=True):
    """Create environment with correct wrapper based on model type"""
    env = gym.make('RealAntMujoco-v0', render_mode=render_mode)
    
    if use_target_walking:
        env = TargetWalkingWrapper(env, target_distance=5.0)
    else:
        env = SuccessRewardWrapper(env)
    
    return env


class SensorNoiseWrapper(gym.Wrapper):
    """
    Wrapper to inject realistic sensor noise into RealAnt observations
    Following research proposal Section 3.3
    """
    
    def __init__(self, env, noise_config: Optional[Dict] = None):
        super().__init__(env)
        
        # Default noise levels from proposal
        default_config = {
            'position_std': 0.05,     # Body position/height noise
            'velocity_std': 0.1,      # Velocity noise  
            'orientation_std': 0.02,  # Orientation noise
            'joint_std': 0.05,        # Joint sensor noise
            'apply_noise': True
        }
        
        self.noise_config = {**default_config, **(noise_config or {})}
        
        # RealAnt observation indices (29D total)
        self.velocity_indices = list(range(0, 3)) + list(range(4, 7)) + list(range(21, 29))  # body_vel + angular_vel + joint_vel
        self.position_indices = [3]  # body_z_pos
        self.orientation_indices = list(range(7, 13))  # sin/cos orientation
        self.joint_indices = list(range(13, 21))  # joint positions
        
        print(f"Sensor noise enabled: pos_σ={self.noise_config['position_std']}, "
              f"vel_σ={self.noise_config['velocity_std']}, "
              f"ori_σ={self.noise_config['orientation_std']}, "
              f"joint_σ={self.noise_config['joint_std']}")
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.noise_config['apply_noise']:
            obs = self._add_sensor_noise(obs)
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        if self.noise_config['apply_noise']:
            obs = self._add_sensor_noise(obs)
        
        return obs, info
    
    def _add_sensor_noise(self, obs):
        """Add Gaussian noise to different sensor groups"""
        noisy_obs = obs.copy()
        
        # Position noise (height sensors)
        if self.noise_config['position_std'] > 0:
            position_noise = np.random.normal(0, self.noise_config['position_std'], 
                                            len(self.position_indices))
            noisy_obs[self.position_indices] += position_noise
        
        # Velocity noise (IMU, encoders)
        if self.noise_config['velocity_std'] > 0:
            velocity_noise = np.random.normal(0, self.noise_config['velocity_std'],
                                            len(self.velocity_indices))
            noisy_obs[self.velocity_indices] += velocity_noise
        
        # Orientation noise (IMU)
        if self.noise_config['orientation_std'] > 0:
            orientation_noise = np.random.normal(0, self.noise_config['orientation_std'],
                                               len(self.orientation_indices))
            noisy_obs[self.orientation_indices] += orientation_noise
            
            # Normalize quaternion representation (sin/cos should stay in valid range)
            # Clamp to reasonable bounds to maintain valid trigonometric values
            noisy_obs[self.orientation_indices] = np.clip(noisy_obs[self.orientation_indices], -1.5, 1.5)
        
        # Joint sensor noise (encoders)
        if self.noise_config['joint_std'] > 0:
            joint_noise = np.random.normal(0, self.noise_config['joint_std'],
                                         len(self.joint_indices))
            noisy_obs[self.joint_indices] += joint_noise
        
        return noisy_obs


def find_latest_model(experiment_pattern: str) -> Tuple[str, str]:
    """
    Find the latest/best model for a given experiment pattern
    
    Args:
        experiment_pattern: Pattern like "experiments/ppo_baseline_*" or "experiments/ppo_sr2l_*"
    
    Returns:
        Tuple of (model_path, vec_normalize_path)
    """
    experiment_dirs = glob.glob(experiment_pattern)
    
    if not experiment_dirs:
        raise FileNotFoundError(f"No experiments found matching pattern: {experiment_pattern}")
    
    # Get the most recent experiment directory
    latest_exp = max(experiment_dirs, key=os.path.getmtime)
    print(f"Found latest experiment: {latest_exp}")
    
    # Look for best model first, then final model, then latest checkpoint
    model_candidates = [
        os.path.join(latest_exp, "best_model", "best_model.zip"),
        os.path.join(latest_exp, "final_model.zip"),
    ]
    
    # Add any checkpoints
    checkpoint_pattern = os.path.join(latest_exp, "checkpoints", "*.zip")
    checkpoints = glob.glob(checkpoint_pattern)
    if checkpoints:
        # Sort by step number and add the latest
        try:
            checkpoints.sort(key=lambda x: int(x.split('_')[-2]) if '_' in x else 0)
            model_candidates.extend(checkpoints[-3:])
        except:
            model_candidates.extend(checkpoints)
    
    # Find first existing model
    model_path = None
    for candidate in model_candidates:
        if os.path.exists(candidate):
            model_path = candidate
            break
    
    if not model_path:
        raise FileNotFoundError(f"No model found in {latest_exp}. Checked: {model_candidates}")
    
    # Look for VecNormalize
    vec_normalize_path = os.path.join(latest_exp, "vec_normalize.pkl")
    if not os.path.exists(vec_normalize_path):
        vec_normalize_path = None
    
    print(f"Using model: {model_path}")
    if vec_normalize_path:
        print(f"Using normalization: {vec_normalize_path}")
    else:
        print("No VecNormalize found")
    
    return model_path, vec_normalize_path


class SR2LEvaluator:
    """
    Comprehensive SR2L evaluation suite
    """
    
    def __init__(self, baseline_model_path: str, sr2l_model_path: str, 
                 vec_normalize_baseline: str = None, vec_normalize_sr2l: str = None):
        
        self.baseline_model_path = baseline_model_path
        self.sr2l_model_path = sr2l_model_path
        
        # Load models
        print("Loading baseline PPO model...")
        self.baseline_model = PPO.load(baseline_model_path)
        
        print("Loading SR2L model...")
        self.sr2l_model = PPO_SR2L.load(sr2l_model_path)
        
        # Load normalization if available
        self.vec_normalize_baseline = None
        self.vec_normalize_sr2l = None
        
        if vec_normalize_baseline and os.path.exists(vec_normalize_baseline):
            print("Loading baseline VecNormalize...")
            dummy_env = DummyVecEnv([lambda: create_correct_env(use_target_walking=True)])
            self.vec_normalize_baseline = VecNormalize.load(vec_normalize_baseline, dummy_env)
            self.vec_normalize_baseline.training = False
            dummy_env.close()
            
        if vec_normalize_sr2l and os.path.exists(vec_normalize_sr2l):
            print("Loading SR2L VecNormalize...")
            dummy_env = DummyVecEnv([lambda: create_correct_env(use_target_walking=True)])
            self.vec_normalize_sr2l = VecNormalize.load(vec_normalize_sr2l, dummy_env)
            self.vec_normalize_sr2l.training = False
            dummy_env.close()
    
    def measure_action_smoothness(self, model, env, n_episodes: int = 10) -> Dict[str, float]:
        """
        Measure how smooth the policy's actions are over time
        """
        print(f"Measuring action smoothness over {n_episodes} episodes...")
        
        action_changes = []
        action_magnitudes = []
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            prev_action = None
            episode_actions = []
            
            for step in range(1000):  # RealAnt max episode length
                action, _ = model.predict(obs, deterministic=True)
                episode_actions.append(action.copy())
                
                if prev_action is not None:
                    # L2 distance between consecutive actions
                    action_change = np.linalg.norm(action - prev_action)
                    action_changes.append(action_change)
                
                action_magnitudes.append(np.linalg.norm(action))
                
                obs, _, done, _, _ = env.step(action)
                prev_action = action
                
                if done:
                    break
        
        return {
            'mean_action_change': np.mean(action_changes),
            'std_action_change': np.std(action_changes),
            'max_action_change': np.max(action_changes),
            'mean_action_magnitude': np.mean(action_magnitudes),
            'action_jerkiness_score': np.std(action_changes) / (np.mean(action_changes) + 1e-8)
        }
    
    def test_noise_robustness(self, model, base_env, noise_levels: List[float] = None, 
                            n_episodes: int = 100) -> Dict[float, Dict[str, float]]:
        """
        Test policy performance under different sensor noise levels
        """
        if noise_levels is None:
            noise_levels = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
        
        print(f"Testing noise robustness across {len(noise_levels)} noise levels...")
        
        results = {}
        
        for noise_std in noise_levels:
            print(f"  Testing with noise std = {noise_std:.3f}")
            
            # Create noisy environment
            noise_config = {
                'position_std': noise_std,
                'velocity_std': noise_std * 2,  # Velocity sensors typically noisier
                'orientation_std': noise_std * 0.4,  # Orientation more precise
                'joint_std': noise_std,
                'apply_noise': noise_std > 0
            }
            
            noisy_env = SensorNoiseWrapper(base_env, noise_config)
            
            # Run evaluation episodes
            episode_rewards = []
            episode_lengths = []
            success_count = 0
            
            for episode in range(n_episodes):
                obs, _ = noisy_env.reset()
                total_reward = 0
                steps = 0
                
                for step in range(1000):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, _, info = noisy_env.step(action)
                    
                    total_reward += reward
                    steps += 1
                    
                    if done:
                        break
                
                episode_rewards.append(total_reward)
                episode_lengths.append(steps)
                
                # Success criteria: episode length > 500 and positive reward
                if steps >= 500 and total_reward > 0:
                    success_count += 1
            
            results[noise_std] = {
                'success_rate': success_count / n_episodes,
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'mean_length': np.mean(episode_lengths),
                'std_length': np.std(episode_lengths)
            }
        
        return results
    
    def record_comparison_videos(self, env, output_dir: str, max_steps: int = 1000, fps: int = 30):
        """
        Record videos comparing baseline vs SR2L under different conditions
        """
        print("Recording comparison videos...")
        
        videos_to_create = [
            # (model, model_name, noise_std, filename, description)
            (self.baseline_model, "Baseline PPO", 0.0, "baseline_clean.mp4", "clean environment"),
            (self.sr2l_model, "PPO + SR2L", 0.0, "sr2l_clean.mp4", "clean environment"),
            (self.baseline_model, "Baseline PPO", 0.05, "baseline_noisy.mp4", "noisy sensors (σ=0.05)"),
            (self.sr2l_model, "PPO + SR2L", 0.05, "sr2l_noisy.mp4", "noisy sensors (σ=0.05)"),
            (self.baseline_model, "Baseline PPO", 0.1, "baseline_very_noisy.mp4", "very noisy sensors (σ=0.1)"),
            (self.sr2l_model, "PPO + SR2L", 0.1, "sr2l_very_noisy.mp4", "very noisy sensors (σ=0.1)"),
        ]
        
        for model, model_name, noise_std, filename, description in videos_to_create:
            print(f"  Recording {model_name} with {description}...")
            
            # Setup environment with appropriate noise
            if noise_std > 0:
                noise_config = {
                    'position_std': noise_std,
                    'velocity_std': noise_std * 2,
                    'orientation_std': noise_std * 0.4,
                    'joint_std': noise_std,
                    'apply_noise': True
                }
                test_env = SensorNoiseWrapper(env, noise_config)
            else:
                test_env = env
            
            # Create video
            video_path = os.path.join(output_dir, filename)
            self._record_single_video(model, test_env, video_path, model_name, description, max_steps, fps)
        
        # Create side-by-side comparison video
        self._create_comparison_grid(output_dir)
        
        print(f"Videos saved to: {output_dir}/")
    
    def _record_single_video(self, model, env, video_path: str, model_name: str, 
                           description: str, max_steps: int, fps: int):
        """Record a single episode video with metrics"""
        
        frames = []
        obs, _ = env.reset()
        
        # Track metrics for display
        velocities = []
        total_reward = 0
        action_changes = []
        prev_action = None
        
        for step in range(max_steps):
            # Render frame
            frame = env.render()
            
            # Get action and step
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            
            # Track metrics
            total_reward += reward
            # Get velocity from wrapper info (correct way)
            if 'current_velocity' in info:
                velocity = abs(info['current_velocity'])  # Use wrapper's velocity calculation
            else:
                velocity = 0  # Fallback
            velocities.append(velocity)
            
            if prev_action is not None:
                action_change = np.linalg.norm(action - prev_action)
                action_changes.append(action_change)
            prev_action = action.copy()
            
            # Calculate smoothness metric
            avg_smoothness = np.mean(action_changes) if action_changes else 0
            
            if frame is not None:
                # Add enhanced text overlay with metrics
                metrics = {
                    'velocity': velocity,
                    'avg_velocity': np.mean(velocities),
                    'total_reward': total_reward,
                    'action_smoothness': avg_smoothness
                }
                frame_with_text = self._add_text_overlay_with_metrics(
                    frame, model_name, description, step, metrics
                )
                frames.append(frame_with_text)
            
            if done:
                break
        
        # Save video
        if frames:
            self._save_video(frames, video_path, fps)
            print(f"    Saved: {video_path} ({len(frames)} frames)")
            print(f"      Final metrics: Avg velocity={np.mean(velocities):.2f}, Total reward={total_reward:.1f}")
        else:
            print(f"    Failed to record: {video_path} (no frames captured)")
    
    def _add_text_overlay_with_metrics(self, frame: np.ndarray, model_name: str, description: str, 
                                     step: int, metrics: dict) -> np.ndarray:
        """Add enhanced text overlay with sensor noise info and metrics"""
        frame_with_text = frame.copy()
        height, width = frame.shape[:2]
        
        # Add text background (larger for more metrics)
        overlay = frame_with_text.copy()
        cv2.rectangle(overlay, (10, 10), (width - 10, 150), (0, 0, 0), -1)
        frame_with_text = cv2.addWeighted(frame_with_text, 0.7, overlay, 0.3, 0)
        
        # Parse sensor noise info from description
        sensor_info = self._parse_sensor_noise_description(description)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_pos = 30
        
        # Model and condition
        cv2.putText(frame_with_text, f"Model: {model_name}", (20, y_pos), 
                   font, 0.7, (255, 255, 255), 2)
        y_pos += 20
        
        cv2.putText(frame_with_text, f"Condition: {sensor_info['condition']}", (20, y_pos), 
                   font, 0.5, (255, 255, 255), 1)
        y_pos += 20
        
        # Sensor noise details
        if sensor_info['has_noise']:
            cv2.putText(frame_with_text, f"Sensor Noise:", (20, y_pos), 
                       font, 0.5, (255, 100, 100), 1)
            y_pos += 15
            cv2.putText(frame_with_text, f"  Position: σ={sensor_info['pos_noise']:.3f}", (20, y_pos), 
                       font, 0.4, (200, 200, 200), 1)
            y_pos += 12
            cv2.putText(frame_with_text, f"  Velocity: σ={sensor_info['vel_noise']:.3f}", (20, y_pos), 
                       font, 0.4, (200, 200, 200), 1)
            y_pos += 12
            cv2.putText(frame_with_text, f"  Orientation: σ={sensor_info['ori_noise']:.3f}", (20, y_pos), 
                       font, 0.4, (200, 200, 200), 1)
        else:
            cv2.putText(frame_with_text, "Sensor Noise: None", (20, y_pos), 
                       font, 0.5, (100, 255, 100), 1)
        
        # Real-time metrics (right side)
        right_x = width - 200
        y_pos = 30
        
        cv2.putText(frame_with_text, f"Step: {step}", (right_x, y_pos), 
                   font, 0.5, (255, 255, 255), 1)
        y_pos += 20
        
        cv2.putText(frame_with_text, f"Velocity: {metrics['velocity']:.2f} m/s", (right_x, y_pos), 
                   font, 0.4, (100, 255, 255), 1)
        y_pos += 15
        
        cv2.putText(frame_with_text, f"Avg Vel: {metrics['avg_velocity']:.2f} m/s", (right_x, y_pos), 
                   font, 0.4, (100, 255, 255), 1)
        y_pos += 15
        
        cv2.putText(frame_with_text, f"Reward: {metrics['total_reward']:.1f}", (right_x, y_pos), 
                   font, 0.4, (255, 255, 100), 1)
        y_pos += 15
        
        cv2.putText(frame_with_text, f"Smoothness: {metrics['action_smoothness']:.3f}", (right_x, y_pos), 
                   font, 0.4, (255, 150, 255), 1)
        
        return frame_with_text
    
    def _parse_sensor_noise_description(self, description: str) -> dict:
        """Parse sensor noise information from description string"""
        info = {
            'condition': description,
            'has_noise': 'noisy' in description.lower(),
            'pos_noise': 0.0,
            'vel_noise': 0.0, 
            'ori_noise': 0.0
        }
        
        if 'σ=0.05' in description:
            info['pos_noise'] = 0.05
            info['vel_noise'] = 0.10  # velocity_std = noise_std * 2
            info['ori_noise'] = 0.02  # orientation_std = noise_std * 0.4
        elif 'σ=0.1' in description:
            info['pos_noise'] = 0.10
            info['vel_noise'] = 0.20
            info['ori_noise'] = 0.04
        
        return info
    
    def _add_text_overlay(self, frame: np.ndarray, model_name: str, description: str, step: int) -> np.ndarray:
        """Add text overlay to video frame (fallback method)"""
        metrics = {'velocity': 0, 'avg_velocity': 0, 'total_reward': 0, 'action_smoothness': 0}
        return self._add_text_overlay_with_metrics(frame, model_name, description, step, metrics)
    
    def _save_video(self, frames: List[np.ndarray], video_path: str, fps: int):
        """Save frames as MP4 video"""
        if not frames:
            return
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
    
    def _create_comparison_grid(self, output_dir: str):
        """Create a grid comparison video showing all conditions"""
        print("  Creating comparison grid video...")
        
        video_files = [
            os.path.join(output_dir, "baseline_clean.mp4"),
            os.path.join(output_dir, "sr2l_clean.mp4"),
            os.path.join(output_dir, "baseline_noisy.mp4"),
            os.path.join(output_dir, "sr2l_noisy.mp4"),
        ]
        
        # Check if all videos exist
        if not all(os.path.exists(vf) for vf in video_files):
            print("    Skipping grid video - not all individual videos were created")
            return
        
        try:
            # This would require more complex video processing
            # For now, just create a simple instruction file
            instruction_path = os.path.join(output_dir, "create_comparison_grid.txt")
            with open(instruction_path, 'w') as f:
                f.write("To create a comparison grid video, use ffmpeg:\n\n")
                f.write("ffmpeg -i baseline_clean.mp4 -i sr2l_clean.mp4 ")
                f.write("-i baseline_noisy.mp4 -i sr2l_noisy.mp4 ")
                f.write("-filter_complex '[0:v][1:v]hstack=inputs=2[top]; ")
                f.write("[2:v][3:v]hstack=inputs=2[bottom]; ")
                f.write("[top][bottom]vstack=inputs=2[out]' ")
                f.write("-map '[out]' comparison_grid.mp4\n")
            
            print(f"    Grid instruction saved: {instruction_path}")
        except Exception as e:
            print(f"    Failed to create grid instruction: {e}")
    
    def run_comprehensive_evaluation(self, output_dir: str = "sr2l_evaluation_results"):
        """
        Run full SR2L evaluation and generate report
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("="*60)
        print("SR2L COMPREHENSIVE EVALUATION")
        print("="*60)
        
        # Create clean environment for testing
        env = create_correct_env(use_target_walking=True)
        
        results = {}
        
        # 1. Action Smoothness Analysis
        print("\n1. ACTION SMOOTHNESS ANALYSIS")
        print("-" * 40)
        
        baseline_smoothness = self.measure_action_smoothness(self.baseline_model, env)
        sr2l_smoothness = self.measure_action_smoothness(self.sr2l_model, env)
        
        results['smoothness'] = {
            'baseline': baseline_smoothness,
            'sr2l': sr2l_smoothness
        }
        
        print(f"Baseline PPO:")
        for metric, value in baseline_smoothness.items():
            print(f"  {metric}: {value:.4f}")
        
        print(f"\nPPO + SR2L:")
        for metric, value in sr2l_smoothness.items():
            print(f"  {metric}: {value:.4f}")
        
        print(f"\nImprovement:")
        for metric in baseline_smoothness.keys():
            if 'change' in metric or 'jerkiness' in metric:
                # Lower is better for these metrics
                improvement = (baseline_smoothness[metric] - sr2l_smoothness[metric]) / baseline_smoothness[metric] * 100
                print(f"  {metric}: {improvement:+.1f}% ({'better' if improvement > 0 else 'worse'})")
        
        # 2. Noise Robustness Testing
        print("\n2. NOISE ROBUSTNESS TESTING")
        print("-" * 40)
        
        baseline_noise_results = self.test_noise_robustness(self.baseline_model, env)
        sr2l_noise_results = self.test_noise_robustness(self.sr2l_model, env)
        
        results['noise_robustness'] = {
            'baseline': baseline_noise_results,
            'sr2l': sr2l_noise_results
        }
        
        # Print noise robustness table
        print(f"{'Noise Level':<12} {'Baseline Success':<16} {'SR2L Success':<16} {'Improvement':<12}")
        print("-" * 60)
        
        for noise_level in sorted(baseline_noise_results.keys()):
            baseline_success = baseline_noise_results[noise_level]['success_rate']
            sr2l_success = sr2l_noise_results[noise_level]['success_rate']
            improvement = (sr2l_success - baseline_success) * 100
            
            print(f"{noise_level:<12.3f} {baseline_success:<16.1%} {sr2l_success:<16.1%} {improvement:<12.1f}%")
        
        # 3. Record Comparison Videos
        print("\n3. RECORDING COMPARISON VIDEOS")
        print("-" * 40)
        
        # Create environment with render mode for video recording
        env_for_video = create_correct_env(render_mode='rgb_array', use_target_walking=True)
        self.record_comparison_videos(env_for_video, output_dir)
        env_for_video.close()
        
        # 4. Generate Plots
        print("\n4. GENERATING VISUALIZATIONS")
        print("-" * 40)
        
        self._create_visualizations(results, output_dir)
        
        # 4. Save Results
        results_file = os.path.join(output_dir, "sr2l_evaluation_results.npz")
        np.savez(results_file, **results)
        print(f"Results saved to: {results_file}")
        
        env.close()
        return results
    
    def _create_visualizations(self, results: Dict, output_dir: str):
        """Create plots comparing baseline vs SR2L"""
        
        # Plot 1: Noise Robustness Comparison
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        noise_levels = sorted(results['noise_robustness']['baseline'].keys())
        baseline_success = [results['noise_robustness']['baseline'][nl]['success_rate'] for nl in noise_levels]
        sr2l_success = [results['noise_robustness']['sr2l'][nl]['success_rate'] for nl in noise_levels]
        
        plt.plot(noise_levels, baseline_success, 'o-', label='PPO Baseline', linewidth=2)
        plt.plot(noise_levels, sr2l_success, 's-', label='PPO + SR2L', linewidth=2)
        plt.xlabel('Sensor Noise Level (σ)')
        plt.ylabel('Success Rate')
        plt.title('Robustness to Sensor Noise')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Plot 2: Smoothness Metrics Comparison
        plt.subplot(1, 2, 2)
        metrics = ['mean_action_change', 'std_action_change', 'action_jerkiness_score']
        baseline_vals = [results['smoothness']['baseline'][m] for m in metrics]
        sr2l_vals = [results['smoothness']['sr2l'][m] for m in metrics]
        
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x_pos - width/2, baseline_vals, width, label='PPO Baseline', alpha=0.7)
        plt.bar(x_pos + width/2, sr2l_vals, width, label='PPO + SR2L', alpha=0.7)
        
        plt.xlabel('Smoothness Metrics')
        plt.ylabel('Value (Lower = Smoother)')
        plt.title('Action Smoothness Comparison')
        plt.xticks(x_pos, ['Mean Change', 'Std Change', 'Jerkiness'])
        plt.legend()
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sr2l_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {output_dir}/sr2l_comparison.png")


def main():
    parser = argparse.ArgumentParser(description='Evaluate SR2L effectiveness')
    parser.add_argument('--baseline_model', type=str, 
                       help='Path to baseline PPO model (.zip). Use "auto" to find latest custom_reward model')
    parser.add_argument('--sr2l_model', type=str,
                       help='Path to PPO+SR2L model (.zip). Use "auto" to find latest SR2L model')
    parser.add_argument('--baseline_norm', type=str, default=None,
                       help='Path to baseline VecNormalize (.pkl)')
    parser.add_argument('--sr2l_norm', type=str, default=None,
                       help='Path to SR2L VecNormalize (.pkl)')
    parser.add_argument('--output_dir', type=str, default='sr2l_evaluation_results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Auto-detect models if requested
    if args.baseline_model == "auto" or args.baseline_model is None:
        print("Auto-detecting latest custom reward baseline model...")
        try:
            baseline_model, baseline_norm = find_latest_model("experiments/*custom_reward*")
            args.baseline_model = baseline_model
            if args.baseline_norm is None:
                args.baseline_norm = baseline_norm
        except FileNotFoundError:
            # Fallback to ppo_baseline if no custom_reward found
            print("No custom_reward model found, looking for ppo_baseline...")
            baseline_model, baseline_norm = find_latest_model("experiments/ppo_baseline*")
            args.baseline_model = baseline_model
            if args.baseline_norm is None:
                args.baseline_norm = baseline_norm
    
    if args.sr2l_model == "auto" or args.sr2l_model is None:
        print("Auto-detecting latest SR2L model...")
        # Look for both ppo_sr2l and ppo_sr2l_phased models
        sr2l_model, sr2l_norm = find_latest_model("experiments/ppo_sr2l*")
        if sr2l_model is None:
            sr2l_model, sr2l_norm = find_latest_model("experiments/*sr2l*")
        args.sr2l_model = sr2l_model
        if args.sr2l_norm is None:
            args.sr2l_norm = sr2l_norm
    
    # Validate model files exist
    if not os.path.exists(args.baseline_model):
        raise FileNotFoundError(f"Baseline model not found: {args.baseline_model}")
    if not os.path.exists(args.sr2l_model):
        raise FileNotFoundError(f"SR2L model not found: {args.sr2l_model}")
    
    print(f"Using baseline model: {args.baseline_model}")
    print(f"Using SR2L model: {args.sr2l_model}")
    
    # Run evaluation
    evaluator = SR2LEvaluator(
        baseline_model_path=args.baseline_model,
        sr2l_model_path=args.sr2l_model,
        vec_normalize_baseline=args.baseline_norm,
        vec_normalize_sr2l=args.sr2l_norm
    )
    
    results = evaluator.run_comprehensive_evaluation(args.output_dir)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print(f"Results saved to: {args.output_dir}/")
    print("Key files:")
    print(f"  - sr2l_evaluation_results.npz (raw data)")
    print(f"  - sr2l_comparison.png (plots)")


if __name__ == "__main__":
    main()