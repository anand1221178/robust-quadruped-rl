"""
Two-Pass Video Recording Utilities
Separates trajectory collection from rendering to get accurate performance metrics
"""

import numpy as np
import gymnasium as gym
import imageio
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from typing import Dict, List, Optional, Tuple
import os

class TwoPassVideoRecorder:
    """
    Records robot performance in two passes:
    1. Collect trajectory without rendering (true performance)
    2. Replay trajectory with rendering (accurate video)
    """
    
    def __init__(self, env_name: str = 'RealAntMujoco-v0'):
        self.env_name = env_name
        self.trajectory = None
        self.performance_metrics = {}
        
    def collect_trajectory(self, model, env, num_steps: int = 500, 
                          show_progress: bool = True) -> Dict:
        """
        Pass 1: Collect trajectory without rendering overhead
        
        Args:
            model: Trained model
            env: Environment (already wrapped and normalized)
            num_steps: Number of steps to collect
            show_progress: Print progress updates
            
        Returns:
            Dict with trajectory data and performance metrics
        """
        if show_progress:
            print("=" * 50)
            print("PASS 1: Collecting trajectory (no rendering)...")
            print("=" * 50)
        
        # Initialize trajectory storage
        trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'infos': [],
            'dones': [],
            'velocities': [],
            'distances': []
        }
        
        # Reset and collect
        obs = env.reset()
        episode_reward = 0
        velocities = []
        
        for step in range(num_steps):
            # Store observation
            trajectory['observations'].append(obs.copy())
            
            # Get action
            action, _ = model.predict(obs, deterministic=True)
            trajectory['actions'].append(action.copy())
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            # Store results
            trajectory['rewards'].append(reward[0] if hasattr(reward, '__len__') else reward)
            trajectory['infos'].append(info[0] if hasattr(info, '__len__') else info)
            trajectory['dones'].append(done[0] if hasattr(done, '__len__') else done)
            
            episode_reward += reward[0] if hasattr(reward, '__len__') else reward
            
            # Extract velocity
            if info and len(info) > 0:
                info_dict = info[0] if hasattr(info, '__len__') else info
                if info_dict and isinstance(info_dict, dict):
                    if 'current_velocity' in info_dict:
                        vel = info_dict['current_velocity']
                    elif 'speed' in info_dict:
                        vel = info_dict['speed']
                    else:
                        vel = 0.0
                    velocities.append(vel)
                    trajectory['velocities'].append(vel)
            
            # Check for early termination
            if done[0] if hasattr(done, '__len__') else done:
                if show_progress:
                    print(f"Episode ended early at step {step}")
                break
        
        # Calculate metrics
        self.performance_metrics = {
            'episode_reward': episode_reward,
            'episode_length': len(trajectory['actions']),
            'avg_velocity': np.mean(velocities) if velocities else 0.0,
            'max_velocity': np.max(velocities) if velocities else 0.0,
            'min_velocity': np.min(velocities) if velocities else 0.0,
            'velocity_std': np.std(velocities) if velocities else 0.0
        }
        
        if show_progress:
            print("\n" + "=" * 50)
            print("TRUE PERFORMANCE (No Rendering Overhead):")
            print("=" * 50)
            print(f"Total Reward: {episode_reward:.2f}")
            print(f"Episode Length: {len(trajectory['actions'])} steps")
            print(f"Average Velocity: {self.performance_metrics['avg_velocity']:.3f} m/s")
            print(f"Max Velocity: {self.performance_metrics['max_velocity']:.3f} m/s")
            print("=" * 50)
        
        self.trajectory = trajectory
        return trajectory
    
    def replay_with_rendering(self, trajectory: Optional[Dict] = None,
                            output_path: str = "robot_video_twopass.mp4",
                            wrapper_class = None,
                            show_progress: bool = True) -> bool:
        """
        Pass 2: Replay trajectory with rendering to create video
        
        Args:
            trajectory: Trajectory to replay (uses self.trajectory if None)
            output_path: Video output path
            wrapper_class: Environment wrapper class to apply
            show_progress: Print progress updates
            
        Returns:
            Success flag
        """
        if trajectory is None:
            trajectory = self.trajectory
            
        if trajectory is None or len(trajectory['actions']) == 0:
            print("No trajectory to replay!")
            return False
        
        if show_progress:
            print("\n" + "=" * 50)
            print("PASS 2: Creating video from trajectory...")
            print("=" * 50)
        
        try:
            # Create environment with rendering
            def make_env():
                env = gym.make(self.env_name, render_mode='rgb_array')
                if wrapper_class:
                    env = wrapper_class(env)
                return env
            
            env = DummyVecEnv([make_env])
            
            # Reset environment
            obs = env.reset()
            frames = []
            
            # Replay trajectory
            for i, action in enumerate(trajectory['actions']):
                # Step with recorded action
                obs, _, done, _ = env.step(action)
                
                # Capture frame
                frame = env.render(mode='rgb_array')
                if frame is not None:
                    frames.append(frame)
                
                if done[0] if hasattr(done, '__len__') else done:
                    break
            
            env.close()
            
            # Save video
            if frames:
                if show_progress:
                    print(f"Saving video to {output_path}...")
                imageio.mimsave(output_path, frames, fps=50)
                if show_progress:
                    print(f"Video saved! ({len(frames)} frames)")
                return True
            else:
                print("No frames captured!")
                return False
                
        except Exception as e:
            print(f"Error creating video: {str(e)}")
            return False
    
    def get_metrics(self) -> Dict:
        """Get performance metrics from last trajectory collection"""
        return self.performance_metrics.copy()
    
    def record_episode(self, model, env, output_path: str = "robot_video.mp4",
                      num_steps: int = 500, wrapper_class = None) -> Dict:
        """
        Convenience method to do both passes automatically
        
        Returns:
            Performance metrics dict
        """
        # Pass 1: Collect trajectory
        self.collect_trajectory(model, env, num_steps)
        
        # Pass 2: Create video
        self.replay_with_rendering(output_path=output_path, wrapper_class=wrapper_class)
        
        print("\n" + "=" * 50)
        print("TWO-PASS RECORDING COMPLETE!")
        print("=" * 50)
        print(f"✅ True Performance: {self.performance_metrics['avg_velocity']:.3f} m/s")
        print(f"✅ Video saved to: {output_path}")
        print("=" * 50)
        
        return self.performance_metrics