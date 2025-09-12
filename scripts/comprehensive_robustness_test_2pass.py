#!/usr/bin/env python3
"""
Comprehensive Robustness Test Suite - 2-PASS with Video Recording
Pass 1: Collect accurate performance data without rendering
Pass 2: Replay with rendering for video visualization
"""
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
from envs.domain_randomization_wrapper import DomainRandomizationWrapper
import realant_sim
import argparse
from datetime import datetime
import json
import cv2
import os

def collect_performance_data(model_path, vec_normalize_path, model_name, 
                           test_name, dr_config=None, episodes=5):
    """PASS 1: Collect accurate performance data without rendering"""
    print(f"\n🔬 PASS 1: Collecting performance data for {test_name}")
    print("=" * 80)
    
    def make_env():
        env = gym.make('RealAntMujoco-v0')  # NO RENDERING
        env = SuccessRewardWrapper(env)
        
        if dr_config:
            env = DomainRandomizationWrapper(env, dr_config)
            print(f"  🎲 DR Config: {dr_config}")
        
        return env
    
    env = DummyVecEnv([make_env])
    
    # Load VecNormalize
    try:
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
        print("  ✅ VecNormalize loaded")
    except:
        print("  ⚠️  No VecNormalize")
    
    # Load model
    model = PPO.load(model_path)
    print(f"  ✅ Model loaded")
    
    # Collect data
    all_episode_data = []
    
    print(f"\n  Collecting data from {episodes} episodes...")
    print("  " + "-" * 70)
    
    for episode in range(episodes):
        obs = env.reset()
        
        episode_data = {
            'actions': [],
            'rewards': [],
            'positions': [],
            'failed_joints_log': [],
            'episode_length': 0
        }
        
        for step in range(1000):
            # Get action (deterministic for reproducibility)
            action, _ = model.predict(obs, deterministic=True)
            episode_data['actions'].append(action.copy())
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            # Track position
            x_pos = env.envs[0].unwrapped.data.qpos[0]
            episode_data['positions'].append(x_pos)
            episode_data['rewards'].append(reward[0])
            
            # Track joint failures - check for dropped_joints attribute
            current_failed_joints = []
            try:
                # Check DomainRandomizationWrapper for dropped_joints
                wrapper = env.envs[0]
                while wrapper:
                    if hasattr(wrapper, 'dropped_joints'):
                        current_failed_joints = getattr(wrapper, 'dropped_joints', [])
                        break
                    elif hasattr(wrapper, 'failed_joints'):
                        current_failed_joints = getattr(wrapper, 'failed_joints', [])
                        break
                    wrapper = getattr(wrapper, 'env', None)
            except:
                pass
            
            episode_data['failed_joints_log'].append(current_failed_joints.copy() if current_failed_joints else [])
            episode_data['episode_length'] = step + 1
            
            if done[0]:
                print(f"    Episode {episode+1} ended at step {step}")
                break
        
        # Calculate metrics for this episode
        positions = episode_data['positions']
        if len(positions) >= 2:
            total_distance = sum(abs(positions[i] - positions[i-1]) for i in range(1, len(positions)))
            displacement = positions[-1] - positions[0]
            max_distance = max(abs(p) for p in positions)
            total_reward = sum(episode_data['rewards'])
            
            # Analyze joint failures
            all_failed_joints = set()
            failure_steps = 0
            for step_failures in episode_data['failed_joints_log']:
                if step_failures:
                    all_failed_joints.update(step_failures)
                    failure_steps += 1
            
            failure_rate = failure_steps / len(episode_data['failed_joints_log']) if episode_data['failed_joints_log'] else 0
            
            episode_data['metrics'] = {
                'total_distance': float(total_distance),
                'displacement': float(displacement),
                'max_distance': float(max_distance),
                'total_reward': float(total_reward),
                'unique_failed_joints': list(all_failed_joints),
                'failure_rate': float(failure_rate)
            }
            
            print(f"  Episode {episode+1:2d}: TotalDist={total_distance:6.1f}m, "
                  f"Disp={displacement:5.1f}m, Reward={total_reward:6.0f}, "
                  f"FailedJoints={list(all_failed_joints) if all_failed_joints else 'None'}, "
                  f"FailRate={failure_rate*100:.1f}%")
        else:
            episode_data['metrics'] = {
                'total_distance': 0.0,
                'displacement': 0.0,
                'max_distance': 0.0,
                'total_reward': 0.0,
                'unique_failed_joints': [],
                'failure_rate': 0.0
            }
            print(f"  Episode {episode+1:2d}: FAILED (too short)")
        
        all_episode_data.append(episode_data)
    
    env.close()
    
    # Calculate overall statistics
    metrics_list = [ep['metrics'] for ep in all_episode_data]
    distances = [m['total_distance'] for m in metrics_list]
    rewards = [m['total_reward'] for m in metrics_list]
    
    stats = {
        'test_name': test_name,
        'episodes': len(all_episode_data),
        'avg_distance': float(np.mean(distances)),
        'std_distance': float(np.std(distances)),
        'avg_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'success_rate': float(len([d for d in distances if d > 2]) / len(distances)),
        'all_failed_joints': list(set().union(*[m['unique_failed_joints'] for m in metrics_list])),
        'avg_failure_rate': float(np.mean([m['failure_rate'] for m in metrics_list]))
    }
    
    print(f"\n  📊 PASS 1 SUMMARY:")
    print(f"    Average Distance:   {stats['avg_distance']:6.1f} ± {stats['std_distance']:.1f} m")
    print(f"    Average Reward:     {stats['avg_reward']:6.0f} ± {stats['std_reward']:.0f}")
    print(f"    Success Rate:       {stats['success_rate']*100:5.1f}% (episodes >2m)")
    
    if dr_config:
        print(f"    Joint Failure Rate: {stats['avg_failure_rate']*100:5.1f}%")
        print(f"    Failed Joints:      {stats['all_failed_joints']}")
    
    return all_episode_data, stats

def create_video_replay(model_path, vec_normalize_path, model_name, test_name, 
                       episode_data_list, dr_config=None):
    """PASS 2: Replay collected data with rendering for video"""
    print(f"\n🎬 PASS 2: Creating video replay for {test_name}")
    print("=" * 80)
    
    def make_env():
        env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')  # WITH RENDERING
        env = SuccessRewardWrapper(env)
        
        if dr_config:
            env = DomainRandomizationWrapper(env, dr_config)
        
        return env
    
    env = DummyVecEnv([make_env])
    
    # Load VecNormalize
    try:
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
    except:
        pass
    
    # Video setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20
    frame_size = (1280, 720)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_test_name = test_name.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
    video_path = f"videos/{model_name}_{safe_test_name}_{timestamp}.mp4"
    os.makedirs("videos", exist_ok=True)
    
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
    print(f"  📹 Recording to: {video_path}")
    
    frame_count = 0
    
    for episode_idx, episode_data in enumerate(episode_data_list):
        print(f"\n  Replaying Episode {episode_idx + 1}/{len(episode_data_list)}")
        
        # Reset environment for this episode
        obs = env.reset()
        
        # Replay the collected actions
        for step, action in enumerate(episode_data['actions']):
            # Use the recorded action
            obs, reward, done, info = env.step(action)
            
            # Get current state for overlay
            x_pos = env.envs[0].unwrapped.data.qpos[0]
            
            # Get recorded data for accurate overlay
            recorded_pos = episode_data['positions'][step] if step < len(episode_data['positions']) else x_pos
            recorded_reward = episode_data['rewards'][step] if step < len(episode_data['rewards']) else 0
            recorded_failed_joints = episode_data['failed_joints_log'][step] if step < len(episode_data['failed_joints_log']) else []
            total_reward_so_far = sum(episode_data['rewards'][:step+1])
            
            # Render frame
            frame = env.envs[0].render()
            frame = cv2.resize(frame, frame_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Create detailed overlay with Pass 1 data
            overlay_texts = [
                f"Model: {model_name}",
                f"Test: {test_name}",
                f"Episode: {episode_idx+1}/{len(episode_data_list)} | Step: {step}",
                f"",
                f"=== POSITION (Pass 1 Data) ===",
                f"Recorded Position: {recorded_pos:.2f}m",
                f"Current Position: {x_pos:.2f}m",
                f"",
                f"=== PERFORMANCE ===",
                f"Instant Reward: {recorded_reward:.1f}",
                f"Total Reward: {total_reward_so_far:.0f}",
                f"",
                f"=== JOINT FAILURES ===",
                f"Failed Joints: {recorded_failed_joints if recorded_failed_joints else 'None'}",
            ]
            
            # Add status indicators
            if dr_config and dr_config.get('joint_dropout_prob', 0) > 0:
                if recorded_failed_joints:
                    overlay_texts.append("🔴 JOINT FAILURE ACTIVE!")
                    # Add specific joint info
                    joint_names = ['Hip1', 'Ankle1', 'Hip2', 'Ankle2', 'Hip3', 'Ankle3', 'Hip4', 'Ankle4']
                    failed_names = [joint_names[j] if j < len(joint_names) else f'Joint{j}' for j in recorded_failed_joints]
                    overlay_texts.append(f"Failed: {', '.join(failed_names)}")
                else:
                    overlay_texts.append("✅ All joints working")
            
            # Add distance tracking
            if step > 0 and episode_data['positions']:
                distance_so_far = sum(abs(episode_data['positions'][i] - episode_data['positions'][i-1]) 
                                    for i in range(1, min(step+1, len(episode_data['positions']))))
                overlay_texts.append(f"Distance traveled: {distance_so_far:.1f}m")
            
            # Draw text overlay with black backgrounds
            for i, text in enumerate(overlay_texts):
                y_pos = 30 + i * 25
                text_width = len(text) * 10
                cv2.rectangle(frame, (10, y_pos - 20), (text_width + 20, y_pos + 5), (0, 0, 0), -1)
                cv2.putText(frame, text, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Color-coded performance border
            final_distance = episode_data['metrics']['total_distance']
            if final_distance > 8:
                border_color = (0, 255, 0)  # Green - excellent
            elif final_distance > 5:
                border_color = (0, 255, 255)  # Yellow - good
            elif final_distance > 2:
                border_color = (255, 255, 0)  # Cyan - moderate
            else:
                border_color = (0, 0, 255)  # Red - poor
            
            cv2.rectangle(frame, (0, 0), (frame_size[0]-1, frame_size[1]-1), border_color, 5)
            
            # Draw trajectory path from Pass 1 data
            if step > 10:
                for i in range(max(0, step - 50), step):
                    if i < len(episode_data['positions']) - 1:
                        # Convert to pixel coords
                        x1 = int(640 + episode_data['positions'][i] * 100)
                        y1 = int(360 - 100)  # Fixed y for ground plane
                        x2 = int(640 + episode_data['positions'][i+1] * 100)
                        y2 = int(360 - 100)
                        
                        # Clamp to frame
                        x1 = max(0, min(x1, frame_size[0]-1))
                        y1 = max(0, min(y1, frame_size[1]-1))
                        x2 = max(0, min(x2, frame_size[0]-1))
                        y2 = max(0, min(y2, frame_size[1]-1))
                        
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            video_writer.write(frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"    Rendered {frame_count} frames", end='\r')
            
            if done[0]:
                break
        
        # Add episode summary frames
        for _ in range(60):  # 3 seconds at 20fps
            summary_frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
            
            # Episode summary
            metrics = episode_data['metrics']
            summary_texts = [
                f"Episode {episode_idx + 1} Complete",
                "",
                f"Total Distance: {metrics['total_distance']:.1f}m",
                f"Total Reward: {metrics['total_reward']:.0f}",
                f"Failed Joints: {metrics['unique_failed_joints'] if metrics['unique_failed_joints'] else 'None'}",
                f"Failure Rate: {metrics['failure_rate']*100:.1f}%",
            ]
            
            for i, text in enumerate(summary_texts):
                y_pos = 250 + i * 40
                cv2.putText(summary_frame, text, (400, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            video_writer.write(summary_frame)
    
    video_writer.release()
    env.close()
    
    print(f"\n  ✅ Video saved: {video_path}")
    print(f"  Total frames: {frame_count}")
    
    return video_path

def test_condition_2pass(model_path, vec_normalize_path, model_name, 
                        test_name, dr_config=None, episodes=5, record_video=False):
    """Complete 2-pass test: data collection + video creation"""
    
    # Pass 1: Collect accurate performance data
    episode_data_list, stats = collect_performance_data(
        model_path, vec_normalize_path, model_name, test_name, dr_config, episodes
    )
    
    # Pass 2: Create video if requested
    if record_video:
        video_path = create_video_replay(
            model_path, vec_normalize_path, model_name, test_name, 
            episode_data_list, dr_config
        )
        stats['video_path'] = video_path
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='2-Pass comprehensive robustness test')
    parser.add_argument('--model', type=str, required=True, help='Path to model.zip')
    parser.add_argument('--vec', type=str, required=True, help='Path to vec_normalize.pkl')
    parser.add_argument('--name', type=str, required=True, help='Model name')
    parser.add_argument('--episodes', type=int, default=5, help='Episodes per test')
    parser.add_argument('--video', action='store_true', help='Record video (2-pass)')
    parser.add_argument('--test', type=str, help='Run specific test (baseline, low, moderate, high, noise, combined)')
    args = parser.parse_args()
    
    print("🚀 2-PASS COMPREHENSIVE ROBUSTNESS TEST SUITE")
    print(f"Testing: {args.name}")
    print("Pass 1: Accurate data collection (no rendering)")
    print("Pass 2: Video creation (with rendering)")
    print("=" * 80)
    
    # Define test configurations
    test_configs = {
        'baseline': {
            'name': 'BASELINE (No Failures)',
            'dr_config': None
        },
        'low': {
            'name': 'LOW FAILURES (2% joint dropout)',
            'dr_config': {
                'joint_dropout_prob': 0.02,
                'max_dropped_joints': 1,
                'min_dropped_joints': 0,
                'sensor_noise_std': 0.0,
                'noise_joints_only': True
            }
        },
        'moderate': {
            'name': 'MODERATE FAILURES (5% joint dropout)',
            'dr_config': {
                'joint_dropout_prob': 0.05,
                'max_dropped_joints': 1,
                'min_dropped_joints': 0,
                'sensor_noise_std': 0.0,
                'noise_joints_only': True
            }
        },
        'high': {
            'name': 'HIGH FAILURES (10% joint dropout)',
            'dr_config': {
                'joint_dropout_prob': 0.10,
                'max_dropped_joints': 2,
                'min_dropped_joints': 0,
                'sensor_noise_std': 0.0,
                'noise_joints_only': True
            }
        },
        'noise': {
            'name': 'SENSOR NOISE (1% noise, no failures)',
            'dr_config': {
                'joint_dropout_prob': 0.0,
                'max_dropped_joints': 0,
                'min_dropped_joints': 0,
                'sensor_noise_std': 0.01,
                'noise_joints_only': True
            }
        },
        'combined': {
            'name': 'COMBINED (5% failures + 0.5% noise)',
            'dr_config': {
                'joint_dropout_prob': 0.05,
                'max_dropped_joints': 1,
                'min_dropped_joints': 0,
                'sensor_noise_std': 0.005,
                'noise_joints_only': True
            }
        }
    }
    
    # Run specific test or default to moderate
    test_key = args.test or 'moderate'
    if test_key not in test_configs:
        print(f"❌ Unknown test: {test_key}")
        return
    
    config = test_configs[test_key]
    result = test_condition_2pass(
        args.model, args.vec, args.name,
        config['name'],
        dr_config=config['dr_config'],
        episodes=args.episodes,
        record_video=args.video
    )
    
    if result:
        print(f"\n🎯 FINAL RESULT:")
        print(f"  Performance: {result['avg_distance']:.1f}m average distance")
        print(f"  Success Rate: {result['success_rate']*100:.1f}%")
        if result.get('avg_failure_rate', 0) > 0:
            print(f"  Robustness: {result['avg_failure_rate']*100:.1f}% joint failure rate handled")
        
        if args.video:
            print(f"  📹 Video: {result.get('video_path', 'Not recorded')}")

if __name__ == "__main__":
    main()