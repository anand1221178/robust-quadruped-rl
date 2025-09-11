#!/usr/bin/env python3
"""
Comprehensive Robustness Test Suite with Video Recording
Tests models under various failure conditions with proper metrics + video proof
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

def test_robustness_with_video(model_path, vec_normalize_path, model_name, 
                              test_name, dr_config=None, episodes=5, record_video=False):
    """Test model under specific robustness condition with optional video"""
    print(f"\n🔬 TESTING: {model_name} - {test_name}")
    print("=" * 80)
    
    def make_env(render_mode=None):
        env = gym.make('RealAntMujoco-v0', render_mode=render_mode)
        env = SuccessRewardWrapper(env)
        
        if dr_config:
            env = DomainRandomizationWrapper(env, dr_config)
            print(f"  🎲 DR Config: {dr_config}")
        
        return env
    
    # Setup video recording if requested
    video_writer = None
    if record_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 20
        frame_size = (1280, 720)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = f"videos/{model_name}_{test_name.replace(' ', '_')}_{timestamp}.mp4"
        os.makedirs("videos", exist_ok=True)
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
        print(f"  📹 Recording video to: {video_path}")
    
    env = DummyVecEnv([lambda: make_env('rgb_array' if record_video else None)])
    
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
    
    # Run episodes
    results = []
    
    print(f"\n  Running {episodes} episodes...")
    print("  " + "-" * 70)
    
    for episode in range(episodes):
        obs = env.reset()
        
        positions = []
        rewards = []
        failed_joints_log = []
        frame_count = 0
        
        for step in range(1000):
            # Get action
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Track position
            x_pos = env.envs[0].unwrapped.data.qpos[0]
            positions.append(x_pos)
            rewards.append(reward[0])
            
            # Track joint failures - check for dropped_joints attribute
            current_failed_joints = []
            try:
                # Check DomainRandomizationWrapper for dropped_joints
                wrapper = env.envs[0]
                while wrapper:
                    # Check for dropped_joints (correct attribute name)
                    if hasattr(wrapper, 'dropped_joints'):
                        current_failed_joints = getattr(wrapper, 'dropped_joints', [])
                        break
                    # Also try failed_joints (legacy)
                    elif hasattr(wrapper, 'failed_joints'):
                        current_failed_joints = getattr(wrapper, 'failed_joints', [])
                        break
                    # Move to next wrapper level
                    wrapper = getattr(wrapper, 'env', None)
            except:
                pass
            
            failed_joints_log.append(current_failed_joints.copy() if current_failed_joints else [])
            
            # Record video frame
            if record_video:
                frame = env.envs[0].render()
                frame = cv2.resize(frame, frame_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Add overlay
                distance = abs(x_pos) if len(positions) > 0 else 0
                total_reward = sum(rewards)
                
                overlay_texts = [
                    f"Model: {model_name}",
                    f"Test: {test_name}",
                    f"Episode: {episode+1}/{episodes} | Step: {step}",
                    f"",
                    f"Position: {x_pos:.2f}m",
                    f"Distance: {distance:.2f}m", 
                    f"Total Reward: {total_reward:.0f}",
                    f"",
                    f"Failed Joints: {current_failed_joints if current_failed_joints else 'None'}",
                ]
                
                if dr_config and dr_config.get('joint_dropout_prob', 0) > 0:
                    if current_failed_joints:
                        overlay_texts.append("🔴 JOINT FAILURE ACTIVE!")
                    else:
                        overlay_texts.append("✅ All joints working")
                
                # Draw text overlay
                for i, text in enumerate(overlay_texts):
                    y_pos = 30 + i * 25
                    text_width = len(text) * 10
                    cv2.rectangle(frame, (10, y_pos - 20), (text_width + 20, y_pos + 5), (0, 0, 0), -1)
                    cv2.putText(frame, text, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Color-coded performance indicator
                if distance > 3:
                    color = (0, 255, 0)  # Green - good
                elif distance > 1:
                    color = (0, 255, 255)  # Yellow - moderate
                else:
                    color = (0, 0, 255)  # Red - poor
                
                cv2.rectangle(frame, (0, 0), (frame_size[0]-1, frame_size[1]-1), color, 5)
                
                video_writer.write(frame)
                frame_count += 1
            
            if done[0]:
                if record_video:
                    # Add end-of-episode frames
                    for _ in range(40):  # 2 seconds at 20fps
                        end_frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
                        final_distance = sum(abs(positions[i] - positions[i-1]) for i in range(1, len(positions)))
                        cv2.putText(end_frame, f"Episode {episode+1} Complete", 
                                   (480, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(end_frame, f"Total Distance: {final_distance:.1f}m", 
                                   (480, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.putText(end_frame, f"Total Reward: {sum(rewards):.0f}", 
                                   (480, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        video_writer.write(end_frame)
                break
        
        # Calculate metrics
        if len(positions) >= 2:
            total_distance = sum(abs(positions[i] - positions[i-1]) for i in range(1, len(positions)))
            displacement = positions[-1] - positions[0]
            max_distance = max(abs(p) for p in positions)
            total_reward = sum(rewards)
            episode_length = len(positions)
            
            # Analyze joint failures
            all_failed_joints = set()
            failure_steps = 0
            for step_failures in failed_joints_log:
                if step_failures:
                    all_failed_joints.update(step_failures)
                    failure_steps += 1
            
            failure_rate = failure_steps / len(failed_joints_log) if failed_joints_log else 0
            
            result = {
                'episode': episode + 1,
                'total_distance': float(total_distance),
                'displacement': float(displacement),
                'max_distance': float(max_distance),
                'total_reward': float(total_reward),
                'episode_length': episode_length,
                'unique_failed_joints': list(all_failed_joints),
                'failure_rate': float(failure_rate),
                'fell': bool(done[0])
            }
            
            results.append(result)
            
            print(f"  Episode {episode+1:2d}: TotalDist={total_distance:6.1f}m, "
                  f"Disp={displacement:5.1f}m, Reward={total_reward:6.0f}, "
                  f"FailedJoints={list(all_failed_joints) if all_failed_joints else 'None'}, "
                  f"FailRate={failure_rate*100:.1f}%")
        else:
            print(f"  Episode {episode+1:2d}: FAILED (too short)")
    
    if video_writer:
        video_writer.release()
        print(f"  ✅ Video saved with {frame_count} frames")
    
    env.close()
    
    # Calculate statistics
    if results:
        distances = [r['total_distance'] for r in results]
        displacements = [r['displacement'] for r in results]
        rewards = [r['total_reward'] for r in results]
        
        stats = {
            'test_name': test_name,
            'episodes': len(results),
            'avg_distance': float(np.mean(distances)),
            'std_distance': float(np.std(distances)),
            'avg_displacement': float(np.mean(displacements)),
            'avg_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'success_rate': float(len([r for r in results if r['total_distance'] > 2]) / len(results)),
            'all_failed_joints': list(set().union(*[r['unique_failed_joints'] for r in results])),
            'avg_failure_rate': float(np.mean([r['failure_rate'] for r in results]))
        }
        
        print(f"\n  📊 SUMMARY:")
        print(f"    Average Distance:   {stats['avg_distance']:6.1f} ± {stats['std_distance']:.1f} m")
        print(f"    Average Displacement: {stats['avg_displacement']:6.1f} m")
        print(f"    Average Reward:     {stats['avg_reward']:6.0f} ± {stats['std_reward']:.0f}")
        print(f"    Success Rate:       {stats['success_rate']*100:5.1f}% (episodes >2m)")
        
        if dr_config:
            print(f"    Joint Failure Rate: {stats['avg_failure_rate']*100:5.1f}%")
            print(f"    Failed Joints:      {stats['all_failed_joints']}")
        
        # Performance rating
        if stats['avg_distance'] > 15:
            rating = "🏆 EXCELLENT"
        elif stats['avg_distance'] > 8:
            rating = "✅ GOOD"
        elif stats['avg_distance'] > 3:
            rating = "⚠️  MODERATE"  
        else:
            rating = "❌ POOR"
        
        print(f"    Performance:        {rating}")
        
        if record_video:
            stats['video_path'] = video_path
        
        return stats
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Comprehensive robustness test with video')
    parser.add_argument('--model', type=str, required=True, help='Path to model.zip')
    parser.add_argument('--vec', type=str, required=True, help='Path to vec_normalize.pkl')
    parser.add_argument('--name', type=str, required=True, help='Model name')
    parser.add_argument('--episodes', type=int, default=5, help='Episodes per test')
    parser.add_argument('--video', action='store_true', help='Record video of tests')
    parser.add_argument('--test', type=str, help='Run only specific test (baseline, low, moderate, high, noise, combined)')
    args = parser.parse_args()
    
    print("🚀 COMPREHENSIVE ROBUSTNESS TEST SUITE WITH VIDEO")
    print(f"Testing: {args.name}")
    print("=" * 80)
    
    all_results = []
    
    # Define all tests
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
    
    # Run specific test or all tests
    tests_to_run = [args.test] if args.test else test_configs.keys()
    
    for test_key in tests_to_run:
        if test_key not in test_configs:
            print(f"❌ Unknown test: {test_key}")
            continue
            
        config = test_configs[test_key]
        result = test_robustness_with_video(
            args.model, args.vec, args.name,
            config['name'],
            dr_config=config['dr_config'],
            episodes=args.episodes,
            record_video=args.video
        )
        if result:
            all_results.append(result)
    
    # Final comparison
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("🏆 FINAL ROBUSTNESS REPORT")
        print("=" * 80)
        
        baseline_distance = all_results[0]['avg_distance'] if all_results else 10
        
        for result in all_results:
            retention = (result['avg_distance'] / baseline_distance) * 100 if baseline_distance > 0 else 0
            success_pct = result['success_rate'] * 100
            
            print(f"{result['test_name']:35} | "
                  f"Dist: {result['avg_distance']:5.1f}m | "
                  f"Retention: {retention:5.1f}% | "
                  f"Success: {success_pct:5.1f}%")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"robustness_results_{args.name.replace('-', '_')}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'model_name': args.name,
                'test_date': timestamp,
                'results': all_results
            }, f, indent=2)
        
        print(f"\n📁 Detailed results saved to: {filename}")
        
        # Overall assessment
        if len(all_results) > 1:
            avg_retention = np.mean([
                (r['avg_distance'] / baseline_distance) * 100 
                for r in all_results[1:] if baseline_distance > 0
            ])
            
            print(f"\n🎯 OVERALL ROBUSTNESS SCORE: {avg_retention:.1f}%")
            
            if avg_retention > 80:
                print("🏆 VERDICT: EXCELLENT robustness!")
            elif avg_retention > 60:
                print("✅ VERDICT: GOOD robustness")
            elif avg_retention > 40:
                print("⚠️  VERDICT: MODERATE robustness")
            else:
                print("❌ VERDICT: POOR robustness")

if __name__ == "__main__":
    main()