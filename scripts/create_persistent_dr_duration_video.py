#!/usr/bin/env python3
"""
Create video of Persistent DR model showing progressively longer failure durations
Shows what Persistent DR was actually trained for - realistic hardware failure recovery
"""

import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
from envs.persistent_dr_wrapper import PersistentDRWrapper
import realant_sim
import imageio
import os
from PIL import Image, ImageDraw, ImageFont

def collect_complete_episodes_with_failures(model_path, failure_scenarios, episode_length=400):
    """Collect complete episodes showing adaptation to different failure scenarios"""
    
    print("=" * 70)
    print("COLLECTING PERSISTENT DR COMPLETE EPISODES WITH FAILURE SCENARIOS")
    print("=" * 70)
    
    # Load Persistent DR model
    model = PPO.load(model_path)
    print(f"Loaded Persistent DR model from {model_path}")
    
    # Load VecNormalize once
    vec_path = model_path.replace('final_model.zip', 'vec_normalize.pkl')
    
    # Collect complete episodes for each scenario
    all_trajectory = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'scenarios': [],
        'velocities': [],
        'joint_health': [],
        'active_failures': [],
        'episode_markers': [],  # Mark where episodes start/end
        'distances': [],
        'positions': []
    }
    
    print(f"Testing {len(failure_scenarios)} complete episodes with different failure scenarios")
    print(f"Episode length: {episode_length} steps each")
    
    episode_start_idx = 0
    
    for scenario_idx, scenario in enumerate(failure_scenarios):
        print(f"\n--- Episode {scenario_idx+1}/{len(failure_scenarios)}: {scenario['description']} ---")
        
        # Create environment WITHOUT rendering for this scenario
        def make_env():
            env = gym.make('RealAntMujoco-v0')  # No render_mode!
            env = SuccessRewardWrapper(env)
            
            # Apply Persistent DR wrapper with current scenario
            if scenario['config']:
                env = PersistentDRWrapper(env, scenario['config'])
            return env
        
        env = DummyVecEnv([make_env])
        
        # Load VecNormalize
        if os.path.exists(vec_path):
            env = VecNormalize.load(vec_path, env)
            env.training = False
            env.norm_reward = False
        
        # Reset for this episode - fresh start
        obs = env.reset()
        episode_reward = 0
        episode_distance = 0
        initial_pos = obs[0][:3].copy() if len(obs[0]) > 2 else np.zeros(3)
        
        print(f"    Starting position: [{initial_pos[0]:.3f}, {initial_pos[1]:.3f}, {initial_pos[2]:.3f}]")
        
        # Run complete episode
        for step in range(episode_length):
            # Store data
            all_trajectory['observations'].append(obs.copy())
            all_trajectory['scenarios'].append(scenario)
            all_trajectory['episode_markers'].append(scenario_idx)
            
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            all_trajectory['actions'].append(action.copy())
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            all_trajectory['rewards'].append(reward[0])
            episode_reward += reward[0]
            
            # Track position and distance
            current_pos = obs[0][:3] if len(obs[0]) > 2 else np.zeros(3)
            episode_distance = np.linalg.norm(current_pos - initial_pos)
            all_trajectory['positions'].append(current_pos.copy())
            all_trajectory['distances'].append(episode_distance)
            
            # Track velocity and failure info
            velocity = 0.0
            joint_health = 1.0
            active_failures = []
            
            if info[0] is not None:
                if 'speed' in info[0]:
                    velocity = info[0]['speed']
                elif 'current_velocity' in info[0]:
                    velocity = info[0]['current_velocity']
                
                # Extract failure information
                if 'persistent_failures' in info[0]:
                    failure_info = info[0]['persistent_failures']
                    if 'num_failed_joints' in failure_info:
                        total_joints = 8
                        failed_count = failure_info['num_failed_joints']
                        joint_health = 1.0 - (failed_count / total_joints)
                        active_failures = failure_info.get('failed_joints', [])
            
            # If no velocity in info, calculate from position change
            if velocity == 0.0 and step > 0:
                if len(all_trajectory['positions']) > 1:
                    prev_pos = all_trajectory['positions'][-2]
                    pos_change = np.linalg.norm(current_pos - prev_pos)
                    velocity = pos_change * 20  # 20 Hz
            
            all_trajectory['velocities'].append(velocity)
            all_trajectory['joint_health'].append(joint_health)
            all_trajectory['active_failures'].append(active_failures)
            
            if step % 60 == 0:  # Print every 3 seconds
                print(f"    Step {step:3d}: Pos=[{current_pos[0]:.2f},{current_pos[1]:.2f}], "
                      f"Dist={episode_distance:.3f}m, Vel={velocity:.3f}m/s, Health={joint_health:.1%}")
            
            if done[0]:
                print(f"    Episode ended early at step {step}")
                break
        
        env.close()
        
        # Print episode summary
        final_pos = all_trajectory['positions'][-1]
        episode_vels = all_trajectory['velocities'][episode_start_idx:]
        episode_health = all_trajectory['joint_health'][episode_start_idx:]
        
        print(f"    Episode Summary:")
        print(f"      Final Position: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
        print(f"      Total Distance: {episode_distance:.3f} m")
        print(f"      Total Reward: {episode_reward:.1f}")
        print(f"      Avg Velocity: {np.mean(episode_vels):.3f} m/s")
        print(f"      Avg Joint Health: {np.mean(episode_health):.1%}")
        if active_failures:
            print(f"      Final Failed Joints: {active_failures}")
        
        episode_start_idx = len(all_trajectory['observations'])
    
    # Print overall summary
    print(f"\nPERSISTENT DR COMPLETE EPISODES TEST:")
    print(f"Total Episodes: {len(failure_scenarios)}")
    print(f"Total Steps: {len(all_trajectory['observations'])}")
    if all_trajectory['velocities']:
        print(f"Overall Average Velocity: {np.mean(all_trajectory['velocities']):.3f} m/s")
        print(f"Overall Average Joint Health: {np.mean(all_trajectory['joint_health']):.1%}")
        print(f"Total Distance Traveled: {sum(all_trajectory['distances']):.3f} m")
    
    return all_trajectory

def replay_with_failure_visualization(trajectory, output_path="persistent_dr_duration_test.mp4"):
    """Replay trajectory with failure duration visualization"""
    
    print(f"\nReplaying trajectory with failure duration visualization at 1920x1080...")
    
    # Create environment WITH rendering
    def make_env():
        env = gym.make('RealAntMujoco-v0', render_mode='rgb_array')
        env = SuccessRewardWrapper(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Reset environment
    obs = env.reset()
    frames = []
    
    # Try to load fonts - larger for high resolution
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 48)
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 36)
        small_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 28)
    except:
        title_font = ImageFont.load_default()
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    print(f"Replaying {len(trajectory['actions'])} steps...")
    
    for step, action in enumerate(trajectory['actions']):
        # Step environment
        obs, reward, done, info = env.step(action)
        
        # Capture frame
        frame = env.render(mode='rgb_array')
        if frame is not None:
            # Create overlay and resize to 1920x1080
            img = Image.fromarray(frame)
            img = img.resize((1920, 1080), Image.LANCZOS)
            draw = ImageDraw.Draw(img)
            
            # Get current data
            scenario = trajectory['scenarios'][step]
            episode_num = trajectory['episode_markers'][step]
            velocity = trajectory['velocities'][step] if step < len(trajectory['velocities']) else 0.0
            joint_health = trajectory['joint_health'][step] if step < len(trajectory['joint_health']) else 1.0
            active_failures = trajectory['active_failures'][step] if step < len(trajectory['active_failures']) else []
            position = trajectory['positions'][step] if step < len(trajectory['positions']) else np.zeros(3)
            distance = trajectory['distances'][step] if step < len(trajectory['distances']) else 0.0
            
            # Main title - scaled for 1920x1080
            draw.rectangle([(20, 20), (1100, 90)], fill=(20, 80, 20, 200))
            draw.text((30, 30), "Persistent DR: Complete Episodes", fill=(255, 255, 255), font=title_font)
            
            # Episode indicator with progress
            episode_progress = ((step + 1) % 400) / 400  # Progress within current episode
            episode_color = (
                int(80 + 100 * episode_progress),
                int(40 + 60 * episode_progress),
                20
            )
            draw.rectangle([(1120, 20), (1500, 90)], fill=episode_color)
            draw.text((1140, 35), f"Episode {episode_num + 1}/6", fill=(255, 255, 255), font=font)
            
            # Episode progress bar
            bar_x, bar_y = 1140, 60
            bar_width = 300
            bar_height = 10
            draw.rectangle([(bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height)], fill=(100, 100, 100))
            progress_width = int(bar_width * episode_progress)
            draw.rectangle([(bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height)], fill=(0, 255, 100))
            
            # Current scenario description
            failure_intensity = 1.0 - joint_health
            failure_color = (
                min(255, int(255 * failure_intensity)), 
                max(100, int(255 * (1 - failure_intensity))), 
                int(100 * (1 - failure_intensity))
            )
            
            draw.rectangle([(20, 110), (1400, 170)], fill=(40, 40, 40, 220))
            draw.text((30, 125), scenario['description'], fill=failure_color, font=font)
            
            # Position and distance tracking
            draw.rectangle([(20, 180), (800, 240)], fill=(20, 40, 80, 200))
            pos_text = f"Position: [{position[0]:.2f}, {position[1]:.2f}] | Distance: {distance:.2f}m"
            draw.text((30, 195), pos_text, fill=(200, 220, 255), font=small_font)
            
            # Joint health bar and indicator
            bar_width = 700
            bar_height = 30
            bar_x, bar_y = 30, 260
            
            # Background bar
            draw.rectangle([(bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height)], fill=(80, 80, 80))
            
            # Joint health bar
            health_width = int(bar_width * joint_health)
            health_color = (int(255 * (1 - joint_health)), int(255 * joint_health), 0)
            draw.rectangle([(bar_x, bar_y), (bar_x + health_width, bar_y + bar_height)], fill=health_color)
            
            # Joint health text
            draw.rectangle([(20, 300), (600, 360)], fill=(40, 40, 40, 180))
            draw.text((30, 315), f"Joint Health: {joint_health:.1%} ({8-len(active_failures)}/8 joints functional)", fill=health_color, font=small_font)
            
            # Active failures info
            if active_failures:
                draw.rectangle([(20, 370), (800, 430)], fill=(80, 20, 20, 180))
                failure_text = f"Failed Joints: {active_failures}"
                draw.text((30, 385), failure_text, fill=(255, 150, 150), font=small_font)
            
            # Step counter and progress
            draw.rectangle([(20, 450), (600, 510)], fill=(40, 40, 40, 180))
            progress = (step + 1) / len(trajectory['actions'])
            draw.text((30, 465), f"Step: {step+1} / {len(trajectory['actions'])} ({progress*100:.1f}%)", fill=(255, 255, 255), font=small_font)
            
            # Failure timing info if active
            if active_failures and scenario.get('config'):
                draw.rectangle([(20, 530), (900, 590)], fill=(20, 40, 80, 180))
                config = scenario['config']
                if 'failure_prob' in config:
                    prob_text = f"Failure Settings: {config['failure_prob']*100:.0f}% chance, "
                    prob_text += f"Max {config.get('max_failed_joints', 1)} joints"
                    draw.text((30, 545), prob_text, fill=(200, 200, 255), font=small_font)
            
            frames.append(np.array(img))
        
        # Don't break on done - we want to see the full collected trajectory
        # if done[0]:
        #     break
    
    env.close()
    
    # Save video
    if frames:
        print(f"Saving video to {output_path}...")
        imageio.mimsave(output_path, frames, fps=30)
        print(f"✅ Video saved! ({len(frames)} frames)")
        print(f"This video shows Persistent DR's response to increasing failure durations")
    else:
        print("❌ No frames captured!")
    
    return len(frames)

def create_persistent_dr_duration_video():
    """Create Persistent DR failure duration demonstration video"""
    
    print("🎬 CREATING PERSISTENT DR FAILURE DURATION VIDEO")
    print("=" * 80)
    
    # Persistent DR model path
    model_path = 'experiments/ppo_persistent_dr_resume_h96y2uqe/final_model.zip'
    
    # Realistic failure scenarios - complete episodes showing adaptation
    failure_scenarios = [
        # Episode 1: Perfect baseline
        {
            'description': 'Episode 1: Perfect Conditions - No Failures',
            'config': None  # No DR wrapper applied
        },
        
        # Episode 2: Single quick failure
        {
            'description': 'Episode 2: Quick Joint Lock (80 steps) - Tests Recovery',
            'config': {
                'failure_prob': 1.0,  # Guarantee failure occurs
                'max_failed_joints': 1,
                'short_duration': [80, 80],  # Exactly 80 steps
                'medium_duration': [200, 400],
                'duration_probs': [1.0, 0.0, 0.0],  # Only short failures
                'failure_types': ['lock'],
                'failure_type_probs': [1.0],
                'use_curriculum': False,
                'warmup_steps': 0,
                'curriculum_steps': 0
            }
        },
        
        # Episode 3: Medium failure
        {
            'description': 'Episode 3: Medium Joint Failure (300 steps) - Extended Adaptation',
            'config': {
                'failure_prob': 1.0,
                'max_failed_joints': 1,
                'short_duration': [50, 100],
                'medium_duration': [300, 300],  # Exactly 300 steps
                'duration_probs': [0.0, 1.0, 0.0],  # Only medium failures
                'failure_types': ['lock'],
                'failure_type_probs': [1.0],
                'use_curriculum': False,
                'warmup_steps': 0,
                'curriculum_steps': 0
            }
        },
        
        # Episode 4: Multiple failures
        {
            'description': 'Episode 4: Two Weak Joints (150 steps each) - Dual Adaptation',
            'config': {
                'failure_prob': 1.0,
                'max_failed_joints': 2,
                'short_duration': [150, 150],  # Exactly 150 steps
                'medium_duration': [200, 400],
                'duration_probs': [1.0, 0.0, 0.0],
                'failure_types': ['weak'],  # Weak joints instead of locked
                'failure_type_probs': [1.0],
                'use_curriculum': False,
                'warmup_steps': 0,
                'curriculum_steps': 0
            }
        },
        
        # Episode 5: Erratic behavior
        {
            'description': 'Episode 5: Erratic Joint Behavior (250 steps) - Unpredictable Motion',
            'config': {
                'failure_prob': 1.0,
                'max_failed_joints': 1,
                'short_duration': [50, 100],
                'medium_duration': [250, 250],  # Exactly 250 steps
                'duration_probs': [0.0, 1.0, 0.0],
                'failure_types': ['erratic'],  # Erratic behavior
                'failure_type_probs': [1.0],
                'use_curriculum': False,
                'warmup_steps': 0,
                'curriculum_steps': 0
            }
        },
        
        # Episode 6: Episode-long failure
        {
            'description': 'Episode 6: Permanent Episode Failure - No Recovery',
            'config': {
                'failure_prob': 1.0,
                'max_failed_joints': 1,
                'short_duration': [50, 100],
                'medium_duration': [200, 400],
                'duration_probs': [0.0, 0.0, 1.0],  # Episode-long only
                'failure_types': ['lock'],
                'failure_type_probs': [1.0],
                'use_curriculum': False,
                'warmup_steps': 0,
                'curriculum_steps': 0
            }
        }
    ]
    
    print("Testing scenarios:", len(failure_scenarios), "complete episodes")
    print("Range: Perfect conditions → Permanent episode failures")
    
    # Complete episodes - 400 steps each
    episode_length = 400  # Each episode is 400 steps (20 seconds)
    
    # Pass 1: Collect complete episodes without rendering
    trajectory = collect_complete_episodes_with_failures(model_path, failure_scenarios, episode_length)
    
    # Pass 2: Replay with visualization
    frames_rendered = replay_with_failure_visualization(trajectory, "persistent_dr_duration_robustness.mp4")
    
    print("\n" + "=" * 80)
    print("🎉 PERSISTENT DR DURATION VIDEO CREATED!")
    print("=" * 80)
    print("✅ File: persistent_dr_duration_robustness.mp4")
    print(f"✅ Frames: {frames_rendered}")
    print("✅ Shows progressive increase in failure durations")
    print("✅ Demonstrates what Persistent DR was trained for")
    print("=" * 80)
    
    return trajectory

if __name__ == "__main__":
    trajectory = create_persistent_dr_duration_video()