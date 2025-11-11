#!/usr/bin/env python3
"""
🏆 4-QUADRANT COMPREHENSIVE ROBUSTNESS CHAMPIONSHIP 🏆
Shows all 4 models (Baseline, SR2L, DR V7.7E, Ultimate Combo) simultaneously
Testing: Normal walking, sensor noise, joint failures, combined stress
"""
import sys
sys.path.append('src')

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
import realant_sim
import cv2
import os
from datetime import datetime
import json

class QuadrantChampionshipRecorder:
    """4-Quadrant Championship Video Creator"""

    def __init__(self):
        # Single quadrant size (half of 1920x1080)
        self.quadrant_size = (960, 540)
        self.full_frame_size = (1920, 1080)
        self.fps = 60

        # Model configurations (paths WITHOUT .zip - SB3 adds .zip automatically!)
        self.models = {
            'baseline': {
                'name': 'PPO Baseline',
                'path': '../done/ppo_baseline_ueqbjf2x/best_model/best_model',
                'vec_path': '../done/ppo_baseline_ueqbjf2x/vec_normalize.pkl',
                'color': (0, 215, 255),  # Gold (BGR)
                'position': 'top_left'
            },
            'sr2l': {
                'name': 'PPO + SR2L',
                'path': '../done/ppo_sr2l_forward_m7gtjtpa/final_model',
                'vec_path': '../done/ppo_sr2l_forward_m7gtjtpa/vec_normalize.pkl',
                'color': (0, 255, 0),  # Green
                'position': 'top_right'
            },
            'dr': {
                'name': 'PPO + DR (V7.7E)',
                'path': '../done/v7_7e_ultra_speed_jtfwl2qf/final_model',
                'vec_path': '../done/v7_7e_ultra_speed_jtfwl2qf/vec_normalize.pkl',
                'color': (255, 0, 255),  # Magenta
                'position': 'bottom_left'
            },
            'combo': {
                'name': 'Ultimate Combo',
                'path': '../done/ultimate_robustness_combo_ju7lfsk2/final_model',
                'vec_path': '../done/ultimate_robustness_combo_ju7lfsk2/vec_normalize.pkl',
                'color': (0, 255, 255),  # Yellow
                'position': 'bottom_right'
            }
        }

        # Test sections with matching noise levels from extreme_championship.py
        self.test_sections = [
            {
                'name': 'NORMAL WALKING',
                'type': 'normal',
                'duration_frames': 900,  # 15 seconds
                'noise_level': 0.0,
                'failed_joints': []
            },
            # Noise levels matching extreme_championship.py
            {
                'name': 'NOISE: 0.0 (Baseline)',
                'type': 'noise',
                'duration_frames': 900,
                'noise_level': 0.0,
                'failed_joints': []
            },
            {
                'name': 'NOISE: 0.01 (1X Training)',
                'type': 'noise',
                'duration_frames': 900,
                'noise_level': 0.01,
                'failed_joints': []
            },
            {
                'name': 'NOISE: 0.05 (5X Training)',
                'type': 'noise',
                'duration_frames': 900,
                'noise_level': 0.05,
                'failed_joints': []
            },
            {
                'name': 'NOISE: 0.1 (10X Training)',
                'type': 'noise',
                'duration_frames': 900,
                'noise_level': 0.1,
                'failed_joints': []
            },
            {
                'name': 'NOISE: 0.2 (20X Training)',
                'type': 'noise',
                'duration_frames': 900,
                'noise_level': 0.2,
                'failed_joints': []
            },
            {
                'name': 'NOISE: 0.3 (30X Training)',
                'type': 'noise',
                'duration_frames': 900,
                'noise_level': 0.3,
                'failed_joints': []
            },
            {
                'name': 'NOISE: 0.5 (50X Training)',
                'type': 'noise',
                'duration_frames': 900,
                'noise_level': 0.5,
                'failed_joints': []
            },
            {
                'name': 'NOISE: 0.7 (70X Training)',
                'type': 'noise',
                'duration_frames': 900,
                'noise_level': 0.7,
                'failed_joints': []
            },
            # Joint failures - Best performers from V7.7E testing
            {
                'name': 'HIP_1 LOCKED (Best: 81.8%)',
                'type': 'joint_failure',
                'duration_frames': 1800,  # 30 seconds like DR championship
                'noise_level': 0.0,
                'failed_joints': ['hip_1']
            },
            {
                'name': 'HIP_4 LOCKED (Rear: 43.5%)',
                'type': 'joint_failure',
                'duration_frames': 1800,
                'noise_level': 0.0,
                'failed_joints': ['hip_4']
            },
            {
                'name': 'ANKLE_2 LOCKED (Best: 59.4%)',
                'type': 'joint_failure',
                'duration_frames': 1800,
                'noise_level': 0.0,
                'failed_joints': ['ankle_2']
            },
            {
                'name': 'ANKLE_3 LOCKED (Rear: 43.6%)',
                'type': 'joint_failure',
                'duration_frames': 1800,
                'noise_level': 0.0,
                'failed_joints': ['ankle_3']
            },
            # Combined stress tests
            {
                'name': 'COMBINED: Noise 0.05 + Ankle_3',
                'type': 'combined',
                'duration_frames': 1800,
                'noise_level': 0.05,
                'failed_joints': ['ankle_3']
            },
            {
                'name': 'COMBINED: Noise 0.1 + Hip_4',
                'type': 'combined',
                'duration_frames': 1800,
                'noise_level': 0.1,
                'failed_joints': ['hip_4']
            }
        ]

        # Joint to action index mapping
        self.joint_to_action_index = {
            'hip_1': 0,
            'ankle_1': 1,
            'hip_2': 2,
            'ankle_2': 3,
            'hip_3': 4,
            'ankle_3': 5,
            'hip_4': 6,
            'ankle_4': 7
        }

        # Baseline performances for each model
        self.baseline_velocities = {}

        # Colors
        self.colors = {
            'background': (0, 0, 0),
            'text': (255, 255, 255),
            'excellent': (0, 255, 0),
            'good': (0, 255, 255),
            'challenge': (0, 165, 255),
            'extreme': (0, 0, 255),
            'failure': (255, 0, 255)
        }

    def apply_sensor_noise(self, obs, noise_std, rng):
        """Apply sensor noise ONLY to joint observations (dims 13-28)"""
        if noise_std <= 0:
            return obs

        obs_copy = obs.copy()
        joint_start = 13
        joint_end = 29  # 16 joint sensor values

        for idx in range(joint_start, min(joint_end, len(obs_copy[0]))):
            noise = rng.normal(0, noise_std)
            obs_copy[0][idx] += noise

        return obs_copy

    def apply_joint_failure(self, action, failed_joint_names, step, delay=120):
        """Apply joint failure with delayed locking (matching DR championship)"""
        if not failed_joint_names or step < delay:
            return action

        action_copy = action.copy()
        for joint_name in failed_joint_names:
            if joint_name in self.joint_to_action_index:
                joint_idx = self.joint_to_action_index[joint_name]
                if joint_idx < len(action_copy[0]):
                    action_copy[0][joint_idx] = 0.0

        return action_copy

    def create_quadrant_overlay(self, frame, model_name, model_color, velocity, distance,
                                retention, section_name, progress):
        """Create overlay for a single quadrant"""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Semi-transparent header
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), self.colors['background'], -1)
        cv2.rectangle(overlay, (0, h-80), (w, h), self.colors['background'], -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Model name
        cv2.putText(frame, model_name, (10, 30), font, 0.7, model_color, 2)

        # Current test
        cv2.putText(frame, section_name, (10, 60), font, 0.4, self.colors['text'], 1)

        # Metrics
        cv2.putText(frame, f"V: {velocity:.3f} m/s", (10, h-50), font, 0.4, self.colors['text'], 1)
        cv2.putText(frame, f"D: {distance:.1f}m", (10, h-30), font, 0.4, self.colors['text'], 1)

        # Retention
        if retention >= 90:
            ret_color = self.colors['excellent']
        elif retention >= 70:
            ret_color = self.colors['good']
        elif retention >= 50:
            ret_color = self.colors['challenge']
        else:
            ret_color = self.colors['extreme']

        cv2.putText(frame, f"Ret: {retention:.1f}%", (10, h-10), font, 0.4, ret_color, 1)

        # Progress bar
        bar_width = w - 20
        bar_height = 5
        bar_x = 10
        bar_y = 85

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (100, 100, 100), -1)
        progress_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height),
                     model_color, -1)

        return frame

    def create_master_overlay(self, combined_frame, section_name, progress, section_num, total_sections):
        """Create master overlay for the full 4-quadrant frame"""
        h, w = combined_frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Semi-transparent top banner
        overlay = combined_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), self.colors['background'], -1)
        combined_frame = cv2.addWeighted(combined_frame, 0.8, overlay, 0.2, 0)

        # Championship title
        title = "4-MODEL ROBUSTNESS CHAMPIONSHIP"
        cv2.putText(combined_frame, title, (w//2 - 400, 40), font, 1.5, (0, 215, 255), 3)

        # Section info
        section_text = f"SECTION {section_num}/{total_sections}: {section_name}"
        cv2.putText(combined_frame, section_text, (w//2 - 350, 80), font, 0.8,
                   self.colors['text'], 2)

        # Overall progress
        progress_text = f"Overall Progress: {progress*100:.0f}%"
        cv2.putText(combined_frame, progress_text, (w//2 - 150, 110), font, 0.6,
                   self.colors['text'], 1)

        return combined_frame

    def load_model_and_env(self, model_key):
        """Load a model and its environment"""
        model_info = self.models[model_key]

        # Create environment
        def make_env():
            base_env = gym.make('RealAntMujoco-v0', render_mode='rgb_array',
                               disable_env_checker=True)

            # Remove existing TimeLimit
            while isinstance(base_env, TimeLimit):
                base_env = base_env.env

            # Extended episodes
            env = TimeLimit(base_env, max_episode_steps=2500)
            env = SuccessRewardWrapper(env)
            return env

        env = DummyVecEnv([make_env])

        # Load normalization if available
        try:
            env = VecNormalize.load(model_info['vec_path'], env)
            env.training = False
            env.norm_reward = False
        except:
            pass

        # Load model
        model = PPO.load(model_info['path'])

        return model, env, model_info

    def record_championship(self, output_path):
        """Record the complete 4-quadrant championship with 2-pass rendering"""
        print("=" * 80, flush=True)
        print("4-QUADRANT ROBUSTNESS CHAMPIONSHIP (2-PASS RENDERING)", flush=True)
        print("=" * 80, flush=True)

        total_sections = len(self.test_sections)
        all_trajectories = []  # Store trajectories for pass 2
        performance_data = []

        # ========== PASS 1: COLLECT TRAJECTORIES (NO RENDERING) ==========
        print("\n🔬 PASS 1: Collecting accurate performance data (NO rendering overhead)", flush=True)
        print("=" * 80, flush=True)

        for section_idx, section in enumerate(self.test_sections):
            print(f"\n{'='*80}", flush=True)
            print(f"SECTION {section_idx+1}/{total_sections}: {section['name']}", flush=True)
            print(f"Duration: {section['duration_frames']/self.fps:.1f}s", flush=True)
            print(f"{'='*80}", flush=True)

            # Load all 4 models and environments (NO render mode for Pass 1)
            models_data = {}
            for model_key in ['baseline', 'sr2l', 'dr', 'combo']:
                print(f"  Loading {self.models[model_key]['name']}...", flush=True)

                # Create environment WITHOUT rendering for Pass 1
                def make_env_no_render():
                    base_env = gym.make('RealAntMujoco-v0', disable_env_checker=True)
                    while isinstance(base_env, TimeLimit):
                        base_env = base_env.env
                    env = TimeLimit(base_env, max_episode_steps=2500)
                    env = SuccessRewardWrapper(env)
                    return env

                env = DummyVecEnv([make_env_no_render])
                model_info = self.models[model_key]

                try:
                    env = VecNormalize.load(model_info['vec_path'], env)
                    env.training = False
                    env.norm_reward = False
                except:
                    pass

                model = PPO.load(model_info['path'])
                obs = env.reset()

                models_data[model_key] = {
                    'model': model,
                    'env': env,
                    'info': model_info,
                    'obs': obs,
                    'positions': [],
                    'actions': [],  # Store actions for replay
                    'observations': [],  # Store observations for replay
                    'rewards': [],
                    'rng': np.random.RandomState(42),
                    'step': 0
                }

            steps_collected = 0

            # Collect trajectory WITHOUT rendering
            while steps_collected < section['duration_frames']:
                for model_key in ['baseline', 'sr2l', 'dr', 'combo']:
                    data = models_data[model_key]

                    # Apply noise if specified
                    obs_input = data['obs']
                    if section['noise_level'] > 0:
                        obs_input = self.apply_sensor_noise(
                            obs_input, section['noise_level'], data['rng']
                        )

                    # Get action
                    action, _ = data['model'].predict(obs_input, deterministic=True)

                    # Apply joint failures if specified
                    if section['failed_joints']:
                        action = self.apply_joint_failure(
                            action, section['failed_joints'], data['step']
                        )

                    # Store for replay
                    data['actions'].append(action.copy())
                    data['observations'].append(data['obs'].copy())

                    # Step environment (NO RENDERING)
                    obs, reward, done, info = data['env'].step(action)
                    data['obs'] = obs
                    data['step'] += 1

                    # Track metrics
                    x_pos = data['env'].envs[0].unwrapped.data.qpos[0]
                    data['positions'].append(x_pos)
                    data['rewards'].append(reward[0])

                    # Reset if done
                    if done[0]:
                        data['obs'] = data['env'].reset()
                        data['step'] = 0

                steps_collected += 1

                if steps_collected % 300 == 0:
                    print(f"    Collected: {steps_collected}/{section['duration_frames']} steps", flush=True)

            # Calculate and store section performance
            section_perf = {
                'section': section['name'],
                'type': section['type'],
                'noise_level': section['noise_level'],
                'failed_joints': section['failed_joints'],
                'models': {}
            }

            for model_key in ['baseline', 'sr2l', 'dr', 'combo']:
                data = models_data[model_key]
                if len(data['positions']) >= 2:
                    distance = data['positions'][-1] - data['positions'][0]
                    time_elapsed = len(data['positions']) * 0.05
                    velocity = distance / time_elapsed
                    baseline = self.baseline_velocities.get(model_key, velocity)
                    retention = (velocity / baseline * 100) if baseline > 0 else 0

                    # Set baseline from first section
                    if model_key not in self.baseline_velocities:
                        self.baseline_velocities[model_key] = velocity

                    section_perf['models'][model_key] = {
                        'velocity': velocity,
                        'distance': distance,
                        'retention': retention
                    }

            performance_data.append(section_perf)

            # Store trajectories for Pass 2
            all_trajectories.append({
                'section': section,
                'section_idx': section_idx,
                'models_data': models_data  # Keep the data with actions/observations
            })

            print(f"   Section {section_idx+1} complete - Metrics collected", flush=True)

        # ========== PASS 2: REPLAY AND RENDER ==========
        print("\n\n🎬 PASS 2: Replaying trajectories WITH rendering for video generation", flush=True)
        print("=" * 80, flush=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, self.full_frame_size)
        total_frames = 0

        for traj_data in all_trajectories:
            section = traj_data['section']
            section_idx = traj_data['section_idx']
            saved_models_data = traj_data['models_data']

            print(f"\n  Rendering Section {section_idx+1}/{total_sections}: {section['name']}", flush=True)

            # Create NEW environments WITH rendering for replay
            render_envs = {}
            for model_key in ['baseline', 'sr2l', 'dr', 'combo']:
                def make_env_with_render():
                    base_env = gym.make('RealAntMujoco-v0', render_mode='rgb_array', disable_env_checker=True)
                    while isinstance(base_env, TimeLimit):
                        base_env = base_env.env
                    env = TimeLimit(base_env, max_episode_steps=2500)
                    env = SuccessRewardWrapper(env)
                    return env

                env = DummyVecEnv([make_env_with_render])
                model_info = self.models[model_key]

                try:
                    env = VecNormalize.load(model_info['vec_path'], env)
                    env.training = False
                    env.norm_reward = False
                except:
                    pass

                obs = env.reset()
                render_envs[model_key] = {'env': env, 'obs': obs}

            # Replay stored actions and render
            num_steps = len(saved_models_data['baseline']['actions'])
            for step_idx in range(min(num_steps, section['duration_frames'])):
                quadrant_frames = {}

                for model_key in ['baseline', 'sr2l', 'dr', 'combo']:
                    saved_data = saved_models_data[model_key]
                    render_data = render_envs[model_key]

                    # Get saved action
                    action = saved_data['actions'][step_idx]

                    # Step environment with saved action
                    obs, reward, done, info = render_data['env'].step(action)
                    render_data['obs'] = obs

                    # Get performance metrics from Pass 1
                    if step_idx < len(saved_data['positions']):
                        positions = saved_data['positions'][:step_idx+1]
                        if len(positions) >= 2:
                            distance = positions[-1] - positions[0]
                            time_elapsed = len(positions) * 0.05
                            velocity = distance / time_elapsed
                        else:
                            distance = 0
                            velocity = 0

                        baseline = self.baseline_velocities.get(model_key, velocity)
                        retention = (velocity / baseline * 100) if baseline > 0 else 0
                    else:
                        velocity = 0
                        distance = 0
                        retention = 0

                    # Render frame
                    frame = render_data['env'].envs[0].render()
                    if frame is not None:
                        frame = cv2.resize(frame, self.quadrant_size)

                        # Add quadrant overlay
                        progress = step_idx / section['duration_frames']
                        frame = self.create_quadrant_overlay(
                            frame, saved_data['info']['name'], saved_data['info']['color'],
                            velocity, distance, retention, section['name'], progress
                        )

                        quadrant_frames[model_key] = frame

                    # Reset if done
                    if done[0]:
                        render_data['obs'] = render_data['env'].reset()

                # Combine quadrants
                if len(quadrant_frames) == 4:
                    top_row = np.hstack([quadrant_frames['baseline'], quadrant_frames['sr2l']])
                    bottom_row = np.hstack([quadrant_frames['dr'], quadrant_frames['combo']])
                    combined_frame = np.vstack([top_row, bottom_row])

                    # Add master overlay
                    overall_progress = (section_idx + step_idx/section['duration_frames']) / total_sections
                    combined_frame = self.create_master_overlay(
                        combined_frame, section['name'], overall_progress,
                        section_idx + 1, total_sections
                    )

                    video_writer.write(combined_frame)
                    total_frames += 1

            # Clean up render environments
            for model_key in ['baseline', 'sr2l', 'dr', 'combo']:
                render_envs[model_key]['env'].close()
                saved_models_data[model_key]['env'].close()  # Close Pass 1 env too

            print(f"     Rendered {min(num_steps, section['duration_frames'])} frames", flush=True)

        video_writer.release()

        # Save performance data
        with open(output_path.replace('.mp4', '_performance.json'), 'w') as f:
            json.dump(performance_data, f, indent=2)

        print(f"\n{'='*80}")
        print(f"4-QUADRANT CHAMPIONSHIP COMPLETE!")
        print(f"Video: {output_path}")
        print(f"Total frames: {total_frames}")
        print(f"Duration: {total_frames/self.fps:.1f} seconds ({total_frames/self.fps/60:.1f} minutes)")
        print(f"{'='*80}")

        return output_path, performance_data

def main():
    """Create the 4-quadrant championship video"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"evaluations/4_quadrant_championship_{timestamp}.mp4"

    os.makedirs("evaluations", exist_ok=True)

    recorder = QuadrantChampionshipRecorder()

    video_path, performance_data = recorder.record_championship(output_path)

    print(f"\n{'='*80}")
    print(f"CHAMPIONSHIP SUMMARY")
    print(f"{'='*80}")

    for section_data in performance_data:
        print(f"\n{section_data['section']}:")
        for model_key, model_perf in section_data['models'].items():
            model_name = recorder.models[model_key]['name']
            print(f"  {model_name}: {model_perf['velocity']:.3f} m/s ({model_perf['retention']:.1f}% retention)")

if __name__ == "__main__":
    main()
