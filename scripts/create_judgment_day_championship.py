#!/usr/bin/env python3
"""
🏆 JUDGMENT DAY CHAMPIONSHIP - 4-MODEL ROBUSTNESS DEMONSTRATION 🏆

Professional championship video for thesis defense
Shows all 4 RETRAINED models simultaneously:
- M1: Baseline PPO (Pure PPO, 32M steps)
- M2: SR2L (Sensor noise specialist, 32M steps)
- M3: Domain Randomization (Actuator failure specialist, 32M steps)
- M4: Combo (SR2L + DR combination, 32M steps)

Testing:
- Normal walking baseline
- 5 sensor noise levels (0.01 → 0.5)
- 8 joint failures (all hips + ankles)

Output: 6-minute championship video with 4-quadrant layout
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

class JudgmentDayChampionship:
    """4-Quadrant Championship Video Creator for Thesis Defense"""

    def __init__(self):
        # Frame sizes
        self.quadrant_size = (960, 540)
        self.full_frame_size = (1920, 1080)
        self.fps = 60

        # 4 RETRAINED Models (all 32M steps for fair comparison)
        self.models = {
            'M1': {
                'name': 'M1 Baseline',
                'path': 'experiments/M1_baseline_32M_RETRAINED_ym2jcllj/final_model',
                'vec_path': 'experiments/M1_baseline_32M_RETRAINED_ym2jcllj/vec_normalize.pkl',
                'color': (0, 215, 255),  # Gold (BGR)
                'position': 'top_left',
                'specialty': 'Pure PPO'
            },
            'M2': {
                'name': 'M2 SR2L',
                'path': 'experiments/M2_sr2l_32M_RETRAINED_ze09p0vf/final_model',
                'vec_path': 'experiments/M2_sr2l_32M_RETRAINED_ze09p0vf/vec_normalize.pkl',
                'color': (0, 255, 0),  # Green
                'position': 'top_right',
                'specialty': 'Sensor Noise Robust'
            },
            'M3': {
                'name': 'M3 DR',
                'path': 'experiments/M3_dr_v2_single_failures_32M_15cxapkl/final_model',
                'vec_path': 'experiments/M3_dr_v2_single_failures_32M_15cxapkl/vec_normalize.pkl',
                'color': (255, 0, 255),  # Magenta
                'position': 'bottom_left',
                'specialty': 'Actuator Failure Robust'
            },
            'M4': {
                'name': 'M4 Combo',
                'path': 'done/ultimate_robustness_combo_ju7lfsk2/final_model',
                'vec_path': 'done/ultimate_robustness_combo_ju7lfsk2/vec_normalize.pkl',
                'color': (0, 255, 255),  # Yellow
                'position': 'bottom_right',
                'specialty': 'SR2L + DR Combined'
            }
        }

        # Test scenarios (14 total, ~6 minutes)
        self.test_sections = [
            # 1. Baseline (30s)
            {
                'name': 'NORMAL WALKING',
                'type': 'baseline',
                'duration_frames': 1800,  # 30 seconds
                'noise_level': 0.0,
                'failed_joints': []
            },

            # 2-6. Sensor Noise Tests (5 × 15s = 75s)
            {
                'name': 'SENSOR NOISE 0.01',
                'type': 'noise',
                'duration_frames': 900,  # 15 seconds
                'noise_level': 0.01,
                'failed_joints': []
            },
            {
                'name': 'SENSOR NOISE 0.05',
                'type': 'noise',
                'duration_frames': 900,
                'noise_level': 0.05,
                'failed_joints': []
            },
            {
                'name': 'SENSOR NOISE 0.1',
                'type': 'noise',
                'duration_frames': 900,
                'noise_level': 0.1,
                'failed_joints': []
            },
            {
                'name': 'SENSOR NOISE 0.3',
                'type': 'noise',
                'duration_frames': 900,
                'noise_level': 0.3,
                'failed_joints': []
            },
            {
                'name': 'SENSOR NOISE 0.5',
                'type': 'noise',
                'duration_frames': 900,
                'noise_level': 0.5,
                'failed_joints': []
            },

            # 7-14. Joint Failures (8 × 30s = 240s)
            {
                'name': 'HIP_1 LOCKED',
                'type': 'failure',
                'duration_frames': 1800,  # 30 seconds
                'noise_level': 0.0,
                'failed_joints': ['hip_1']
            },
            {
                'name': 'ANKLE_1 LOCKED',
                'type': 'failure',
                'duration_frames': 1800,
                'noise_level': 0.0,
                'failed_joints': ['ankle_1']
            },
            {
                'name': 'HIP_2 LOCKED',
                'type': 'failure',
                'duration_frames': 1800,
                'noise_level': 0.0,
                'failed_joints': ['hip_2']
            },
            {
                'name': 'ANKLE_2 LOCKED',
                'type': 'failure',
                'duration_frames': 1800,
                'noise_level': 0.0,
                'failed_joints': ['ankle_2']
            },
            {
                'name': 'HIP_3 LOCKED',
                'type': 'failure',
                'duration_frames': 1800,
                'noise_level': 0.0,
                'failed_joints': ['hip_3']
            },
            {
                'name': 'ANKLE_3 LOCKED',
                'type': 'failure',
                'duration_frames': 1800,
                'noise_level': 0.0,
                'failed_joints': ['ankle_3']
            },
            {
                'name': 'HIP_4 LOCKED',
                'type': 'failure',
                'duration_frames': 1800,
                'noise_level': 0.0,
                'failed_joints': ['hip_4']
            },
            {
                'name': 'ANKLE_4 LOCKED',
                'type': 'failure',
                'duration_frames': 1800,
                'noise_level': 0.0,
                'failed_joints': ['ankle_4']
            }
        ]

        # Joint to action index mapping
        self.joint_to_action_index = {
            'hip_1': 0, 'ankle_1': 1,
            'hip_2': 2, 'ankle_2': 3,
            'hip_3': 4, 'ankle_3': 5,
            'hip_4': 6, 'ankle_4': 7
        }

        # Baseline velocities (set from first test)
        self.baseline_velocities = {}

        # Colors
        self.colors = {
            'background': (0, 0, 0),
            'text': (255, 255, 255),
            'champion': (0, 215, 255),  # Gold
            'excellent': (0, 255, 0),  # Green
            'good': (0, 255, 255),  # Yellow
            'challenge': (0, 165, 255),  # Orange
            'extreme': (0, 0, 255),  # Red
            'failure': (255, 0, 255)  # Magenta
        }

    def apply_sensor_noise(self, obs, noise_std, rng):
        """Apply Gaussian noise to joint observations (dims 13-28)"""
        if noise_std <= 0:
            return obs

        obs_copy = obs.copy()
        joint_start = 13
        joint_end = 29  # 16 joint sensor values

        for idx in range(joint_start, min(joint_end, len(obs_copy[0]))):
            noise = rng.normal(0, noise_std)
            obs_copy[0][idx] += noise

        return obs_copy

    def apply_joint_failure(self, action, failed_joints, step, delay=120):
        """Lock joints after warmup period (2 seconds)"""
        if not failed_joints or step < delay:
            return action

        action_copy = action.copy()
        for joint_name in failed_joints:
            joint_idx = self.joint_to_action_index[joint_name]
            if joint_idx < len(action_copy[0]):
                action_copy[0][joint_idx] = 0.0  # Lock joint

        return action_copy

    def get_performance_color(self, retention_pct):
        """Get color based on retention percentage"""
        if retention_pct >= 95:
            return self.colors['champion']
        elif retention_pct >= 80:
            return self.colors['excellent']
        elif retention_pct >= 60:
            return self.colors['good']
        elif retention_pct >= 40:
            return self.colors['challenge']
        else:
            return self.colors['extreme']

    def setup_topdown_camera(self, env):
        """Configure top-down camera view"""
        try:
            if hasattr(env.envs[0], 'unwrapped'):
                unwrapped = env.envs[0].unwrapped
                if hasattr(unwrapped, 'model') and hasattr(unwrapped, 'data'):
                    model = unwrapped.model
                    model.cam_pos[0] = [0, -8, 8]
                    model.cam_quat[0] = [0.7071, 0.7071, 0, 0]
                    return True
        except Exception as e:
            print(f"  Camera setup: {e}")
        return False

    def create_quadrant_overlay(self, frame, model_key, test_name,
                                velocity, distance, retention, progress):
        """Create compact overlay for each quadrant (960×540)"""
        h, w = frame.shape[:2]

        # Semi-transparent header
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), self.colors['background'], -1)
        frame = cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)

        # Model name (colored)
        model_info = self.models[model_key]
        cv2.putText(frame, model_info['name'], (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, model_info['color'], 2, cv2.LINE_AA)

        # Metrics
        cv2.putText(frame, f"{velocity:.3f} m/s", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1, cv2.LINE_AA)
        cv2.putText(frame, f"{distance:.1f}m", (150, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1, cv2.LINE_AA)

        # Retention (color-coded)
        ret_color = self.get_performance_color(retention)
        cv2.putText(frame, f"{retention:.0f}%", (w - 80, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, ret_color, 2, cv2.LINE_AA)

        # Progress bar
        bar_width = w - 20
        cv2.rectangle(frame, (10, 75), (10 + int(bar_width * progress), 80),
                     model_info['color'], -1)

        return frame

    def create_master_overlay(self, frame, section, progress):
        """Create master overlay across full frame"""
        h, w = frame.shape[:2]

        # Top banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), self.colors['background'], -1)
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)

        # Title
        cv2.putText(frame, "4-MODEL ROBUSTNESS CHAMPIONSHIP",
                   (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                   self.colors['champion'], 3, cv2.LINE_AA)

        # Current test
        cv2.putText(frame, f"TEST: {section['name']}",
                   (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                   self.colors['text'], 2, cv2.LINE_AA)

        # Progress
        cv2.putText(frame, f"Progress: {progress*100:.0f}%",
                   (w - 250, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                   self.colors['text'], 2, cv2.LINE_AA)

        return frame

    def record_championship(self, output_path):
        """Record complete championship with 2-pass rendering"""
        print("=" * 80)
        print("🏆 JUDGMENT DAY CHAMPIONSHIP - 4-MODEL DEMONSTRATION")
        print("=" * 80)

        total_sections = len(self.test_sections)
        all_section_data = []

        # ========== PASS 1: COLLECT TRAJECTORIES ==========
        print("\n🔬 PASS 1: Collecting Performance Data (NO rendering)")
        print("=" * 80)

        for section_idx, section in enumerate(self.test_sections):
            print(f"\n{'='*80}")
            print(f"SECTION {section_idx+1}/{total_sections}: {section['name']}")
            print(f"Duration: {section['duration_frames']/self.fps:.1f}s")
            print(f"{'='*80}")

            # Load all 4 models WITHOUT rendering
            models_data = {}
            for model_key in ['M1', 'M2', 'M3', 'M4']:
                print(f"  Loading {self.models[model_key]['name']}...")

                # Create environment WITHOUT render_mode
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
                    print(f"    No VecNormalize for {model_key}")

                model = PPO.load(model_info['path'])
                obs = env.reset()

                models_data[model_key] = {
                    'model': model,
                    'env': env,
                    'info': model_info,
                    'obs': obs,
                    'positions': [],
                    'actions': [],
                    'rng': np.random.RandomState(42),
                    'step': 0
                }

            # Collect trajectories
            steps_collected = 0
            while steps_collected < section['duration_frames']:
                for model_key in ['M1', 'M2', 'M3', 'M4']:
                    data = models_data[model_key]

                    # Apply noise
                    obs_input = data['obs']
                    if section['noise_level'] > 0:
                        obs_input = self.apply_sensor_noise(
                            obs_input, section['noise_level'], data['rng']
                        )

                    # Get action
                    action, _ = data['model'].predict(obs_input, deterministic=True)

                    # Apply joint failure
                    if section['failed_joints']:
                        action = self.apply_joint_failure(
                            action, section['failed_joints'], data['step']
                        )

                    # Store action
                    data['actions'].append(action.copy())

                    # Step environment
                    obs, reward, done, info = data['env'].step(action)
                    data['obs'] = obs
                    data['step'] += 1

                    # Track position
                    x_pos = data['env'].envs[0].unwrapped.data.qpos[0]
                    data['positions'].append(x_pos)

                    # Reset if done
                    if done[0]:
                        data['obs'] = data['env'].reset()
                        data['step'] = 0

                steps_collected += 1

                if steps_collected % 300 == 0:
                    print(f"    Collected: {steps_collected}/{section['duration_frames']} steps")

            # Calculate performance
            section_data = {
                'section': section,
                'model_trajectories': {}
            }

            for model_key in ['M1', 'M2', 'M3', 'M4']:
                data = models_data[model_key]
                positions = data['positions']

                if len(positions) >= 2:
                    distance = positions[-1] - positions[0]
                    time_elapsed = len(positions) * 0.05  # 50ms per step
                    velocity = distance / time_elapsed if time_elapsed > 0 else 0

                    # Set baseline from first section
                    if section_idx == 0:
                        self.baseline_velocities[model_key] = velocity
                        retention = 100.0
                    else:
                        baseline = self.baseline_velocities.get(model_key, velocity)
                        retention = (velocity / baseline * 100) if baseline > 0 else 0

                    section_data['model_trajectories'][model_key] = {
                        'actions': data['actions'],
                        'positions': positions,
                        'velocity': velocity,
                        'distance': distance,
                        'retention': retention
                    }

                    print(f"  {self.models[model_key]['name']}: {velocity:.3f} m/s ({retention:.1f}%)")

                # Close environment
                data['env'].close()

            all_section_data.append(section_data)

        # ========== PASS 2: RENDER VIDEO ==========
        print("\n\n🎬 PASS 2: Rendering Championship Video")
        print("=" * 80)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, self.full_frame_size)

        for section_idx, section_data in enumerate(all_section_data):
            section = section_data['section']
            print(f"\nRendering: {section['name']}")

            # Load all 4 models WITH rendering
            render_envs = {}
            for model_key in ['M1', 'M2', 'M3', 'M4']:
                def make_env_render():
                    base_env = gym.make('RealAntMujoco-v0',
                                       render_mode='rgb_array',
                                       disable_env_checker=True)
                    while isinstance(base_env, TimeLimit):
                        base_env = base_env.env
                    env = TimeLimit(base_env, max_episode_steps=2500)
                    env = SuccessRewardWrapper(env)
                    return env

                env = DummyVecEnv([make_env_render])
                model_info = self.models[model_key]

                try:
                    env = VecNormalize.load(model_info['vec_path'], env)
                    env.training = False
                    env.norm_reward = False
                except:
                    pass

                env.reset()
                self.setup_topdown_camera(env)
                render_envs[model_key] = env

            # Replay and render
            duration_frames = section['duration_frames']
            for step in range(duration_frames):
                quadrant_frames = {}

                for model_key in ['M1', 'M2', 'M3', 'M4']:
                    traj = section_data['model_trajectories'][model_key]

                    # Replay action
                    if step < len(traj['actions']):
                        action = traj['actions'][step]
                        obs, reward, done, info = render_envs[model_key].step(action)

                        # Render frame
                        frame = render_envs[model_key].envs[0].render()
                        frame = cv2.resize(frame, self.quadrant_size)

                        # Calculate metrics
                        positions = traj['positions']
                        if step < len(positions):
                            distance = positions[step] - positions[0]
                            velocity = distance / (step * 0.05) if step > 0 else 0
                            baseline = self.baseline_velocities.get(model_key, velocity)
                            retention = (velocity / baseline * 100) if baseline > 0 else 100

                            # Add overlay
                            frame = self.create_quadrant_overlay(
                                frame, model_key, section['name'],
                                velocity, distance, retention, step / duration_frames
                            )

                        quadrant_frames[model_key] = frame

                        if done[0]:
                            render_envs[model_key].reset()
                            self.setup_topdown_camera(render_envs[model_key])

                # Compose 2×2 grid
                top_row = np.hstack([quadrant_frames['M1'], quadrant_frames['M2']])
                bottom_row = np.hstack([quadrant_frames['M3'], quadrant_frames['M4']])
                combined = np.vstack([top_row, bottom_row])

                # Add master overlay
                overall_progress = (section_idx + step/duration_frames) / total_sections
                combined = self.create_master_overlay(combined, section, overall_progress)

                video_writer.write(combined)

                if (step + 1) % 300 == 0:
                    print(f"  Rendered: {step+1}/{duration_frames} frames")

            # Close render environments
            for env in render_envs.values():
                env.close()

        video_writer.release()

        # Save performance JSON
        json_path = output_path.replace('.mp4', '_performance.json')
        json_data = []
        for section_data in all_section_data:
            section_json = {
                'test': section_data['section']['name'],
                'type': section_data['section']['type'],
                'models': {}
            }
            for model_key, traj in section_data['model_trajectories'].items():
                section_json['models'][model_key] = {
                    'velocity': traj['velocity'],
                    'distance': traj['distance'],
                    'retention': traj['retention']
                }
            json_data.append(section_json)

        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        print("\n" + "=" * 80)
        print("✅ JUDGMENT DAY CHAMPIONSHIP COMPLETE!")
        print("=" * 80)
        print(f"📹 Video: {output_path}")
        print(f"📊 Performance: {json_path}")
        print(f"⏱️  Duration: {sum(s['section']['duration_frames'] for s in all_section_data)/self.fps/60:.1f} minutes")
        print("\n🏆 READY FOR THESIS DEFENSE! 🎓")


def main():
    """Create judgment day championship video"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"videos/JUDGMENT_DAY_CHAMPIONSHIP_{timestamp}.mp4"

    os.makedirs("videos", exist_ok=True)

    recorder = JudgmentDayChampionship()
    recorder.record_championship(output_path)


if __name__ == "__main__":
    main()
