#!/usr/bin/env python3
"""
EXPERIMENT 7: JOINT FAILURE + NOISE TYPE ABLATION
Tests all 4 models with joint failures COMBINED with different noise types
Full factorial design: 8 joints × 4 noise conditions × 4 models

NEW EXPERIMENT (Oct 19, 2025):
This tests interaction effects between mechanical failures and sensor noise.
Research question: Do different noise types affect recovery from joint failures differently?

Noise Conditions:
1. No noise (baseline joint failure)
2. Gaussian σ=0.1
3. Poisson λ=1.0 (SNR-matched to Gaussian)
4. Salt-and-Pepper p=0.0044 (SNR-matched to Gaussian)

Metrics: Distance, success rate, retention, recovery time per (joint, noise) pair
Total Episodes: 12,800 (4 models × 8 joints × 4 noise conditions × 100 rollouts)
Runtime: ~18 hours
"""

import sys
import os

# Add paths
sys.path.append('src')
sys.path.append(os.path.dirname(__file__))  # Add evaluations directory

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
import realant_sim
import json
from datetime import datetime
from tqdm import tqdm

# Import utilities
from utils.noise_types import NoiseGenerator
from utils.recovery_time_tracker import RecoveryTimeTracker


class JointNoiseAblationEvaluator:
    """Full factorial evaluation: joints × noise types"""

    def __init__(self):
        # Episode parameters
        self.episode_length = 1200  # 20 seconds at 60fps
        self.num_rollouts = 100

        # Joint failure parameters
        self.delay_steps = 120  # 2-second delay before locking
        self.lock_value = 0.0

        # All 8 joints
        self.joints_to_test = [
            'hip_1', 'ankle_1',
            'hip_2', 'ankle_2',
            'hip_3', 'ankle_3',
            'hip_4', 'ankle_4'
        ]

        # Joint mapping
        self.joint_to_action = {
            'hip_1': 0, 'ankle_1': 1,
            'hip_2': 2, 'ankle_2': 3,
            'hip_3': 4, 'ankle_3': 5,
            'hip_4': 6, 'ankle_4': 7
        }

        # Noise configuration (4 conditions)
        self.noise_generator = NoiseGenerator(obs_dims=(13, 28))
        self.noise_conditions = [
            {'type': 'none', 'level': 0.0, 'description': 'No noise'},
            {'type': 'gaussian', 'level': 0.1, 'description': 'Gaussian σ=0.1'},
            {'type': 'poisson', 'level': 1.0, 'description': 'Poisson λ=1.0'},
            {'type': 'salt_pepper', 'level': 0.0044, 'description': 'Salt-Pepper p=0.44%'}
        ]

        # Success threshold
        self.success_threshold = 1.5

        # Model configurations
        self.models = {
            'M1_baseline': {
                'name': 'PPO Baseline',
                'path': '../done/ppo_baseline_ueqbjf2x/best_model/best_model',
                'vec_path': '../done/ppo_baseline_ueqbjf2x/vec_normalize.pkl',
                'description': 'No robustness training'
            },
            'M2_sr2l': {
                'name': 'PPO + SR2L',
                'path': '../done/ppo_sr2l_forward_m7gtjtpa/final_model',
                'vec_path': '../done/ppo_sr2l_forward_m7gtjtpa/vec_normalize.pkl',
                'description': 'Sensor noise specialist'
            },
            'M3_dr': {
                'name': 'PPO + DR (V7.7E)',
                'path': '../done/v7_7e_ultra_speed_jtfwl2qf/final_model',
                'vec_path': '../done/v7_7e_ultra_speed_jtfwl2qf/vec_normalize.pkl',
                'description': 'Joint failure specialist'
            },
            'M4_combo': {
                'name': 'Ultimate Combo',
                'path': '../done/ultimate_robustness_combo_ju7lfsk2/final_model',
                'vec_path': '../done/ultimate_robustness_combo_ju7lfsk2/vec_normalize.pkl',
                'description': 'SR2L + DR combined'
            }
        }

        self.results = {}

    def create_environment(self):
        """Create clean evaluation environment"""
        def make_env():
            base_env = gym.make('RealAntMujoco-v0', disable_env_checker=True)
            env = SuccessRewardWrapper(base_env)
            env = TimeLimit(env, max_episode_steps=self.episode_length)
            return env

        env = DummyVecEnv([make_env])
        return env

    def load_model(self, model_key):
        """Load model and VecNormalize"""
        config = self.models[model_key]

        env = self.create_environment()
        env = VecNormalize.load(config['vec_path'], env)
        env.training = False
        env.norm_reward = False

        model = PPO.load(config['path'], env=env)
        return model, env

    def apply_joint_failure(self, action, failed_joint, step_count):
        """Apply joint failure after delay"""
        if step_count < self.delay_steps:
            return action

        modified_action = action.copy()
        if failed_joint in self.joint_to_action:
            action_idx = self.joint_to_action[failed_joint]
            modified_action[0][action_idx] = self.lock_value

        return modified_action

    def apply_noise_to_observation(self, obs, noise_type, noise_level):
        """Apply noise to observation"""
        if noise_type == 'none' or noise_level == 0:
            return obs
        elif noise_type == 'gaussian':
            return self.noise_generator.apply_gaussian_noise(obs, noise_level)
        elif noise_type == 'poisson':
            return self.noise_generator.apply_poisson_noise(obs, noise_level)
        elif noise_type == 'salt_pepper':
            return self.noise_generator.apply_salt_pepper_noise(obs, noise_level)
        else:
            return obs

    def evaluate_condition(self, model_key, failed_joint, noise_config):
        """Run rollouts for specific joint + noise combination"""
        model, env = self.load_model(model_key)

        rollout_results = []

        for rollout in range(self.num_rollouts):
            obs = env.reset()
            done = False
            step_count = 0

            positions = []
            rewards = []

            # Initialize recovery tracker
            recovery_tracker = RecoveryTimeTracker(
                fault_injection_step=self.delay_steps,
                recovery_threshold=0.5,
                pre_fault_window=(100, 119),
                fps=60
            )

            # Get initial position
            try:
                unwrapped = env.envs[0].unwrapped
                start_x = unwrapped.data.qpos[0]
            except:
                start_x = 0.0

            while not done and step_count < self.episode_length:
                # Apply noise to observation
                if noise_config['level'] > 0:
                    noisy_obs = self.apply_noise_to_observation(
                        obs[0],
                        noise_config['type'],
                        noise_config['level']
                    )
                    noisy_obs_vec = np.expand_dims(noisy_obs, axis=0)
                else:
                    noisy_obs_vec = obs
                    noisy_obs = obs[0]

                # Track recovery (use noisy obs since that's what policy sees)
                recovery_tracker.track_step(noisy_obs)

                # Get action from model
                action, _ = model.predict(noisy_obs_vec, deterministic=True)

                # Apply joint failure
                modified_action = self.apply_joint_failure(action, failed_joint, step_count)

                # Track position
                try:
                    current_x = unwrapped.data.qpos[0]
                    positions.append(current_x)
                except:
                    positions.append(0.0)

                # Step environment
                obs, reward, done, info = env.step(modified_action)
                rewards.append(reward[0])

                step_count += 1

            # Calculate metrics
            final_position = positions[-1] if positions else start_x
            total_distance = abs(final_position - start_x)
            total_reward = sum(rewards)
            success = total_distance >= self.success_threshold

            # Get recovery metrics
            recovery_results = recovery_tracker.get_results()

            rollout_results.append({
                'rollout_id': rollout,
                'distance': float(total_distance),
                'reward': float(total_reward),
                'success': bool(success),
                'steps': int(step_count),
                'recovered': recovery_results['recovered'],
                'recovery_time_seconds': recovery_results['recovery_time_seconds'],
                'pre_fault_velocity': recovery_results['pre_fault_velocity'],
                'post_fault_avg_velocity': recovery_results['post_fault_avg_velocity']
            })

        # Aggregate statistics
        distances = [r['distance'] for r in rollout_results]
        rewards = [r['reward'] for r in rollout_results]
        success_count = sum(1 for r in rollout_results if r['success'])

        # Recovery statistics
        recovery_times = [r['recovery_time_seconds'] for r in rollout_results
                         if r['recovery_time_seconds'] is not None]
        recovery_count = sum(1 for r in rollout_results if r['recovered'])

        summary = {
            'failed_joint': failed_joint,
            'noise_type': noise_config['type'],
            'noise_level': noise_config['level'],
            'noise_description': noise_config['description'],
            'num_rollouts': self.num_rollouts,
            'distance': {
                'mean': float(np.mean(distances)),
                'std': float(np.std(distances)),
                'median': float(np.median(distances))
            },
            'reward': {
                'mean': float(np.mean(rewards)),
                'std': float(np.std(rewards))
            },
            'success_rate': float(success_count / self.num_rollouts),
            'recovery': {
                'recovery_rate': float(recovery_count / self.num_rollouts),
                'recovery_time_mean': float(np.mean(recovery_times)) if recovery_times else None,
                'recovery_time_std': float(np.std(recovery_times)) if recovery_times else None
            },
            'rollouts': rollout_results
        }

        env.close()
        return summary

    def evaluate_model(self, model_key, baseline_distance):
        """Evaluate model across all joint × noise combinations"""
        config = self.models[model_key]

        print(f"\n{'='*80}")
        print(f"Evaluating: {config['name']}")
        print(f"Baseline: {baseline_distance:.3f}m")
        print(f"{'='*80}")

        ablation_results = []
        total_conditions = len(self.joints_to_test) * len(self.noise_conditions)

        with tqdm(total=total_conditions, desc=f"{config['name']}") as pbar:
            for joint in self.joints_to_test:
                for noise_config in self.noise_conditions:
                    result = self.evaluate_condition(model_key, joint, noise_config)

                    # Calculate retention
                    retention = (result['distance']['mean'] / baseline_distance * 100
                                if baseline_distance > 0 else 0.0)
                    result['retention_percent'] = float(retention)

                    ablation_results.append(result)

                    pbar.set_postfix({
                        'joint': joint,
                        'noise': noise_config['type'][:4],
                        'retention': f"{retention:.1f}%"
                    })
                    pbar.update(1)

        return {
            'model_key': model_key,
            'model_name': config['name'],
            'baseline_distance': float(baseline_distance),
            'ablation_results': ablation_results
        }

    def run_all_evaluations(self):
        """Run complete ablation study"""
        print("="*80)
        print("EXPERIMENT 7: JOINT FAILURE + NOISE TYPE ABLATION")
        print("="*80)
        print(f"Models: {len(self.models)}")
        print(f"Joints: {len(self.joints_to_test)}")
        print(f"Noise conditions: {len(self.noise_conditions)}")
        print(f"Rollouts per condition: {self.num_rollouts}")
        total_episodes = (len(self.models) * len(self.joints_to_test) *
                         len(self.noise_conditions) * self.num_rollouts)
        print(f"Total episodes: {total_episodes:,}")
        print(f"Estimated time: ~18 hours")
        print("="*80)

        # Get baselines
        print("\n📊 Computing baselines...")
        baselines = {}
        for model_key in self.models.keys():
            baseline_result = self.evaluate_condition(
                model_key,
                self.joints_to_test[0],  # Use hip_1 as baseline (easiest joint)
                self.noise_conditions[0]  # No noise
            )
            # Actually, use clean baseline from Experiment 1
            # For now, we'll use a simpler approach
            model, env = self.load_model(model_key)
            obs = env.reset()
            distances_baseline = []
            for _ in range(10):
                done = False
                positions = []
                start_x = 0.0
                try:
                    unwrapped = env.envs[0].unwrapped
                    start_x = unwrapped.data.qpos[0]
                except:
                    pass

                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    try:
                        positions.append(unwrapped.data.qpos[0])
                    except:
                        positions.append(0.0)
                    obs, _, done, _ = env.step(action)

                distances_baseline.append(abs(positions[-1] - start_x) if positions else 0)
                obs = env.reset()

            baselines[model_key] = np.mean(distances_baseline)
            env.close()
            print(f"  {self.models[model_key]['name']}: {baselines[model_key]:.3f}m")

        # Evaluate all models
        for model_key in self.models.keys():
            result = self.evaluate_model(model_key, baselines[model_key])
            self.results[model_key] = result

        # Save results
        self.save_results()

        # Print summary
        self.print_summary()

    def save_results(self):
        """Save results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "evaluations/experiment_7_joint_noise_ablation/data"
        os.makedirs(output_dir, exist_ok=True)

        output_file = f"{output_dir}/joint_noise_ablation_{timestamp}.json"

        # Convert numpy types to native Python types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        results_clean = convert_numpy(self.results)

        with open(output_file, 'w') as f:
            json.dump(results_clean, f, indent=2)

        print(f"\n✅ Results saved to: {output_file}")

    def print_summary(self):
        """Print interaction effect summary"""
        print("\n" + "="*80)
        print("EXPERIMENT 7 SUMMARY: JOINT × NOISE INTERACTION EFFECTS")
        print("="*80)

        for model_key, result in self.results.items():
            print(f"\n{result['model_name']}:")

            # Find best/worst combinations
            all_results = result['ablation_results']
            best = max(all_results, key=lambda x: x['retention_percent'])
            worst = min(all_results, key=lambda x: x['retention_percent'])

            print(f"  Best condition: {best['failed_joint']} + {best['noise_description']}")
            print(f"    Retention: {best['retention_percent']:.1f}%")
            print(f"  Worst condition: {worst['failed_joint']} + {worst['noise_description']}")
            print(f"    Retention: {worst['retention_percent']:.1f}%")

            # Interaction effect: Does noise help or hurt recovery from joint failures?
            for joint in self.joints_to_test:
                joint_results = [r for r in all_results if r['failed_joint'] == joint]
                no_noise = next(r for r in joint_results if r['noise_type'] == 'none')
                with_noise = [r for r in joint_results if r['noise_type'] != 'none']

                if with_noise:
                    avg_noise_retention = np.mean([r['retention_percent'] for r in with_noise])
                    interaction = avg_noise_retention - no_noise['retention_percent']

                    if abs(interaction) > 5:  # Meaningful interaction
                        effect = "helps" if interaction > 0 else "hurts"
                        print(f"    {joint}: Noise {effect} recovery ({interaction:+.1f}%)")

        print("\n" + "="*80)


def main():
    evaluator = JointNoiseAblationEvaluator()
    evaluator.run_all_evaluations()


if __name__ == "__main__":
    main()
