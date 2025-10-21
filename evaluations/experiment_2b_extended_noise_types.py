#!/usr/bin/env python3
"""
EXPERIMENT 2B: EXTENDED SENSOR NOISE ROBUSTNESS EVALUATION
Tests all 4 models with 3 noise types (Gaussian, Poisson, Salt-and-Pepper)
SNR-matched noise levels for fair comparison across distributions

NEW EXPERIMENT (Oct 19, 2025):
This extends Experiment 2 to test generalization across noise types, not just magnitude.
Addresses research question: Does SR2L robustness transfer across noise distributions?

Metrics: Distance, success rate, retention percentage per noise type
Total Episodes: 7,200 (4 models × 3 noise types × 6 levels × 100 rollouts)
Runtime: ~12 hours
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

# Import noise utilities
from utils.noise_types import NoiseGenerator


class ExtendedNoiseEvaluator:
    """Evaluates all 4 models under multiple noise type distributions"""

    def __init__(self):
        # Episode parameters
        self.episode_length = 1200  # 20 seconds at 60fps
        self.num_rollouts = 100

        # Noise configuration
        self.noise_generator = NoiseGenerator(obs_dims=(13, 28))  # Joint sensors

        # Test 3 noise types
        self.noise_types = ['gaussian', 'poisson', 'salt_pepper']

        # Test 6 noise levels (SNR-matched across types)
        # Reduced from 12 levels to keep runtime reasonable
        self.gaussian_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3]

        # Success threshold
        self.success_threshold = 1.5  # meters

        # Model configurations
        self.models = {
            'M1_baseline': {
                'name': 'PPO Baseline',
                'path': '../done/ppo_baseline_ueqbjf2x/best_model/best_model',
                'vec_path': '../done/ppo_baseline_ueqbjf2x/vec_normalize.pkl',
                'description': 'PPO only (no noise training)'
            },
            'M2_sr2l': {
                'name': 'PPO + SR2L',
                'path': '../done/ppo_sr2l_forward_m7gtjtpa/final_model',
                'vec_path': '../done/ppo_sr2l_forward_m7gtjtpa/vec_normalize.pkl',
                'description': 'PPO + SR2L (trained with Gaussian σ=0.01)'
            },
            'M3_dr': {
                'name': 'PPO + DR (V7.7E)',
                'path': '../done/v7_7e_ultra_speed_jtfwl2qf/final_model',
                'vec_path': '../done/v7_7e_ultra_speed_jtfwl2qf/vec_normalize.pkl',
                'description': 'PPO + DR (joint failure specialist)'
            },
            'M4_combo': {
                'name': 'Ultimate Combo',
                'path': '../done/ultimate_robustness_combo_ju7lfsk2/final_model',
                'vec_path': '../done/ultimate_robustness_combo_ju7lfsk2/vec_normalize.pkl',
                'description': 'PPO + SR2L + DR'
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

        # Create environment
        env = self.create_environment()

        # Load VecNormalize
        env = VecNormalize.load(config['vec_path'], env)
        env.training = False
        env.norm_reward = False

        # Load model
        model = PPO.load(config['path'], env=env)

        return model, env

    def apply_noise_to_observation(self, obs, noise_type, noise_level):
        """Apply specified noise type to observation"""
        if noise_type == 'gaussian':
            return self.noise_generator.apply_gaussian_noise(obs, noise_level)
        elif noise_type == 'poisson':
            equiv_params = self.noise_generator.get_equivalent_params(noise_level)
            return self.noise_generator.apply_poisson_noise(obs, equiv_params['poisson_lambda'])
        elif noise_type == 'salt_pepper':
            equiv_params = self.noise_generator.get_equivalent_params(noise_level)
            return self.noise_generator.apply_salt_pepper_noise(obs, equiv_params['salt_pepper_prob'])
        else:
            return obs

    def evaluate_model_with_noise(self, model_key, noise_type, noise_level):
        """Run rollouts with specific noise type and level"""
        model, env = self.load_model(model_key)

        rollout_results = []

        for rollout in range(self.num_rollouts):
            obs = env.reset()
            done = False
            step_count = 0

            # Track metrics
            positions = []
            rewards = []

            # Get initial position
            try:
                unwrapped = env.envs[0].unwrapped
                start_x = unwrapped.data.qpos[0]
            except:
                start_x = 0.0

            while not done and step_count < self.episode_length:
                # Apply noise to observation
                if noise_level > 0:
                    noisy_obs = self.apply_noise_to_observation(obs[0], noise_type, noise_level)
                    noisy_obs_vec = np.expand_dims(noisy_obs, axis=0)
                else:
                    noisy_obs_vec = obs

                # Get action from model (using noisy observation)
                action, _ = model.predict(noisy_obs_vec, deterministic=True)

                # Track position BEFORE step
                try:
                    current_x = unwrapped.data.qpos[0]
                    positions.append(current_x)
                except:
                    positions.append(0.0)

                # Step environment with CLEAN action (noise only affects perception)
                obs, reward, done, info = env.step(action)
                rewards.append(reward[0])

                step_count += 1

            # Calculate metrics
            final_position = positions[-1] if positions else start_x
            total_distance = abs(final_position - start_x)
            total_reward = sum(rewards)
            success = total_distance >= self.success_threshold

            rollout_results.append({
                'rollout_id': rollout,
                'distance': float(total_distance),
                'reward': float(total_reward),
                'success': bool(success),
                'steps': int(step_count)
            })

        # Calculate aggregate statistics
        distances = [r['distance'] for r in rollout_results]
        rewards = [r['reward'] for r in rollout_results]
        success_count = sum(1 for r in rollout_results if r['success'])

        summary = {
            'noise_type': noise_type,
            'noise_level': float(noise_level),
            'num_rollouts': self.num_rollouts,
            'distance': {
                'mean': float(np.mean(distances)),
                'std': float(np.std(distances)),
                'min': float(np.min(distances)),
                'max': float(np.max(distances)),
                'median': float(np.median(distances))
            },
            'reward': {
                'mean': float(np.mean(rewards)),
                'std': float(np.std(rewards))
            },
            'success_rate': float(success_count / self.num_rollouts),
            'rollouts': rollout_results
        }

        env.close()
        return summary

    def evaluate_model(self, model_key, baseline_distance):
        """Evaluate model across all noise types and levels"""
        config = self.models[model_key]

        print(f"\n{'='*80}")
        print(f"Evaluating: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"Baseline distance: {baseline_distance:.3f}m")
        print(f"{'='*80}")

        noise_results = []
        total_tests = len(self.noise_types) * len(self.gaussian_levels)

        with tqdm(total=total_tests, desc=f"{config['name']}") as pbar:
            for noise_type in self.noise_types:
                for noise_level in self.gaussian_levels:
                    result = self.evaluate_model_with_noise(model_key, noise_type, noise_level)

                    # Calculate retention percentage
                    retention = (result['distance']['mean'] / baseline_distance * 100
                                if baseline_distance > 0 else 0.0)
                    result['retention_percent'] = float(retention)

                    noise_results.append(result)

                    # Update progress bar with current result
                    pbar.set_postfix({
                        'type': noise_type[:4],
                        'level': f"{noise_level:.2f}",
                        'retention': f"{retention:.1f}%"
                    })
                    pbar.update(1)

        return {
            'model_key': model_key,
            'model_name': config['name'],
            'baseline_distance': float(baseline_distance),
            'noise_results': noise_results
        }

    def run_all_evaluations(self):
        """Run complete evaluation suite"""
        print("="*80)
        print("EXPERIMENT 2B: EXTENDED SENSOR NOISE ROBUSTNESS EVALUATION")
        print("="*80)
        print(f"Models: {len(self.models)}")
        print(f"Noise types: {len(self.noise_types)} (Gaussian, Poisson, Salt-and-Pepper)")
        print(f"Noise levels: {len(self.gaussian_levels)} (SNR-matched)")
        print(f"Rollouts per condition: {self.num_rollouts}")
        print(f"Total episodes: {len(self.models) * len(self.noise_types) * len(self.gaussian_levels) * self.num_rollouts}")
        print(f"Estimated time: ~12 hours")
        print("="*80)

        # First, get baseline performance (no noise)
        print("\n📊 Computing baseline performance (σ=0.0)...")
        baselines = {}
        for model_key in self.models.keys():
            baseline_result = self.evaluate_model_with_noise(model_key, 'gaussian', 0.0)
            baselines[model_key] = baseline_result['distance']['mean']
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
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "evaluations/experiment_2b_extended_noise/data"
        os.makedirs(output_dir, exist_ok=True)

        output_file = f"{output_dir}/extended_noise_results_{timestamp}.json"

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
        """Print comprehensive summary"""
        print("\n" + "="*80)
        print("EXPERIMENT 2B SUMMARY: NOISE TYPE ROBUSTNESS")
        print("="*80)

        for model_key, result in self.results.items():
            print(f"\n{result['model_name']}:")
            print(f"  Baseline: {result['baseline_distance']:.3f}m")

            # Summary by noise type
            for noise_type in self.noise_types:
                type_results = [r for r in result['noise_results'] if r['noise_type'] == noise_type]

                # Get worst-case retention (highest noise level)
                worst_case = min(type_results, key=lambda x: x['retention_percent'])

                print(f"\n  {noise_type.upper()}:")
                print(f"    Best retention: {max(r['retention_percent'] for r in type_results):.1f}%")
                print(f"    Worst retention: {worst_case['retention_percent']:.1f}% "
                      f"(σ={worst_case['noise_level']:.2f})")

        print("\n" + "="*80)


def main():
    evaluator = ExtendedNoiseEvaluator()
    evaluator.run_all_evaluations()


if __name__ == "__main__":
    main()
