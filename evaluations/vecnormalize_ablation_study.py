#!/usr/bin/env python3
"""
VecNormalize Ablation Study
============================

Compares models WITH vs WITHOUT VecNormalize to isolate its contribution to robustness.

Tests 3 model pairs:
- M2 (SR2L) vs M2_no_vecnorm
- M3 (DR) vs M3_no_vecnorm
- M4 (Combo) vs M4_no_vecnorm

Evaluation protocol:
1. Baseline performance (no perturbations)
2. Sensor noise robustness (σ = 0.01, 0.05, 0.1)
3. Joint failure robustness (all 8 joints)

Output:
- JSON results for each model pair
- Comparison figures showing VecNormalize's contribution
- Statistical analysis (paired t-tests)
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym

# Import custom environment and wrappers
from src.envs.success_reward_wrapper import SuccessRewardWrapper
from src.realant_sim import AntEnv


class VecNormalizeAblationEvaluator:
    """Evaluator for VecNormalize ablation study"""

    def __init__(self, results_dir: str = "evaluations/vecnormalize_ablation_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Model pairs to compare
        self.model_pairs = {
            'M2_SR2L': {
                'with_vecnorm': 'done/ppo_sr2l_forward_m7gtjtpa',
                'without_vecnorm': 'experiments/ppo_sr2l_no_vecnorm_tdxq60kz',
                'name': 'SR2L'
            },
            'M3_DR': {
                'with_vecnorm': 'done/v7_7e_ultra_speed_jtfwl2qf',
                'without_vecnorm': 'experiments/ppo_dr_no_vecnorm_b5zqadfj',
                'name': 'Domain Randomization'
            },
            'M4_Combo': {
                'with_vecnorm': 'done/ultimate_robustness_combo_ju7lfsk2',
                'without_vecnorm': 'experiments/ppo_combo_no_vecnorm_qubss2iq',
                'name': 'SR2L + DR Combo'
            }
        }

        self.joint_names = ['hip_1', 'ankle_1', 'hip_2', 'ankle_2',
                           'hip_3', 'ankle_3', 'hip_4', 'ankle_4']

    def find_ablation_models(self):
        """Find the no_vecnorm model directories"""
        experiments_dir = Path('experiments')

        # Find SR2L ablation
        sr2l_dirs = list(experiments_dir.glob('ppo_sr2l_no_vecnorm_*'))
        if sr2l_dirs:
            self.model_pairs['M2_SR2L']['without_vecnorm'] = str(sr2l_dirs[0])

        # Find DR ablation
        dr_dirs = list(experiments_dir.glob('ppo_dr_no_vecnorm_*'))
        if dr_dirs:
            self.model_pairs['M3_DR']['without_vecnorm'] = str(dr_dirs[0])

        # Find Combo ablation
        combo_dirs = list(experiments_dir.glob('ppo_combo_no_vecnorm_*'))
        if combo_dirs:
            self.model_pairs['M4_Combo']['without_vecnorm'] = str(combo_dirs[0])

        # Print found models
        print("=" * 80)
        print("VECNORMALIZE ABLATION STUDY - MODEL PAIRS")
        print("=" * 80)
        for key, pair in self.model_pairs.items():
            print(f"\n{pair['name']}:")
            print(f"  WITH VecNormalize:    {pair['with_vecnorm']}")
            if pair['without_vecnorm']:
                print(f"  WITHOUT VecNormalize: {pair['without_vecnorm']} ")
            else:
                print(f"  WITHOUT VecNormalize: NOT FOUND ❌")
        print()

    def load_model(self, model_path: str, use_vecnormalize: bool = True):
        """Load a model with or without VecNormalize"""
        model_dir = Path(model_path)

        # Find model file
        if (model_dir / 'final_model.zip').exists():
            model_file = model_dir / 'final_model.zip'
        elif (model_dir / 'best_model' / 'best_model.zip').exists():
            model_file = model_dir / 'best_model' / 'best_model.zip'
        else:
            raise FileNotFoundError(f"No model found in {model_dir}")

        # Create environment
        def make_env():
            env = gym.make('RealAntMujoco-v0')
            env = SuccessRewardWrapper(env)
            return env

        env = DummyVecEnv([make_env])

        # Load VecNormalize if requested AND available
        if use_vecnormalize:
            vec_norm_path = model_dir / 'vec_normalize.pkl'
            if vec_norm_path.exists():
                env = VecNormalize.load(str(vec_norm_path), env)
                env.training = False
                env.norm_reward = False
            else:
                print(f"  ⚠️  VecNormalize file not found at {vec_norm_path}")

        # Load model
        model = PPO.load(str(model_file), env=env)

        return model, env

    def evaluate_baseline(self, model, env, n_episodes: int = 100) -> Dict:
        """Evaluate baseline performance without any perturbations"""
        distances = []
        velocities = []
        rewards = []
        successes = []

        for ep in range(n_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            positions = [env.envs[0].env.env.get_body_com("torso")[0]]

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0]

                if not done[0]:
                    positions.append(env.envs[0].env.env.get_body_com("torso")[0])

            # Calculate metrics
            distance = positions[-1] - positions[0]
            velocity = distance / (len(positions) * 0.02)  # 0.02s timestep

            distances.append(distance)
            velocities.append(velocity)
            rewards.append(episode_reward)
            successes.append(1 if distance > 2.0 else 0)

        return {
            'distance_mean': np.mean(distances),
            'distance_std': np.std(distances),
            'velocity_mean': np.mean(velocities),
            'velocity_std': np.std(velocities),
            'reward_mean': np.mean(rewards),
            'success_rate': np.mean(successes) * 100,
            'n_episodes': n_episodes
        }

    def evaluate_sensor_noise(self, model, env, noise_std: float,
                             baseline_dist: float, n_episodes: int = 100) -> Dict:
        """Evaluate robustness to sensor noise"""
        distances = []

        for ep in range(n_episodes):
            obs = env.reset()
            done = False
            positions = [env.envs[0].env.env.get_body_com("torso")[0]]

            while not done:
                # Add Gaussian noise to observations
                noisy_obs = obs + np.random.normal(0, noise_std, obs.shape)
                action, _ = model.predict(noisy_obs, deterministic=True)
                obs, reward, done, info = env.step(action)

                if not done[0]:
                    positions.append(env.envs[0].env.env.get_body_com("torso")[0])

            distance = positions[-1] - positions[0]
            distances.append(distance)

        mean_dist = np.mean(distances)
        retention = (mean_dist / baseline_dist * 100) if baseline_dist > 0 else 0

        return {
            'noise_std': noise_std,
            'distance_mean': mean_dist,
            'distance_std': np.std(distances),
            'retention_percent': retention,
            'n_episodes': n_episodes
        }

    def evaluate_joint_failure(self, model, env, joint_idx: int,
                              baseline_dist: float, n_episodes: int = 100) -> Dict:
        """Evaluate robustness to single joint failure"""
        distances = []

        for ep in range(n_episodes):
            obs = env.reset()
            done = False
            positions = [env.envs[0].env.env.get_body_com("torso")[0]]

            # Lock the joint at 0.5 radians
            env.envs[0].env.env.data.qpos[7 + joint_idx] = 0.5

            while not done:
                action, _ = model.predict(obs, deterministic=True)

                # Keep joint locked
                action[0][joint_idx] = 0.5

                obs, reward, done, info = env.step(action)

                if not done[0]:
                    positions.append(env.envs[0].env.env.get_body_com("torso")[0])

            distance = positions[-1] - positions[0]
            distances.append(distance)

        mean_dist = np.mean(distances)
        retention = (mean_dist / baseline_dist * 100) if baseline_dist > 0 else 0

        return {
            'joint_name': self.joint_names[joint_idx],
            'joint_idx': joint_idx,
            'distance_mean': mean_dist,
            'distance_std': np.std(distances),
            'retention_percent': retention,
            'n_episodes': n_episodes
        }

    def run_full_evaluation(self, model_key: str, use_vecnormalize: bool) -> Dict:
        """Run complete evaluation protocol for one model"""
        pair = self.model_pairs[model_key]
        model_path = pair['with_vecnorm'] if use_vecnormalize else pair['without_vecnorm']

        if not model_path:
            print(f"❌ Model path not found for {model_key} (vecnorm={use_vecnormalize})")
            return None

        print(f"\n{'='*80}")
        print(f"Evaluating: {pair['name']}")
        print(f"VecNormalize: {' ENABLED' if use_vecnormalize else '❌ DISABLED'}")
        print(f"Model: {model_path}")
        print('='*80)

        # Load model
        model, env = self.load_model(model_path, use_vecnormalize)

        results = {
            'model_name': pair['name'],
            'model_path': model_path,
            'use_vecnormalize': use_vecnormalize
        }

        # 1. Baseline evaluation
        print("\n1️⃣  Baseline Performance (no perturbations)...")
        baseline = self.evaluate_baseline(model, env, n_episodes=100)
        results['baseline'] = baseline
        print(f"   Distance: {baseline['distance_mean']:.2f}m ± {baseline['distance_std']:.2f}m")
        print(f"   Success Rate: {baseline['success_rate']:.1f}%")

        # 2. Sensor noise evaluation
        print("\n2️⃣  Sensor Noise Robustness...")
        noise_levels = [0.01, 0.05, 0.1]
        results['sensor_noise'] = []

        for noise_std in noise_levels:
            print(f"   Testing σ = {noise_std}...")
            noise_result = self.evaluate_sensor_noise(
                model, env, noise_std, baseline['distance_mean'], n_episodes=100
            )
            results['sensor_noise'].append(noise_result)
            print(f"     Retention: {noise_result['retention_percent']:.1f}%")

        # 3. Joint failure evaluation
        print("\n3️⃣  Joint Failure Robustness...")
        results['joint_failures'] = []

        for joint_idx in range(8):
            print(f"   Testing {self.joint_names[joint_idx]}...")
            joint_result = self.evaluate_joint_failure(
                model, env, joint_idx, baseline['distance_mean'], n_episodes=100
            )
            results['joint_failures'].append(joint_result)
            print(f"     Retention: {joint_result['retention_percent']:.1f}%")

        return results

    def run_all_evaluations(self):
        """Run evaluations for all model pairs"""
        self.find_ablation_models()

        all_results = {}

        for model_key in self.model_pairs.keys():
            # Evaluate WITH VecNormalize
            results_with = self.run_full_evaluation(model_key, use_vecnormalize=True)
            if results_with:
                all_results[f"{model_key}_with_vecnorm"] = results_with

            # Evaluate WITHOUT VecNormalize
            results_without = self.run_full_evaluation(model_key, use_vecnormalize=False)
            if results_without:
                all_results[f"{model_key}_without_vecnorm"] = results_without

        # Save results
        results_file = self.results_dir / 'ablation_results.json'
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n Results saved to {results_file}")

        return all_results

    def generate_comparison_figures(self, results: Dict):
        """Generate figures comparing with/without VecNormalize"""
        # TODO: Create comparison visualizations
        # 1. Baseline performance comparison (bar chart)
        # 2. Sensor noise retention curves (line plots)
        # 3. Joint failure retention heatmaps (side-by-side)
        # 4. Statistical significance markers
        pass


if __name__ == "__main__":
    evaluator = VecNormalizeAblationEvaluator()
    results = evaluator.run_all_evaluations()
    evaluator.generate_comparison_figures(results)
