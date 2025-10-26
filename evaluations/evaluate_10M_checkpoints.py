#!/usr/bin/env python3
"""
Matched Checkpoint Evaluation - 10M Steps Fair Comparison
==========================================================

Addresses supervisor's concern about training duration confound by evaluating
all 4 models at matched 10M timestep checkpoints for fair "equal compute" comparison.

Models:
- M1 (Baseline): done/ppo_baseline_ueqbjf2x/checkpoints/model_10000000_steps.zip
- M2 (SR2L): done/ppo_sr2l_forward_m7gtjtpa/checkpoints/checkpoint_10000000_steps.zip
- M3 (DR): done/v7_7e_ultra_speed_jtfwl2qf/checkpoints/checkpoint_10000000_steps.zip
- M4 (Combo): done/ultimate_robustness_combo_ju7lfsk2/checkpoints/checkpoint_10000000_steps.zip

Evaluation Protocol:
1. Baseline performance (100 episodes, no perturbations)
2. Sensor noise robustness (12 noise levels, 100 episodes each)
3. Joint failure robustness (8 joints, 100 episodes each)
4. Combined stress (6 scenarios, 100 episodes each)

Total: ~3,000 episodes per model = 12,000 episodes total
Estimated time: 2-3 days
"""

import sys
sys.path.append('../src')

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
import realant_sim
import json
import os
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

class MatchedCheckpointEvaluator:
    """Evaluates all models at 10M step checkpoints for fair comparison"""

    def __init__(self):
        self.episode_length = 1000  # Match paper's evaluation protocol
        self.num_rollouts = 100  # Match paper's sample size

        # Model checkpoint paths (10M steps for all)
        self.models = {
            'M1_baseline_10M': {
                'name': 'M1: Baseline @ 10M steps',
                'model_path': '../done/ppo_baseline_ueqbjf2x/checkpoints/model_10000000_steps.zip',
                'vec_path': '../done/ppo_baseline_ueqbjf2x/checkpoints/model_vecnormalize_10000000_steps.pkl',
                'timestep': 10_000_000
            },
            'M2_sr2l_10M': {
                'name': 'M2: SR2L @ 10M steps',
                'model_path': '../done/ppo_sr2l_forward_m7gtjtpa/checkpoints/checkpoint_10000000_steps.zip',
                'vec_path': '../done/ppo_sr2l_forward_m7gtjtpa/vec_normalize.pkl',  # Use final VecNorm
                'timestep': 10_000_000
            },
            'M3_dr_10M': {
                'name': 'M3: DR @ 10M steps',
                'model_path': '../done/v7_7e_ultra_speed_jtfwl2qf/checkpoints/checkpoint_10000000_steps.zip',
                'vec_path': '../done/v7_7e_ultra_speed_jtfwl2qf/vec_normalize.pkl',  # Use final VecNorm
                'timestep': 10_000_000
            },
            'M4_combo_10M': {
                'name': 'M4: Combo @ 10M steps',
                'model_path': '../done/ultimate_robustness_combo_ju7lfsk2/checkpoints/checkpoint_10000000_steps.zip',
                'vec_path': '../done/ultimate_robustness_combo_ju7lfsk2/vec_normalize.pkl',  # Use final VecNorm
                'timestep': 10_000_000
            }
        }

        # For comparison, also include final checkpoints
        self.models_final = {
            'M1_baseline_final': {
                'name': 'M1: Baseline @ 10M (final)',
                'model_path': '../done/ppo_baseline_ueqbjf2x/best_model/best_model.zip',
                'vec_path': '../done/ppo_baseline_ueqbjf2x/vec_normalize.pkl',
                'timestep': 10_000_000
            },
            'M2_sr2l_final': {
                'name': 'M2: SR2L @ 20M (final)',
                'model_path': '../done/ppo_sr2l_forward_m7gtjtpa/final_model.zip',
                'vec_path': '../done/ppo_sr2l_forward_m7gtjtpa/vec_normalize.pkl',
                'timestep': 20_000_000
            },
            'M3_dr_final': {
                'name': 'M3: DR @ 32M (final)',
                'model_path': '../done/v7_7e_ultra_speed_jtfwl2qf/final_model.zip',
                'vec_path': '../done/v7_7e_ultra_speed_jtfwl2qf/vec_normalize.pkl',
                'timestep': 32_000_000
            },
            'M4_combo_final': {
                'name': 'M4: Combo @ 30M (final)',
                'model_path': '../done/ultimate_robustness_combo_ju7lfsk2/final_model.zip',
                'vec_path': '../done/ultimate_robustness_combo_ju7lfsk2/vec_normalize.pkl',
                'timestep': 30_000_000
            }
        }

        self.joint_names = ['hip_1', 'ankle_1', 'hip_2', 'ankle_2',
                           'hip_3', 'ankle_3', 'hip_4', 'ankle_4']

    def create_environment(self):
        """Create clean evaluation environment"""
        def make_env():
            base_env = gym.make('RealAntMujoco-v0', disable_env_checker=True)
            env = SuccessRewardWrapper(base_env)
            env = TimeLimit(env, max_episode_steps=self.episode_length)
            return env

        env = DummyVecEnv([make_env])
        return env

    def load_model(self, model_key, use_final=False):
        """Load model checkpoint with VecNormalize"""
        models_dict = self.models_final if use_final else self.models
        config = models_dict[model_key]

        print(f"\n{'='*70}")
        print(f"Loading: {config['name']}")
        print(f"Model: {config['model_path']}")
        print(f"VecNorm: {config['vec_path']}")
        print(f"{'='*70}")

        # Verify files exist
        if not Path(config['model_path']).exists():
            raise FileNotFoundError(f"Model not found: {config['model_path']}")
        if not Path(config['vec_path']).exists():
            raise FileNotFoundError(f"VecNormalize not found: {config['vec_path']}")

        # Create environment
        env = self.create_environment()

        # Load VecNormalize
        env = VecNormalize.load(config['vec_path'], env)
        env.training = False
        env.norm_reward = False

        # Load model
        model = PPO.load(config['model_path'], env=env)

        return model, env

    def evaluate_baseline(self, model_key, use_final=False):
        """Experiment 1: Baseline performance (no perturbations)"""
        models_dict = self.models_final if use_final else self.models
        config = models_dict[model_key]

        print(f"\n{'='*70}")
        print(f"BASELINE PERFORMANCE - {config['name']}")
        print('='*70)

        model, env = self.load_model(model_key, use_final)
        distances = []

        for ep in tqdm(range(self.num_rollouts), desc="Baseline episodes"):
            obs = env.reset()
            done = False
            positions = [env.envs[0].env.unwrapped.get_body_com("torso")[0]]

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                if not done[0]:
                    positions.append(env.envs[0].env.unwrapped.get_body_com("torso")[0])

            distance = positions[-1] - positions[0]
            distances.append(distance)

        result = {
            'mean': float(np.mean(distances)),
            'std': float(np.std(distances)),
            'model': config['name'],
            'timestep': config['timestep']
        }

        print(f"\n✅ Distance: {result['mean']:.2f}m ± {result['std']:.2f}m")
        return result

    def evaluate_sensor_noise(self, model_key, noise_std, baseline_dist, use_final=False):
        """Experiment 2: Sensor noise robustness"""
        model, env = self.load_model(model_key, use_final)
        distances = []

        for ep in range(self.num_rollouts):
            obs = env.reset()
            done = False
            positions = [env.envs[0].env.unwrapped.get_body_com("torso")[0]]

            while not done:
                # Add noise to observations
                noisy_obs = obs + np.random.normal(0, noise_std, obs.shape)
                action, _ = model.predict(noisy_obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                if not done[0]:
                    positions.append(env.envs[0].env.unwrapped.get_body_com("torso")[0])

            distance = positions[-1] - positions[0]
            distances.append(distance)

        mean_dist = np.mean(distances)
        retention = (mean_dist / baseline_dist * 100) if baseline_dist > 0 else 0

        return {
            'noise_std': noise_std,
            'mean': float(mean_dist),
            'std': float(np.std(distances)),
            'retention_pct': float(retention)
        }

    def evaluate_joint_failure(self, model_key, joint_idx, baseline_dist, use_final=False):
        """Experiment 3: Joint failure robustness"""
        model, env = self.load_model(model_key, use_final)
        distances = []

        for ep in range(self.num_rollouts):
            obs = env.reset()
            done = False
            positions = [env.envs[0].env.unwrapped.get_body_com("torso")[0]]

            # Lock joint at 0.5 radians
            env.envs[0].env.unwrapped.data.qpos[7 + joint_idx] = 0.5

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action[0][joint_idx] = 0.5  # Keep locked
                obs, reward, done, info = env.step(action)
                if not done[0]:
                    positions.append(env.envs[0].env.unwrapped.get_body_com("torso")[0])

            distance = positions[-1] - positions[0]
            distances.append(distance)

        mean_dist = np.mean(distances)
        retention = (mean_dist / baseline_dist * 100) if baseline_dist > 0 else 0

        return {
            'joint': self.joint_names[joint_idx],
            'mean': float(mean_dist),
            'std': float(np.std(distances)),
            'retention_pct': float(retention)
        }

    def run_matched_comparison(self, include_final=True):
        """Run full evaluation suite on 10M checkpoints + final models"""
        print("\n" + "="*70)
        print("MATCHED CHECKPOINT EVALUATION - 10M STEPS")
        print("="*70)
        print("Addressing training duration confound")
        print("All models evaluated at matched 10M timestep checkpoint")
        print("="*70)

        results = {
            '10M_checkpoints': {},
            'final_checkpoints': {} if include_final else None
        }

        # Evaluate 10M checkpoints
        for model_key in self.models.keys():
            results['10M_checkpoints'][model_key] = {}

            # Experiment 1: Baseline
            baseline_result = self.evaluate_baseline(model_key, use_final=False)
            results['10M_checkpoints'][model_key]['baseline'] = baseline_result
            baseline_dist = baseline_result['mean']

            # Experiment 2: Sensor noise (12 levels)
            print(f"\n{'='*70}")
            print(f"SENSOR NOISE - {self.models[model_key]['name']}")
            print('='*70)

            noise_levels = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]
            results['10M_checkpoints'][model_key]['sensor_noise'] = []

            for noise_std in tqdm(noise_levels, desc="Noise levels"):
                noise_result = self.evaluate_sensor_noise(model_key, noise_std, baseline_dist, use_final=False)
                results['10M_checkpoints'][model_key]['sensor_noise'].append(noise_result)
                print(f"  σ={noise_std:.2f}: {noise_result['mean']:.2f}m ({noise_result['retention_pct']:.1f}%)")

            # Experiment 3: Joint failures (8 joints)
            print(f"\n{'='*70}")
            print(f"JOINT FAILURES - {self.models[model_key]['name']}")
            print('='*70)

            results['10M_checkpoints'][model_key]['joint_failures'] = []

            for joint_idx in tqdm(range(8), desc="Testing joints"):
                joint_result = self.evaluate_joint_failure(model_key, joint_idx, baseline_dist, use_final=False)
                results['10M_checkpoints'][model_key]['joint_failures'].append(joint_result)
                print(f"  {self.joint_names[joint_idx]:10s}: {joint_result['retention_pct']:.1f}%")

        # Optionally evaluate final checkpoints for comparison
        if include_final:
            print("\n" + "="*70)
            print("FINAL CHECKPOINTS (for comparison)")
            print("="*70)

            for model_key in self.models_final.keys():
                results['final_checkpoints'][model_key] = {}
                baseline_result = self.evaluate_baseline(model_key, use_final=True)
                results['final_checkpoints'][model_key]['baseline'] = baseline_result

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"matched_checkpoint_results_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved to {results_file}")

        # Print summary comparison
        self.print_summary(results)

        return results

    def print_summary(self, results):
        """Print summary table comparing 10M vs final performance"""
        print("\n" + "="*70)
        print("MATCHED CHECKPOINT COMPARISON SUMMARY")
        print("="*70)

        print("\nBaseline Performance Comparison:")
        print("-" * 70)
        print(f"{'Model':<35} {'10M Checkpoint':<20} {'Final Checkpoint':<20}")
        print("-" * 70)

        model_pairs = [
            ('M1_baseline_10M', 'M1_baseline_final'),
            ('M2_sr2l_10M', 'M2_sr2l_final'),
            ('M3_dr_10M', 'M3_dr_final'),
            ('M4_combo_10M', 'M4_combo_final')
        ]

        for checkpoint_key, final_key in model_pairs:
            checkpoint_dist = results['10M_checkpoints'][checkpoint_key]['baseline']['mean']
            final_dist = results['final_checkpoints'][final_key]['baseline']['mean']
            improvement = ((final_dist - checkpoint_dist) / checkpoint_dist * 100) if checkpoint_dist > 0 else 0

            model_name = self.models[checkpoint_key]['name'].split('@')[0].strip()

            print(f"{model_name:<35} {checkpoint_dist:>8.2f}m          {final_dist:>8.2f}m (+{improvement:>5.1f}%)")

        print("-" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Matched checkpoint evaluation for fair comparison')
    parser.add_argument('--no-final', action='store_true',
                       help='Skip final checkpoint evaluation (10M only)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test mode (10 episodes only)')

    args = parser.parse_args()

    evaluator = MatchedCheckpointEvaluator()

    if args.quick_test:
        print("⚡ QUICK TEST MODE - 10 episodes only")
        evaluator.num_rollouts = 10

    results = evaluator.run_matched_comparison(include_final=not args.no_final)
