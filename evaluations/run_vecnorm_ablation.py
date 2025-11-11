#!/usr/bin/env python3
"""
VecNormalize Ablation: Quick Evaluation
========================================

Adds the 3 no_vecnorm models to the evaluation suite and runs:
- Experiment 1: Baseline performance
- Experiment 2: Sensor noise robustness
- Experiment 3: Joint failure robustness

This gives us the core comparison data we need.

Total: ~2,400 episodes (baseline + noise + joint failures) × 7 models = 16,800 episodes (~12 hours)
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
import json
import os
from datetime import datetime
from tqdm import tqdm

class VecNormAblationEvaluator:
    """Evaluates all 7 models (4 original + 3 ablation)"""

    def __init__(self):
        self.episode_length = 1200
        self.num_rollouts = 100

        # ALL 7 MODELS: 4 original + 3 ablation
        self.models = {
            # Original models (WITH VecNormalize)
            'M1_baseline': {
                'name': 'M1: Baseline',
                'path': '../done/ppo_baseline_ueqbjf2x/best_model/best_model',
                'vec_path': '../done/ppo_baseline_ueqbjf2x/vec_normalize.pkl',
                'has_vecnorm': True
            },
            'M2_sr2l': {
                'name': 'M2: SR2L',
                'path': '../done/ppo_sr2l_forward_m7gtjtpa/final_model',
                'vec_path': '../done/ppo_sr2l_forward_m7gtjtpa/vec_normalize.pkl',
                'has_vecnorm': True
            },
            'M3_dr': {
                'name': 'M3: DR',
                'path': '../done/v7_7e_ultra_speed_jtfwl2qf/final_model',
                'vec_path': '../done/v7_7e_ultra_speed_jtfwl2qf/vec_normalize.pkl',
                'has_vecnorm': True
            },
            'M4_combo': {
                'name': 'M4: Combo',
                'path': '../done/ultimate_robustness_combo_ju7lfsk2/final_model',
                'vec_path': '../done/ultimate_robustness_combo_ju7lfsk2/vec_normalize.pkl',
                'has_vecnorm': True
            },

            # Ablation models (WITHOUT VecNormalize) 🔥
            'M2_sr2l_no_vecnorm': {
                'name': 'M2_ablation: SR2L (no VecNorm)',
                'path': '../experiments/ppo_sr2l_no_vecnorm_tdxq60kz/final_model',
                'vec_path': None,  # 🔥 No VecNormalize!
                'has_vecnorm': False
            },
            'M3_dr_no_vecnorm': {
                'name': 'M3_ablation: DR (no VecNorm)',
                'path': '../experiments/ppo_dr_no_vecnorm_b5zqadfj/final_model',
                'vec_path': None,  # 🔥 No VecNormalize!
                'has_vecnorm': False
            },
            'M4_combo_no_vecnorm': {
                'name': 'M4_ablation: Combo (no VecNorm)',
                'path': '../experiments/ppo_combo_no_vecnorm_qubss2iq/final_model',
                'vec_path': None,  # 🔥 No VecNormalize!
                'has_vecnorm': False
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

    def load_model(self, model_key):
        """Load model with or without VecNormalize"""
        config = self.models[model_key]

        print(f"\n{'='*60}")
        print(f"Loading: {config['name']}")
        print(f"VecNormalize: {' YES' if config['has_vecnorm'] else '❌ NO (ABLATION)'}")
        print(f"{'='*60}")

        # Create environment
        env = self.create_environment()

        # Conditionally load VecNormalize
        if config['vec_path'] is not None:
            env = VecNormalize.load(config['vec_path'], env)
            env.training = False
            env.norm_reward = False
            print("  VecNormalize loaded ")
        else:
            print("  VecNormalize SKIPPED ❌ (ablation mode)")

        # Load model
        model = PPO.load(config['path'], env=env)

        return model, env

    def evaluate_baseline(self, model_key):
        """Test 1: Baseline performance (no perturbations)"""
        print(f"\n{'='*80}")
        print(f"TEST 1: BASELINE PERFORMANCE - {self.models[model_key]['name']}")
        print('='*80)

        model, env = self.load_model(model_key)
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
            'mean': np.mean(distances),
            'std': np.std(distances),
            'model': self.models[model_key]['name'],
            'has_vecnorm': self.models[model_key]['has_vecnorm']
        }

        print(f"\n Distance: {result['mean']:.2f}m ± {result['std']:.2f}m")
        return result

    def evaluate_sensor_noise(self, model_key, noise_std, baseline_dist):
        """Test 2: Sensor noise robustness"""
        model, env = self.load_model(model_key)
        distances = []

        for ep in tqdm(range(self.num_rollouts), desc=f"Noise σ={noise_std}"):
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

        result = {
            'noise_std': noise_std,
            'mean': mean_dist,
            'std': np.std(distances),
            'retention_pct': retention,
            'model': self.models[model_key]['name'],
            'has_vecnorm': self.models[model_key]['has_vecnorm']
        }

        print(f"  σ={noise_std}: {mean_dist:.2f}m ({retention:.1f}% retention)")
        return result

    def evaluate_joint_failure(self, model_key, joint_idx, baseline_dist):
        """Test 3: Joint failure robustness"""
        model, env = self.load_model(model_key)
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
            'mean': mean_dist,
            'std': np.std(distances),
            'retention_pct': retention,
            'model': self.models[model_key]['name'],
            'has_vecnorm': self.models[model_key]['has_vecnorm']
        }

    def run_full_suite(self):
        """Run all 3 tests on all 7 models"""
        print("\n" + "="*80)
        print("VECNORMALIZE ABLATION STUDY")
        print("="*80)
        print("Testing 7 models:")
        for key, config in self.models.items():
            vecnorm_status = "WITH VecNorm" if config['has_vecnorm'] else "WITHOUT VecNorm"
            print(f"  {config['name']:40s} - {vecnorm_status}")
        print("="*80)

        results = {}

        for model_key in self.models.keys():
            results[model_key] = {}

            # Test 1: Baseline
            baseline_result = self.evaluate_baseline(model_key)
            results[model_key]['baseline'] = baseline_result
            baseline_dist = baseline_result['mean']

            # Test 2: Sensor noise (3 levels)
            print(f"\n{'='*80}")
            print(f"TEST 2: SENSOR NOISE - {self.models[model_key]['name']}")
            print('='*80)

            results[model_key]['sensor_noise'] = []
            for noise_std in [0.01, 0.05, 0.1]:
                noise_result = self.evaluate_sensor_noise(model_key, noise_std, baseline_dist)
                results[model_key]['sensor_noise'].append(noise_result)

            # Test 3: Joint failures (all 8 joints)
            print(f"\n{'='*80}")
            print(f"TEST 3: JOINT FAILURES - {self.models[model_key]['name']}")
            print('='*80)

            results[model_key]['joint_failures'] = []
            for joint_idx in tqdm(range(8), desc="Testing joints"):
                joint_result = self.evaluate_joint_failure(model_key, joint_idx, baseline_dist)
                results[model_key]['joint_failures'].append(joint_result)
                print(f"  {self.joint_names[joint_idx]:10s}: {joint_result['retention_pct']:.1f}%")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"vecnorm_ablation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n Results saved to {results_file}")

        return results


if __name__ == "__main__":
    evaluator = VecNormAblationEvaluator()
    results = evaluator.run_full_suite()
