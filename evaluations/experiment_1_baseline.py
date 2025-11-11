#!/usr/bin/env python3
"""
EXPERIMENT 1: BASELINE PERFORMANCE EVALUATION
Tests all 4 models under ideal conditions (no sensor noise, no joint failures)
Metrics: Success rate, failure rate, total distance traveled
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

class BaselineEvaluator:
    """Evaluates all 4 models under perfect conditions"""

    def __init__(self):
        # Episode parameters
        self.episode_length = 1200  # 20 seconds at 60fps
        self.num_rollouts = 100

        # Success/failure thresholds
        self.success_threshold = 1.5  # meters
        self.failure_height_threshold = 0.2  # Consider fallen if below this height

        # Model configurations (UPDATED: Using 32M retrained M1/M2, keeping champion M3/M4)
        self.models = {
            'M1_baseline': {
                'name': 'PPO Baseline (32M)',
                'path': '../experiments/M1_baseline_32M_RETRAINED_ym2jcllj/final_model',
                'vec_path': '../experiments/M1_baseline_32M_RETRAINED_ym2jcllj/vec_normalize.pkl',
                'description': 'PPO only (no robustness training) - RETRAINED 32M'
            },
            'M2_sr2l': {
                'name': 'PPO + SR2L (32M)',
                'path': '../experiments/M2_sr2l_32M_RETRAINED_ze09p0vf/final_model',
                'vec_path': '../experiments/M2_sr2l_32M_RETRAINED_ze09p0vf/vec_normalize.pkl',
                'description': 'PPO + Smooth Regularization (sensor noise specialist) - RETRAINED 32M'
            },
            'M3_dr': {
                'name': 'PPO + DR (V7.7E Champion)',
                'path': '../done/v7_7e_ultra_speed_jtfwl2qf/final_model',
                'vec_path': '../done/v7_7e_ultra_speed_jtfwl2qf/vec_normalize.pkl',
                'description': 'PPO + Domain Randomization (joint failure specialist) - 45% avg retention'
            },
            'M4_combo': {
                'name': 'Ultimate Combo',
                'path': '../done/ultimate_robustness_combo_ju7lfsk2/final_model',
                'vec_path': '../done/ultimate_robustness_combo_ju7lfsk2/vec_normalize.pkl',
                'description': 'PPO + SR2L + DR (full robustness pipeline)'
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

        print(f"\n{'='*60}")
        print(f"Loading: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}")

        # Create environment
        env = self.create_environment()

        # Load VecNormalize
        env = VecNormalize.load(config['vec_path'], env)
        env.training = False
        env.norm_reward = False

        # Load model
        model = PPO.load(config['path'], env=env)

        return model, env

    def check_robot_fallen(self, env):
        """Check if robot has fallen over"""
        try:
            unwrapped = env.envs[0].unwrapped
            if hasattr(unwrapped, 'data'):
                # Get torso height (z-coordinate)
                torso_height = unwrapped.data.qpos[2]
                return torso_height < self.failure_height_threshold
        except:
            pass
        return False

    def evaluate_model(self, model_key):
        """Run 100 rollouts and collect metrics"""
        model, env = self.load_model(model_key)

        rollout_results = []

        print(f"\nRunning {self.num_rollouts} rollouts...")
        for rollout in tqdm(range(self.num_rollouts)):
            obs = env.reset()
            done = False
            step_count = 0

            # Track metrics
            positions = []
            rewards = []
            fell = False

            # Get initial position
            try:
                unwrapped = env.envs[0].unwrapped
                start_x = unwrapped.data.qpos[0]
            except:
                start_x = 0.0

            while not done and step_count < self.episode_length:
                # Get action from model
                action, _ = model.predict(obs, deterministic=True)

                # Track position BEFORE step (in case episode ends and resets)
                try:
                    current_x = unwrapped.data.qpos[0]
                    positions.append(current_x)
                except:
                    positions.append(0.0)

                # Step environment
                obs, reward, done, info = env.step(action)
                rewards.append(reward[0])

                # Check for fall
                if self.check_robot_fallen(env):
                    fell = True

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
                'failure': bool(fell),
                'steps': int(step_count)
            })

        # Calculate aggregate statistics
        distances = [r['distance'] for r in rollout_results]
        rewards = [r['reward'] for r in rollout_results]
        success_count = sum(1 for r in rollout_results if r['success'])
        failure_count = sum(1 for r in rollout_results if r['failure'])

        summary = {
            'model_key': model_key,
            'model_name': self.models[model_key]['name'],
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
                'std': float(np.std(rewards)),
                'min': float(np.min(rewards)),
                'max': float(np.max(rewards)),
                'median': float(np.median(rewards))
            },
            'success_rate': float(success_count / self.num_rollouts),
            'failure_rate': float(failure_count / self.num_rollouts),
            'rollouts': rollout_results
        }

        # Print summary
        print(f"\n{'='*60}")
        print(f"RESULTS: {self.models[model_key]['name']}")
        print(f"{'='*60}")
        print(f"Distance:     {summary['distance']['mean']:.3f} ± {summary['distance']['std']:.3f} m")
        print(f"Reward:       {summary['reward']['mean']:.1f} ± {summary['reward']['std']:.1f}")
        print(f"Success Rate: {summary['success_rate']*100:.1f}% ({success_count}/{self.num_rollouts} episodes)")
        print(f"Failure Rate: {summary['failure_rate']*100:.1f}% ({failure_count}/{self.num_rollouts} episodes)")
        print(f"{'='*60}")

        env.close()
        return summary

    def run_experiment(self):
        """Run baseline evaluation for all 4 models"""
        print("\n" + "="*60)
        print("EXPERIMENT 1: BASELINE PERFORMANCE EVALUATION")
        print("="*60)
        print("Testing all 4 models under ideal conditions")
        print(f"- No sensor noise")
        print(f"- No joint failures")
        print(f"- Episode length: {self.episode_length} steps (20 seconds)")
        print(f"- Rollouts per model: {self.num_rollouts}")
        print(f"- Success threshold: {self.success_threshold}m")
        print("="*60)

        # Evaluate each model
        for model_key in ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo']:
            self.results[model_key] = self.evaluate_model(model_key)

        # Save results
        self.save_results()

        # Print comparison
        self.print_comparison()

    def save_results(self):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = 'evaluations/experiment_1_baseline/data'
        os.makedirs(output_dir, exist_ok=True)

        output_path = f"{output_dir}/baseline_results_{timestamp}.json"

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n Results saved to: {output_path}")

    def print_comparison(self):
        """Print comparison table of all models"""
        print("\n" + "="*80)
        print("BASELINE PERFORMANCE COMPARISON")
        print("="*80)
        print(f"{'Model':<25} {'Distance (m)':<15} {'Success Rate':<15} {'Failure Rate':<15}")
        print("-"*80)

        for model_key in ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo']:
            res = self.results[model_key]
            print(f"{res['model_name']:<25} "
                  f"{res['distance']['mean']:>6.3f} ± {res['distance']['std']:<5.3f}  "
                  f"{res['success_rate']*100:>6.1f}%        "
                  f"{res['failure_rate']*100:>6.1f}%")

        print("="*80)
        print("\nKEY FINDINGS:")

        # Find best performer
        best_distance = max(self.results.values(), key=lambda x: x['distance']['mean'])
        best_success = max(self.results.values(), key=lambda x: x['success_rate'])
        lowest_failure = min(self.results.values(), key=lambda x: x['failure_rate'])

        print(f" Best Distance:    {best_distance['model_name']} ({best_distance['distance']['mean']:.3f}m)")
        print(f" Best Success:     {best_success['model_name']} ({best_success['success_rate']*100:.1f}%)")
        print(f" Most Stable:      {lowest_failure['model_name']} ({lowest_failure['failure_rate']*100:.1f}% failure)")

        print("\n" + "="*80)

if __name__ == "__main__":
    evaluator = BaselineEvaluator()
    evaluator.run_experiment()
