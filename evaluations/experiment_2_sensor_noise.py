#!/usr/bin/env python3
"""
EXPERIMENT 2: SENSOR NOISE ROBUSTNESS EVALUATION
Tests all 4 models under increasing sensor noise levels
Metrics: Success rate, failure rate, total distance traveled across noise levels
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

class SensorNoiseEvaluator:
    """Evaluates all 4 models under sensor noise conditions"""

    def __init__(self):
        # Episode parameters
        self.episode_length = 1200  # 20 seconds at 60fps
        self.num_rollouts = 100

        # EXTREME noise levels matching extreme_championship.py - up to 300X training noise!
        self.noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]

        # Success/failure thresholds
        self.success_threshold = 1.5  # meters
        self.failure_height_threshold = 0.2

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
                'description': 'PPO + SR2L (SENSOR NOISE SPECIALIST) - RETRAINED 32M'
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

    def apply_sensor_noise(self, obs, noise_std, rng):
        """Apply sensor noise to joint observations (dims 13-28)"""
        if noise_std <= 0:
            return obs

        obs_copy = obs.copy()
        joint_start = 13
        joint_end = 29  # 16 joint sensor values

        for idx in range(joint_start, min(joint_end, len(obs_copy[0]))):
            noise = rng.normal(0, noise_std)
            obs_copy[0][idx] += noise

        return obs_copy

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

    def check_robot_fallen(self, env):
        """Check if robot has fallen over"""
        try:
            unwrapped = env.envs[0].unwrapped
            if hasattr(unwrapped, 'data'):
                torso_height = unwrapped.data.qpos[2]
                return torso_height < self.failure_height_threshold
        except:
            pass
        return False

    def evaluate_model_at_noise(self, model_key, noise_level):
        """Run rollouts at specific noise level"""
        model, env = self.load_model(model_key)
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility

        rollout_results = []

        for rollout in range(self.num_rollouts):
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
                # Apply sensor noise to observations
                noisy_obs = self.apply_sensor_noise(obs, noise_level, rng)

                # Get action from model
                action, _ = model.predict(noisy_obs, deterministic=True)

                # Track position BEFORE step (in case episode ends and resets)
                try:
                    current_x = unwrapped.data.qpos[0]
                    positions.append(current_x)
                except:
                    positions.append(0.0)

                # Step environment (with clean observations)
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
            'noise_level': noise_level,
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
            'failure_rate': float(failure_count / self.num_rollouts),
            'rollouts': rollout_results
        }

        env.close()
        return summary

    def evaluate_model(self, model_key):
        """Evaluate model across all noise levels"""
        config = self.models[model_key]

        print(f"\n{'='*60}")
        print(f"Evaluating: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}")

        noise_results = []

        for noise_level in tqdm(self.noise_levels, desc=f"{config['name']}"):
            result = self.evaluate_model_at_noise(model_key, noise_level)
            noise_results.append(result)

            # Print quick summary
            print(f"  Noise {noise_level:.2f}: "
                  f"Distance={result['distance']['mean']:.3f}m, "
                  f"Success={result['success_rate']*100:.1f}%, "
                  f"Failure={result['failure_rate']*100:.1f}%")

        return {
            'model_key': model_key,
            'model_name': config['name'],
            'noise_results': noise_results
        }

    def run_experiment(self):
        """Run sensor noise evaluation for all 4 models"""
        print("\n" + "="*60)
        print("EXPERIMENT 2: EXTREME SENSOR NOISE ROBUSTNESS")
        print("="*60)
        print("Testing all 4 models under EXTREME sensor noise")
        print(f"- Noise levels: {self.noise_levels}")
        print(f"- Range: 0X → 300X training noise (σ=0.01)")
        print(f"- Applied to joint sensors (obs dims 13-28)")
        print(f"- Episode length: {self.episode_length} steps (20 seconds)")
        print(f"- Rollouts per noise level: {self.num_rollouts}")
        print(f"- Total rollouts: {len(self.noise_levels) * self.num_rollouts * 4} episodes")
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
        output_dir = 'evaluations/experiment_2_sensor_noise/data'
        os.makedirs(output_dir, exist_ok=True)

        output_path = f"{output_dir}/sensor_noise_results_{timestamp}.json"

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n Results saved to: {output_path}")

    def print_comparison(self):
        """Print comparison table across all models and noise levels"""
        print("\n" + "="*100)
        print("SENSOR NOISE ROBUSTNESS COMPARISON")
        print("="*100)

        # Print header
        print(f"{'Noise Level':<15}", end='')
        for model_key in ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo']:
            name = self.models[model_key]['name']
            print(f"{name:<25}", end='')
        print()
        print("-"*100)

        # Print results for each noise level
        for i, noise_level in enumerate(self.noise_levels):
            print(f"{noise_level:<15.2f}", end='')

            for model_key in ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo']:
                result = self.results[model_key]['noise_results'][i]
                dist_mean = result['distance']['mean']
                success_rate = result['success_rate'] * 100
                print(f"{dist_mean:>6.2f}m ({success_rate:>4.1f}%)  ", end='')

            print()

        print("="*100)

        # Find best performers at high noise
        print("\nKEY FINDINGS:")
        high_noise_idx = -1  # Last noise level (0.7)
        high_noise_level = self.noise_levels[high_noise_idx]

        best_at_high_noise = max(
            ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo'],
            key=lambda k: self.results[k]['noise_results'][high_noise_idx]['distance']['mean']
        )

        print(f"\n Best at HIGH noise (σ={high_noise_level}):")
        print(f"   {self.models[best_at_high_noise]['name']}: "
              f"{self.results[best_at_high_noise]['noise_results'][high_noise_idx]['distance']['mean']:.3f}m "
              f"({self.results[best_at_high_noise]['noise_results'][high_noise_idx]['success_rate']*100:.1f}% success)")

        # Check if SR2L (M2) dominates
        m2_distances = [r['distance']['mean'] for r in self.results['M2_sr2l']['noise_results']]
        m1_distances = [r['distance']['mean'] for r in self.results['M1_baseline']['noise_results']]

        avg_m2 = np.mean(m2_distances[1:])  # Exclude baseline
        avg_m1 = np.mean(m1_distances[1:])

        if avg_m2 > avg_m1:
            improvement = ((avg_m2 - avg_m1) / avg_m1) * 100
            print(f"\n🎯 SR2L (M2) shows {improvement:.1f}% better average performance across noise levels vs Baseline (M1)")

        print("\n" + "="*100)

if __name__ == "__main__":
    evaluator = SensorNoiseEvaluator()
    evaluator.run_experiment()
