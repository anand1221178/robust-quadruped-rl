#!/usr/bin/env python3
"""
🔬 ULTIMATE 4-WAY ABLATION STUDY COMPARISON 🔬
Complete robustness method comparison for research paper

Models to compare:
1. PPO Baseline - Pure forward locomotion (done/ppo_baseline_ueqbjf2x)
2. PPO + SR2L - Sensor noise robustness (done/ppo_sr2l_forward_m7gtjtpa)
3. PPO + DR - Joint failure robustness (done/dr/v7_7e_ultra_speed)
4. PPO + SR2L + DR - ULTIMATE COMBO (experiments/ultimate_robustness_combo_ju7lfsk2)

Test conditions:
- Baseline performance (no perturbations)
- Sensor noise robustness (0%, 1%, 2%, 5%, 10% noise)
- Joint failure robustness (0%, 10%, 25%, 50% failure rates)
- Combined stress test (joint failures + sensor noise)
"""

import sys
sys.path.append('src')

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
from envs.domain_randomization_wrapper import DomainRandomizationWrapper
import realant_sim
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class AblationStudyEvaluator:
    """Comprehensive 4-way ablation study evaluation"""

    def __init__(self):
        self.models = {
            'PPO_Baseline': {
                'path': 'done/ppo_baseline_ueqbjf2x/best_model/best_model.zip',
                'vec_path': 'done/ppo_baseline_ueqbjf2x/vec_normalize.pkl',
                'description': 'Pure PPO forward locomotion',
                'expected_specialty': 'Speed optimization'
            },
            'PPO_SR2L': {
                'path': 'done/ppo_sr2l_forward_m7gtjtpa/final_model.zip',
                'vec_path': 'done/ppo_sr2l_forward_m7gtjtpa/vec_normalize.pkl',
                'description': 'PPO + SR2L sensor noise robustness',
                'expected_specialty': 'Sensor noise tolerance'
            },
            'PPO_DR': {
                'path': 'done/dr/Curr best/v7_7e_ultra_speed_jtfwl2qf/final_model.zip',
                'vec_path': 'done/dr/Curr best/v7_7e_ultra_speed_jtfwl2qf/vec_normalize.pkl',
                'description': 'PPO + Domain Randomization joint failures',
                'expected_specialty': 'Joint failure adaptation'
            },
            'PPO_ULTIMATE': {
                'path': 'experiments/ultimate_robustness_combo_ju7lfsk2/final_model.zip',
                'vec_path': 'experiments/ultimate_robustness_combo_ju7lfsk2/vec_normalize.pkl',
                'description': 'PPO + SR2L + DR ultimate combo',
                'expected_specialty': 'Universal robustness'
            }
        }

        # Test conditions
        self.noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10]  # 0% to 10% sensor noise
        self.failure_rates = [0.0, 0.1, 0.25, 0.5]         # 0% to 50% joint failures
        self.episodes_per_test = 10                         # Statistical significance

        # Results storage
        self.results = {}

    def create_test_env(self, noise_std=0.0, failure_prob=0.0):
        """Create test environment with specified perturbations"""
        env = gym.make('RealAntMujoco-v0')
        env = SuccessRewardWrapper(env)

        # Add domain randomization if needed
        if failure_prob > 0:
            env = DomainRandomizationWrapper(
                env,
                joint_dropout_prob=failure_prob,
                min_dropped_joints=0,
                max_dropped_joints=2,
                sensor_noise_std=noise_std
            )

        env = TimeLimit(env, max_episode_steps=2500)
        env = DummyVecEnv([lambda: env])
        return env

    def evaluate_model(self, model_name, model_path, vec_path, test_conditions):
        """Evaluate single model across all test conditions"""
        print(f"\\n{'='*60}")
        print(f"EVALUATING: {model_name}")
        print(f"{'='*60}")

        model_results = {}

        for condition_name, (noise, failure_rate) in test_conditions.items():
            print(f"\\nTest: {condition_name} (noise={noise:.1%}, failures={failure_rate:.1%})")

            # Create test environment
            env = self.create_test_env(noise_std=noise, failure_prob=failure_rate)

            try:
                # Load model
                if vec_path:
                    env = VecNormalize.load(vec_path, env)
                    env.training = False
                    env.norm_reward = False

                model = PPO.load(model_path, env=env)

                # Run episodes
                episode_rewards = []
                episode_distances = []
                episode_velocities = []

                for episode in range(self.episodes_per_test):
                    obs = env.reset()
                    positions = [env.envs[0].unwrapped.get_body_com("torso")[0]]
                    total_reward = 0

                    for step in range(2500):
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, done, info = env.step(action)
                        total_reward += reward[0]

                        # Track position
                        current_pos = env.envs[0].unwrapped.get_body_com("torso")[0]
                        positions.append(current_pos)

                        if done[0]:
                            break

                    # Calculate metrics
                    total_distance = positions[-1] - positions[0]
                    avg_velocity = total_distance / (len(positions) * 0.02)  # 20ms timesteps

                    episode_rewards.append(total_reward)
                    episode_distances.append(total_distance)
                    episode_velocities.append(avg_velocity)

                # Store results
                model_results[condition_name] = {
                    'noise_level': noise,
                    'failure_rate': failure_rate,
                    'avg_reward': np.mean(episode_rewards),
                    'std_reward': np.std(episode_rewards),
                    'avg_velocity': np.mean(episode_velocities),
                    'std_velocity': np.std(episode_velocities),
                    'avg_distance': np.mean(episode_distances),
                    'success_rate': len([v for v in episode_velocities if v > 0.05]) / len(episode_velocities)
                }

                print(f"  Velocity: {np.mean(episode_velocities):.3f} ± {np.std(episode_velocities):.3f} m/s")
                print(f"  Success rate: {model_results[condition_name]['success_rate']:.1%}")

            except Exception as e:
                print(f"  ❌ FAILED: {e}")
                model_results[condition_name] = {'error': str(e)}

            finally:
                env.close()

        return model_results

    def run_complete_ablation(self):
        """Run complete 4-way ablation study"""
        print("🔬 STARTING ULTIMATE 4-WAY ABLATION STUDY 🔬")
        print("Testing all combinations of sensor noise and joint failures...")

        # Define test conditions
        test_conditions = {
            'Baseline': (0.0, 0.0),
            'Low_Noise': (0.01, 0.0),
            'Med_Noise': (0.02, 0.0),
            'High_Noise': (0.05, 0.0),
            'Extreme_Noise': (0.10, 0.0),
            'Low_Failures': (0.0, 0.1),
            'Med_Failures': (0.0, 0.25),
            'High_Failures': (0.0, 0.5),
            'Combined_Mild': (0.01, 0.1),
            'Combined_Moderate': (0.02, 0.25),
            'Combined_Extreme': (0.05, 0.5)
        }

        # Evaluate each model
        for model_name, model_info in self.models.items():
            if model_name == 'PPO_ULTIMATE':
                # Check if ultimate combo is trained yet
                import os
                if not os.path.exists(model_info['path']):
                    print(f"\\n⏳ {model_name} still training, skipping for now...")
                    continue

            self.results[model_name] = self.evaluate_model(
                model_name,
                model_info['path'],
                model_info['vec_path'],
                test_conditions
            )

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"ablation_study_results_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\\n📊 Results saved to: {results_file}")

        # Generate comparison plots
        self.create_comparison_plots(timestamp)

        return self.results

    def create_comparison_plots(self, timestamp):
        """Create comprehensive comparison visualizations"""
        print("\\n📈 Generating comparison plots...")

        # Extract baseline velocities for retention calculation
        baseline_velocities = {}
        for model_name, results in self.results.items():
            if 'Baseline' in results and 'avg_velocity' in results['Baseline']:
                baseline_velocities[model_name] = results['Baseline']['avg_velocity']

        # 1. Baseline Performance Comparison
        plt.figure(figsize=(12, 8))

        models = list(baseline_velocities.keys())
        velocities = list(baseline_velocities.values())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        bars = plt.bar(models, velocities, color=colors[:len(models)])
        plt.title('Baseline Performance Comparison\\n(No Perturbations)', fontsize=16, fontweight='bold')
        plt.ylabel('Velocity (m/s)', fontsize=12)
        plt.xticks(rotation=45)

        # Add value labels on bars
        for bar, velocity in zip(bars, velocities):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{velocity:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'baseline_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Sensor Noise Robustness
        plt.figure(figsize=(14, 8))

        noise_conditions = ['Baseline', 'Low_Noise', 'Med_Noise', 'High_Noise', 'Extreme_Noise']
        noise_labels = ['0%', '1%', '2%', '5%', '10%']

        for i, (model_name, results) in enumerate(self.results.items()):
            retentions = []
            baseline_vel = baseline_velocities.get(model_name, 1.0)

            for condition in noise_conditions:
                if condition in results and 'avg_velocity' in results[condition]:
                    retention = (results[condition]['avg_velocity'] / baseline_vel) * 100
                    retentions.append(retention)
                else:
                    retentions.append(0)

            plt.plot(noise_labels, retentions, marker='o', linewidth=3,
                    label=model_name, color=colors[i % len(colors)])

        plt.title('Sensor Noise Robustness Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Sensor Noise Level', fontsize=12)
        plt.ylabel('Performance Retention (%)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=100, color='black', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(f'noise_robustness_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 Plots saved: baseline_comparison_{timestamp}.png, noise_robustness_{timestamp}.png")

def main():
    """Run the complete ablation study"""
    evaluator = AblationStudyEvaluator()
    results = evaluator.run_complete_ablation()

    print("\\n🏆 ABLATION STUDY COMPLETE! 🏆")
    print("\\nSUMMARY:")
    print("="*50)

    for model_name, model_results in results.items():
        if 'Baseline' in model_results:
            baseline = model_results['Baseline']
            if 'avg_velocity' in baseline:
                print(f"{model_name}: {baseline['avg_velocity']:.3f} m/s baseline")
            else:
                print(f"{model_name}: Failed to evaluate")

if __name__ == "__main__":
    main()