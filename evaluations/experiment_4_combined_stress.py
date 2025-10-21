#!/usr/bin/env python3
"""
EXPERIMENT 4: COMBINED STRESS EVALUATION
Tests all 4 models under combined sensor noise + joint failure conditions
Uses best performing joints: Hip_1, Hip_4, Ankle_2, Ankle_3
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

class CombinedStressEvaluator:
    """Evaluates all 4 models under combined noise + joint failure stress"""

    def __init__(self):
        # Episode parameters
        self.episode_length = 1200  # 20 seconds at 60fps
        self.num_rollouts = 100

        # Joint failure parameters
        self.delay_steps = 120  # 2-second delay before locking
        self.lock_value = 0.0

        # Best performing joints based on V7.7E data
        self.best_joints = {
            'hip_1': {'name': 'Hip_1', 'retention': 81.8},
            'hip_4': {'name': 'Hip_4', 'retention': 43.5},
            'ankle_2': {'name': 'Ankle_2', 'retention': 59.4},
            'ankle_3': {'name': 'Ankle_3', 'retention': 43.6}
        }

        # Joint mapping to action indices
        self.joint_to_action = {
            'hip_1': 0,    'ankle_1': 1,
            'hip_2': 2,    'ankle_2': 3,
            'hip_3': 4,    'ankle_3': 5,
            'hip_4': 6,    'ankle_4': 7
        }

        # Combined test scenarios
        self.test_scenarios = [
            {
                'name': 'Mild Combined',
                'noise': 0.05,
                'failed_joints': ['ankle_2'],
                'description': '5X training noise + best ankle failure'
            },
            {
                'name': 'Moderate Combined',
                'noise': 0.10,
                'failed_joints': ['hip_4'],
                'description': '10X training noise + rear hip failure'
            },
            {
                'name': 'Challenging Combined',
                'noise': 0.10,
                'failed_joints': ['ankle_3'],
                'description': '10X training noise + rear ankle failure'
            },
            {
                'name': 'Severe Combined',
                'noise': 0.20,
                'failed_joints': ['hip_1'],
                'description': '20X training noise + best hip failure'
            },
            {
                'name': 'Extreme Dual Failure',
                'noise': 0.05,
                'failed_joints': ['hip_1', 'ankle_2'],
                'description': '5X training noise + dual best joints'
            },
            {
                'name': 'Ultimate Challenge',
                'noise': 0.10,
                'failed_joints': ['hip_4', 'ankle_3'],
                'description': '10X training noise + dual rear joints'
            }
        ]

        # Success/failure thresholds
        self.success_threshold = 1.5  # meters
        self.failure_height_threshold = 0.2

        # Model configurations
        self.models = {
            'M1_baseline': {
                'name': 'PPO Baseline',
                'path': '../done/ppo_baseline_ueqbjf2x/best_model/best_model',
                'vec_path': '../done/ppo_baseline_ueqbjf2x/vec_normalize.pkl',
                'description': 'PPO only (no robustness training)'
            },
            'M2_sr2l': {
                'name': 'PPO + SR2L',
                'path': '../done/ppo_sr2l_forward_m7gtjtpa/final_model',
                'vec_path': '../done/ppo_sr2l_forward_m7gtjtpa/vec_normalize.pkl',
                'description': 'PPO + SR2L (sensor noise specialist)'
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
                'description': 'PPO + SR2L + DR (COMBINED SPECIALIST)'
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
        joint_end = 29

        for idx in range(joint_start, min(joint_end, len(obs_copy[0]))):
            noise = rng.normal(0, noise_std)
            obs_copy[0][idx] += noise

        return obs_copy

    def apply_joint_failure(self, action, failed_joints, step_count):
        """Apply joint failure by locking specific joints after delay"""
        if step_count < self.delay_steps:
            return action

        modified_action = action.copy()
        for joint_name in failed_joints:
            if joint_name in self.joint_to_action:
                action_idx = self.joint_to_action[joint_name]
                modified_action[0][action_idx] = self.lock_value

        return modified_action

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

    def evaluate_model_scenario(self, model_key, scenario):
        """Run rollouts for specific combined stress scenario"""
        model, env = self.load_model(model_key)
        rng = np.random.default_rng(42)

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
                noisy_obs = self.apply_sensor_noise(obs, scenario['noise'], rng)

                # Get action from model
                action, _ = model.predict(noisy_obs, deterministic=True)

                # Apply joint failure
                modified_action = self.apply_joint_failure(action, scenario['failed_joints'], step_count)

                # Track position BEFORE step (in case episode ends and resets)
                try:
                    current_x = unwrapped.data.qpos[0]
                    positions.append(current_x)
                except:
                    positions.append(0.0)

                # Step environment
                obs, reward, done, info = env.step(modified_action)
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
            'scenario_name': scenario['name'],
            'noise_level': scenario['noise'],
            'failed_joints': scenario['failed_joints'],
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
        """Evaluate model across all combined stress scenarios"""
        config = self.models[model_key]

        print(f"\n{'='*60}")
        print(f"Evaluating: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}")

        scenario_results = []

        for scenario in tqdm(self.test_scenarios, desc=f"{config['name']}"):
            result = self.evaluate_model_scenario(model_key, scenario)
            scenario_results.append(result)

            # Print quick summary
            print(f"  {scenario['name']:>25}: "
                  f"Distance={result['distance']['mean']:.3f}m, "
                  f"Success={result['success_rate']*100:.1f}%, "
                  f"Failure={result['failure_rate']*100:.1f}%")

        return {
            'model_key': model_key,
            'model_name': config['name'],
            'scenario_results': scenario_results
        }

    def run_experiment(self):
        """Run combined stress evaluation for all 4 models"""
        print("\n" + "="*60)
        print("EXPERIMENT 4: COMBINED STRESS EVALUATION")
        print("="*60)
        print("Testing all 4 models under combined noise + joint failures")
        print(f"- Scenarios: {len(self.test_scenarios)}")
        print(f"- Best joints: {list(self.best_joints.keys())}")
        print(f"- Delayed locking: {self.delay_steps} steps (2 seconds)")
        print(f"- Episode length: {self.episode_length} steps (20 seconds)")
        print(f"- Rollouts per scenario: {self.num_rollouts}")
        print(f"- Total rollouts: {len(self.test_scenarios) * self.num_rollouts * 4} episodes")
        print("\nTest Scenarios:")
        for i, scenario in enumerate(self.test_scenarios, 1):
            print(f"  {i}. {scenario['name']}: {scenario['description']}")
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
        output_dir = 'evaluations/experiment_4_combined_stress/data'
        os.makedirs(output_dir, exist_ok=True)

        output_path = f"{output_dir}/combined_stress_results_{timestamp}.json"

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✅ Results saved to: {output_path}")

    def print_comparison(self):
        """Print comparison table across all models and scenarios"""
        print("\n" + "="*110)
        print("COMBINED STRESS ROBUSTNESS COMPARISON")
        print("="*110)

        # Print header
        print(f"{'Scenario':<30}", end='')
        for model_key in ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo']:
            name = self.models[model_key]['name'][:20]
            print(f"{name:<20}", end='')
        print()
        print("-"*110)

        # Print results for each scenario
        for i, scenario in enumerate(self.test_scenarios):
            scenario_name = scenario['name']
            print(f"{scenario_name:<30}", end='')

            for model_key in ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo']:
                result = self.results[model_key]['scenario_results'][i]
                dist_mean = result['distance']['mean']
                success_rate = result['success_rate'] * 100
                print(f"{dist_mean:>6.2f}m ({success_rate:>4.1f}%) ", end='')

            print()

        print("="*110)

        # Calculate average performance across all scenarios
        print("\nAVERAGE PERFORMANCE ACROSS ALL COMBINED STRESS SCENARIOS:")
        print("-"*110)

        for model_key in ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo']:
            all_distances = [r['distance']['mean'] for r in self.results[model_key]['scenario_results']]
            all_success = [r['success_rate'] for r in self.results[model_key]['scenario_results']]

            avg_dist = np.mean(all_distances)
            avg_success = np.mean(all_success) * 100

            print(f"{self.models[model_key]['name']:<25}: "
                  f"Avg Distance={avg_dist:.3f}m, "
                  f"Avg Success={avg_success:.1f}%")

        # Find best performer at combined stress
        best_model = max(
            ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo'],
            key=lambda k: np.mean([r['distance']['mean'] for r in self.results[k]['scenario_results']])
        )

        print(f"\n✅ Best Combined Stress Performance: {self.models[best_model]['name']}")

        # Test synergy hypothesis: M4 should outperform max(M2, M3)
        m4_avg = np.mean([r['distance']['mean'] for r in self.results['M4_combo']['scenario_results']])
        m2_avg = np.mean([r['distance']['mean'] for r in self.results['M2_sr2l']['scenario_results']])
        m3_avg = np.mean([r['distance']['mean'] for r in self.results['M3_dr']['scenario_results']])

        max_specialist = max(m2_avg, m3_avg)
        if m4_avg > max_specialist:
            improvement = ((m4_avg - max_specialist) / max_specialist) * 100
            print(f"\n🎯 SYNERGY CONFIRMED: Ultimate Combo (M4) outperforms best specialist by {improvement:.1f}%")
            print(f"   M4: {m4_avg:.3f}m  vs  Max(M2, M3): {max_specialist:.3f}m")
        else:
            print(f"\n⚠️ NO SYNERGY: Ultimate Combo (M4) does not exceed specialists")
            print(f"   M4: {m4_avg:.3f}m  vs  M2: {m2_avg:.3f}m  vs  M3: {m3_avg:.3f}m")

        print("\n" + "="*110)

if __name__ == "__main__":
    evaluator = CombinedStressEvaluator()
    evaluator.run_experiment()
