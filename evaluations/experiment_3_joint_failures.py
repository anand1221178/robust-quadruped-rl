#!/usr/bin/env python3
"""
EXPERIMENT 3: JOINT FAILURE ROBUSTNESS EVALUATION
Tests all 4 models with individual joint failures (all 8 joints)
Metrics: Success rate, failure rate, total distance traveled per joint failure
NEW: Recovery time tracking added (Oct 19, 2025)
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

# Import recovery time tracker
from utils.recovery_time_tracker import RecoveryTimeTracker

class JointFailureEvaluator:
    """Evaluates all 4 models under joint failure conditions"""

    def __init__(self):
        # Episode parameters
        self.episode_length = 1200  # 20 seconds at 60fps
        self.num_rollouts = 100

        # Joint failure parameters
        self.delay_steps = 120  # 2-second delay before locking
        self.lock_value = 0.0  # Lock joints at 0.0

        # All 8 joints to test
        self.joints_to_test = [
            'hip_1', 'ankle_1',
            'hip_2', 'ankle_2',
            'hip_3', 'ankle_3',
            'hip_4', 'ankle_4'
        ]

        # Joint mapping to action indices
        self.joint_to_action = {
            'hip_1': 0,    'ankle_1': 1,
            'hip_2': 2,    'ankle_2': 3,
            'hip_3': 4,    'ankle_3': 5,
            'hip_4': 6,    'ankle_4': 7
        }

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
                'description': 'PPO + SR2L (sensor noise specialist) - RETRAINED 32M'
            },
            'M3_dr': {
                'name': 'PPO + DR (V7.7E Champion)',
                'path': '../done/v7_7e_ultra_speed_jtfwl2qf/final_model',
                'vec_path': '../done/v7_7e_ultra_speed_jtfwl2qf/vec_normalize.pkl',
                'description': 'PPO + DR (JOINT FAILURE SPECIALIST) - 45% avg retention'
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

    def apply_joint_failure(self, action, failed_joints, step_count):
        """Apply joint failure by locking specific joints after delay"""
        if step_count < self.delay_steps:
            return action  # No failure during delay period

        # Lock failed joints
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

    def evaluate_model_with_failure(self, model_key, failed_joint):
        """Run rollouts with specific joint failure"""
        model, env = self.load_model(model_key)

        rollout_results = []

        for rollout in range(self.num_rollouts):
            obs = env.reset()
            done = False
            step_count = 0

            # Track metrics
            positions = []
            rewards = []
            fell = False

            # Initialize recovery time tracker
            recovery_tracker = RecoveryTimeTracker(
                fault_injection_step=self.delay_steps,
                recovery_threshold=0.5,  # 50% of pre-fault velocity
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
                # Track recovery metrics (do this BEFORE action)
                recovery_tracker.track_step(obs[0])  # obs is vectorized, use [0]

                # Get action from model
                action, _ = model.predict(obs, deterministic=True)

                # Apply joint failure (with delayed locking)
                modified_action = self.apply_joint_failure(action, [failed_joint], step_count)

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

            # Get recovery metrics
            recovery_results = recovery_tracker.get_results()

            rollout_results.append({
                'rollout_id': rollout,
                'distance': float(total_distance),
                'reward': float(total_reward),
                'success': bool(success),
                'failure': bool(fell),
                'steps': int(step_count),
                # Recovery time metrics (NEW)
                'recovered': recovery_results['recovered'],
                'recovery_time_seconds': recovery_results['recovery_time_seconds'],
                'pre_fault_velocity': recovery_results['pre_fault_velocity'],
                'post_fault_avg_velocity': recovery_results['post_fault_avg_velocity']
            })

        # Calculate aggregate statistics
        distances = [r['distance'] for r in rollout_results]
        rewards = [r['reward'] for r in rollout_results]
        success_count = sum(1 for r in rollout_results if r['success'])
        failure_count = sum(1 for r in rollout_results if r['failure'])

        # Calculate recovery statistics (NEW)
        recovery_times = [r['recovery_time_seconds'] for r in rollout_results
                         if r['recovery_time_seconds'] is not None]
        recovery_count = sum(1 for r in rollout_results if r['recovered'])
        pre_fault_velocities = [r['pre_fault_velocity'] for r in rollout_results
                               if r['pre_fault_velocity'] is not None]
        post_fault_velocities = [r['post_fault_avg_velocity'] for r in rollout_results
                                if r['post_fault_avg_velocity'] is not None]

        summary = {
            'failed_joint': failed_joint,
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
            # Recovery time metrics (NEW)
            'recovery': {
                'recovery_rate': float(recovery_count / self.num_rollouts),
                'recovery_time_mean': float(np.mean(recovery_times)) if recovery_times else None,
                'recovery_time_std': float(np.std(recovery_times)) if recovery_times else None,
                'recovery_time_median': float(np.median(recovery_times)) if recovery_times else None,
                'recovery_time_min': float(np.min(recovery_times)) if recovery_times else None,
                'recovery_time_max': float(np.max(recovery_times)) if recovery_times else None,
                'pre_fault_velocity_mean': float(np.mean(pre_fault_velocities)) if pre_fault_velocities else None,
                'post_fault_velocity_mean': float(np.mean(post_fault_velocities)) if post_fault_velocities else None
            },
            'rollouts': rollout_results
        }

        env.close()
        return summary

    def evaluate_model(self, model_key):
        """Evaluate model across all joint failures"""
        config = self.models[model_key]

        print(f"\n{'='*60}")
        print(f"Evaluating: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}")

        joint_results = []

        for joint_name in tqdm(self.joints_to_test, desc=f"{config['name']}"):
            result = self.evaluate_model_with_failure(model_key, joint_name)
            joint_results.append(result)

            # Print quick summary (with recovery metrics)
            recovery_info = result['recovery']
            recovery_time_str = (f"{recovery_info['recovery_time_mean']:.2f}s"
                               if recovery_info['recovery_time_mean'] is not None else "N/A")
            print(f"  {joint_name:>10}: "
                  f"Distance={result['distance']['mean']:.3f}m, "
                  f"Success={result['success_rate']*100:.1f}%, "
                  f"Recovery={recovery_info['recovery_rate']*100:.1f}% ({recovery_time_str})")

        return {
            'model_key': model_key,
            'model_name': config['name'],
            'joint_results': joint_results
        }

    def run_experiment(self):
        """Run joint failure evaluation for all 4 models"""
        print("\n" + "="*60)
        print("EXPERIMENT 3: JOINT FAILURE ROBUSTNESS EVALUATION")
        print("="*60)
        print("Testing all 4 models with individual joint failures")
        print(f"- Joints tested: {self.joints_to_test}")
        print(f"- Delayed locking: {self.delay_steps} steps (2 seconds)")
        print(f"- Episode length: {self.episode_length} steps (20 seconds)")
        print(f"- Rollouts per joint: {self.num_rollouts}")
        print(f"- Total rollouts: {len(self.joints_to_test) * self.num_rollouts * 4} episodes")
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
        output_dir = 'evaluations/experiment_3_joint_failures/data'
        os.makedirs(output_dir, exist_ok=True)

        output_path = f"{output_dir}/joint_failure_results_{timestamp}.json"

        # Convert numpy types to native Python types for JSON serialization
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

        with open(output_path, 'w') as f:
            json.dump(results_clean, f, indent=2)

        print(f"\n Results saved to: {output_path}")

    def print_comparison(self):
        """Print comparison table across all models and joints"""
        print("\n" + "="*100)
        print("JOINT FAILURE ROBUSTNESS COMPARISON")
        print("="*100)

        # Print header
        print(f"{'Joint':<15}", end='')
        for model_key in ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo']:
            name = self.models[model_key]['name']
            print(f"{name:<25}", end='')
        print()
        print("-"*100)

        # Print results for each joint
        for i, joint_name in enumerate(self.joints_to_test):
            print(f"{joint_name:<15}", end='')

            for model_key in ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo']:
                result = self.results[model_key]['joint_results'][i]
                dist_mean = result['distance']['mean']
                success_rate = result['success_rate'] * 100
                print(f"{dist_mean:>6.2f}m ({success_rate:>4.1f}%)  ", end='')

            print()

        print("="*100)

        # Calculate average performance across all joints
        print("\nAVERAGE PERFORMANCE ACROSS ALL JOINT FAILURES:")
        print("-"*100)

        for model_key in ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo']:
            all_distances = [r['distance']['mean'] for r in self.results[model_key]['joint_results']]
            all_success = [r['success_rate'] for r in self.results[model_key]['joint_results']]

            avg_dist = np.mean(all_distances)
            avg_success = np.mean(all_success) * 100

            print(f"{self.models[model_key]['name']:<25}: "
                  f"Avg Distance={avg_dist:.3f}m, "
                  f"Avg Success={avg_success:.1f}%")

        # Find best performer
        best_model = max(
            ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo'],
            key=lambda k: np.mean([r['distance']['mean'] for r in self.results[k]['joint_results']])
        )

        print(f"\n Best Overall Joint Failure Robustness: {self.models[best_model]['name']}")

        # Check if DR (M3) dominates
        m3_avg = np.mean([r['distance']['mean'] for r in self.results['M3_dr']['joint_results']])
        m1_avg = np.mean([r['distance']['mean'] for r in self.results['M1_baseline']['joint_results']])

        if m3_avg > m1_avg:
            improvement = ((m3_avg - m1_avg) / m1_avg) * 100
            print(f"\n🎯 DR (M3) shows {improvement:.1f}% better average performance across joint failures vs Baseline (M1)")

        print("\n" + "="*100)

if __name__ == "__main__":
    evaluator = JointFailureEvaluator()
    evaluator.run_experiment()
