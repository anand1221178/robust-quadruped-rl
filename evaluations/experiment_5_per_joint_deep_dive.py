#!/usr/bin/env python3
"""
EXPERIMENT 5: PER-JOINT DEEP DIVE ANALYSIS
Comprehensive individual joint failure analysis for each model
Tests each model-joint combination with extended metrics and statistical analysis
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
from collections import defaultdict

class PerJointDeepDiveEvaluator:
    """
    Deep-dive analysis: Tests each model against each joint failure individually
    with comprehensive metrics and statistical analysis
    """

    def __init__(self):
        # Episode parameters
        self.episode_length = 1200  # 20 seconds at 60fps
        self.num_rollouts = 150  # More rollouts for statistical significance

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

        # Joint anatomical information
        self.joint_anatomy = {
            'hip_1': {'leg': 'front-left', 'type': 'hip', 'camera_facing': False},
            'ankle_1': {'leg': 'front-left', 'type': 'ankle', 'camera_facing': False},
            'hip_2': {'leg': 'front-right', 'type': 'hip', 'camera_facing': True},
            'ankle_2': {'leg': 'front-right', 'type': 'ankle', 'camera_facing': True},
            'hip_3': {'leg': 'rear-left', 'type': 'hip', 'camera_facing': False},
            'ankle_3': {'leg': 'rear-left', 'type': 'ankle', 'camera_facing': False},
            'hip_4': {'leg': 'rear-right', 'type': 'hip', 'camera_facing': True},
            'ankle_4': {'leg': 'rear-right', 'type': 'ankle', 'camera_facing': True},
        }

        # Success/failure thresholds
        self.success_threshold = 1.5  # meters
        self.failure_height_threshold = 0.2

        # Model configurations
        self.models = {
            'M1_baseline': {
                'name': 'PPO Baseline',
                'path': '../done/ppo_baseline_ueqbjf2x/best_model/best_model',
                'vec_path': '../done/ppo_baseline_ueqbjf2x/vec_normalize.pkl',
                'description': 'PPO only (no robustness training)',
                'baseline_distance': 11.20  # From Experiment 1
            },
            'M2_sr2l': {
                'name': 'PPO + SR2L',
                'path': '../done/ppo_sr2l_forward_m7gtjtpa/final_model',
                'vec_path': '../done/ppo_sr2l_forward_m7gtjtpa/vec_normalize.pkl',
                'description': 'PPO + SR2L (sensor noise specialist)',
                'baseline_distance': 8.91  # From Experiment 1
            },
            'M3_dr': {
                'name': 'PPO + DR (V7.7E)',
                'path': '../done/v7_7e_ultra_speed_jtfwl2qf/final_model',
                'vec_path': '../done/v7_7e_ultra_speed_jtfwl2qf/vec_normalize.pkl',
                'description': 'PPO + DR (JOINT FAILURE SPECIALIST)',
                'baseline_distance': 7.90  # From Experiment 1
            },
            'M4_combo': {
                'name': 'Ultimate Combo',
                'path': '../done/ultimate_robustness_combo_ju7lfsk2/final_model',
                'vec_path': '../done/ultimate_robustness_combo_ju7lfsk2/vec_normalize.pkl',
                'description': 'PPO + SR2L + DR (full robustness pipeline)',
                'baseline_distance': 7.86  # From Experiment 1
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

    def evaluate_model_joint_combination(self, model_key, failed_joint):
        """
        Run comprehensive evaluation for specific model-joint combination
        Returns detailed statistics and trajectory information
        """
        model, env = self.load_model(model_key)

        rollout_results = []
        trajectory_stats = {
            'velocity_profiles': [],
            'fall_times': [],
            'recovery_attempts': 0,
            'early_failures': 0  # Failures before delay period ends
        }

        for rollout in range(self.num_rollouts):
            obs = env.reset()
            done = False
            step_count = 0

            # Track detailed metrics
            positions = []
            rewards = []
            velocities = []
            fell = False
            fall_step = None

            # Get initial position
            try:
                unwrapped = env.envs[0].unwrapped
                start_x = unwrapped.data.qpos[0]
            except:
                start_x = 0.0

            prev_x = start_x

            while not done and step_count < self.episode_length:
                # Get action from model
                action, _ = model.predict(obs, deterministic=True)

                # Apply joint failure (with delayed locking)
                modified_action = self.apply_joint_failure(action, [failed_joint], step_count)

                # Track position BEFORE step (in case episode ends and resets)
                try:
                    current_x = unwrapped.data.qpos[0]
                    positions.append(current_x)

                    # Calculate instantaneous velocity
                    velocity = (current_x - prev_x) * 60.0  # Convert to m/s (60fps)
                    velocities.append(velocity)
                    prev_x = current_x
                except:
                    positions.append(0.0)
                    velocities.append(0.0)

                # Step environment
                obs, reward, done, info = env.step(modified_action)
                rewards.append(reward[0])

                # Check for fall
                if not fell and self.check_robot_fallen(env):
                    fell = True
                    fall_step = step_count

                    # Check if fell before delay ended (very early failure)
                    if step_count < self.delay_steps:
                        trajectory_stats['early_failures'] += 1

                step_count += 1

            # Calculate metrics
            final_position = positions[-1] if positions else start_x
            total_distance = abs(final_position - start_x)
            total_reward = sum(rewards)
            success = total_distance >= self.success_threshold

            # Calculate velocity statistics
            avg_velocity = np.mean(velocities) if velocities else 0.0
            max_velocity = np.max(velocities) if velocities else 0.0
            min_velocity = np.min(velocities) if velocities else 0.0

            # Check for recovery attempts (negative velocities after failure)
            if fell and fall_step is not None:
                post_failure_velocities = velocities[fall_step:]
                if any(v < -0.05 for v in post_failure_velocities):  # Backward movement
                    trajectory_stats['recovery_attempts'] += 1

            # Store velocity profile for analysis
            trajectory_stats['velocity_profiles'].append(velocities)
            if fell:
                trajectory_stats['fall_times'].append(fall_step)

            rollout_results.append({
                'rollout_id': rollout,
                'distance': float(total_distance),
                'reward': float(total_reward),
                'success': bool(success),
                'failure': bool(fell),
                'steps': int(step_count),
                'fall_step': int(fall_step) if fall_step is not None else None,
                'avg_velocity': float(avg_velocity),
                'max_velocity': float(max_velocity),
                'min_velocity': float(min_velocity)
            })

        # Calculate aggregate statistics
        distances = [r['distance'] for r in rollout_results]
        rewards = [r['reward'] for r in rollout_results]
        velocities_avg = [r['avg_velocity'] for r in rollout_results]
        success_count = sum(1 for r in rollout_results if r['success'])
        failure_count = sum(1 for r in rollout_results if r['failure'])

        # Calculate retention percentage vs baseline
        baseline_distance = self.models[model_key]['baseline_distance']
        mean_distance = np.mean(distances)
        retention_percentage = (mean_distance / baseline_distance) * 100.0

        # Statistical measures
        summary = {
            'model_key': model_key,
            'model_name': self.models[model_key]['name'],
            'failed_joint': failed_joint,
            'joint_anatomy': self.joint_anatomy[failed_joint],
            'num_rollouts': self.num_rollouts,

            # Distance statistics
            'distance': {
                'mean': float(mean_distance),
                'std': float(np.std(distances)),
                'min': float(np.min(distances)),
                'max': float(np.max(distances)),
                'median': float(np.median(distances)),
                'q25': float(np.percentile(distances, 25)),
                'q75': float(np.percentile(distances, 75)),
            },

            # Reward statistics
            'reward': {
                'mean': float(np.mean(rewards)),
                'std': float(np.std(rewards)),
                'median': float(np.median(rewards))
            },

            # Velocity statistics
            'velocity': {
                'mean': float(np.mean(velocities_avg)),
                'std': float(np.std(velocities_avg)),
                'max_observed': float(np.max([r['max_velocity'] for r in rollout_results])),
                'min_observed': float(np.min([r['min_velocity'] for r in rollout_results]))
            },

            # Performance metrics
            'success_rate': float(success_count / self.num_rollouts),
            'failure_rate': float(failure_count / self.num_rollouts),
            'baseline_distance': float(baseline_distance),
            'retention_percentage': float(retention_percentage),

            # Trajectory analysis
            'trajectory_stats': {
                'early_failure_rate': float(trajectory_stats['early_failures'] / self.num_rollouts),
                'recovery_attempt_rate': float(trajectory_stats['recovery_attempts'] / self.num_rollouts),
                'avg_fall_time_steps': float(np.mean(trajectory_stats['fall_times'])) if trajectory_stats['fall_times'] else None,
                'median_fall_time_steps': float(np.median(trajectory_stats['fall_times'])) if trajectory_stats['fall_times'] else None
            },

            # Raw rollout data
            'rollouts': rollout_results
        }

        env.close()
        return summary

    def evaluate_all_combinations(self):
        """
        Evaluate all model-joint combinations
        Total: 4 models × 8 joints = 32 combinations
        """
        print("\n" + "="*80)
        print("EXPERIMENT 5: PER-JOINT DEEP DIVE ANALYSIS")
        print("="*80)
        print("Comprehensive evaluation of each model-joint failure combination")
        print(f"- Models: 4 (M1 Baseline, M2 SR2L, M3 DR, M4 Combo)")
        print(f"- Joints: {len(self.joints_to_test)}")
        print(f"- Rollouts per combination: {self.num_rollouts}")
        print(f"- Total episodes: {4 * len(self.joints_to_test) * self.num_rollouts}")
        print(f"- Estimated time: ~2.5 hours")
        print("="*80)

        # Store results by model
        all_results = defaultdict(list)

        total_combinations = len(self.models) * len(self.joints_to_test)
        completed = 0

        for model_key in ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo']:
            model_name = self.models[model_key]['name']

            print(f"\n{'='*80}")
            print(f"EVALUATING: {model_name}")
            print(f"{'='*80}")

            for joint_name in tqdm(self.joints_to_test, desc=f"{model_name}"):
                result = self.evaluate_model_joint_combination(model_key, joint_name)
                all_results[model_key].append(result)

                completed += 1

                # Print quick summary
                print(f"  [{completed}/{total_combinations}] {joint_name:>10}: "
                      f"Distance={result['distance']['mean']:.3f}m "
                      f"({result['retention_percentage']:.1f}% retention), "
                      f"Success={result['success_rate']*100:.1f}%, "
                      f"Failure={result['failure_rate']*100:.1f}%")

        self.results = dict(all_results)

        # Save results
        self.save_results()

        # Generate comprehensive analysis
        self.print_comprehensive_analysis()

    def save_results(self):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = 'evaluations/experiment_5_per_joint_deep_dive/data'
        os.makedirs(output_dir, exist_ok=True)

        output_path = f"{output_dir}/per_joint_deep_dive_results_{timestamp}.json"

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✅ Results saved to: {output_path}")

    def print_comprehensive_analysis(self):
        """Print comprehensive analysis across all combinations"""
        print("\n" + "="*100)
        print("COMPREHENSIVE PER-JOINT ANALYSIS")
        print("="*100)

        # ANALYSIS 1: Retention percentage matrix
        print("\n" + "="*100)
        print("RETENTION PERCENTAGE MATRIX (% of baseline performance)")
        print("="*100)
        print(f"{'Joint':<15}", end='')
        for model_key in ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo']:
            name = self.models[model_key]['name']
            print(f"{name:<25}", end='')
        print()
        print("-"*100)

        for i, joint_name in enumerate(self.joints_to_test):
            print(f"{joint_name:<15}", end='')

            for model_key in ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo']:
                result = self.results[model_key][i]
                retention = result['retention_percentage']

                # Color coding for terminal output
                if retention >= 50:
                    marker = "✓"
                elif retention >= 30:
                    marker = "~"
                else:
                    marker = "✗"

                print(f"{retention:>6.1f}% {marker}         ", end='')

            print()

        print("="*100)

        # ANALYSIS 2: Best and worst joints per model
        print("\n" + "="*100)
        print("BEST AND WORST JOINTS FOR EACH MODEL")
        print("="*100)

        for model_key in ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo']:
            model_name = self.models[model_key]['name']
            model_results = self.results[model_key]

            # Sort by retention percentage
            sorted_results = sorted(model_results, key=lambda x: x['retention_percentage'], reverse=True)

            best = sorted_results[0]
            worst = sorted_results[-1]

            print(f"\n{model_name}:")
            print(f"  BEST:  {best['failed_joint']:>10} → {best['retention_percentage']:>6.1f}% retention "
                  f"({best['distance']['mean']:.3f}m, Success: {best['success_rate']*100:.1f}%)")
            print(f"  WORST: {worst['failed_joint']:>10} → {worst['retention_percentage']:>6.1f}% retention "
                  f"({worst['distance']['mean']:.3f}m, Success: {worst['success_rate']*100:.1f}%)")

        # ANALYSIS 3: Anatomical patterns
        print("\n" + "="*100)
        print("ANATOMICAL PATTERN ANALYSIS")
        print("="*100)

        for model_key in ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo']:
            model_name = self.models[model_key]['name']
            model_results = self.results[model_key]

            # Group by joint type
            hip_retentions = [r['retention_percentage'] for r in model_results if r['joint_anatomy']['type'] == 'hip']
            ankle_retentions = [r['retention_percentage'] for r in model_results if r['joint_anatomy']['type'] == 'ankle']

            # Group by camera facing
            camera_facing_retentions = [r['retention_percentage'] for r in model_results if r['joint_anatomy']['camera_facing']]
            camera_away_retentions = [r['retention_percentage'] for r in model_results if not r['joint_anatomy']['camera_facing']]

            print(f"\n{model_name}:")
            print(f"  Hip failures:    {np.mean(hip_retentions):.1f}% avg retention")
            print(f"  Ankle failures:  {np.mean(ankle_retentions):.1f}% avg retention")
            print(f"  Camera-facing:   {np.mean(camera_facing_retentions):.1f}% avg retention")
            print(f"  Camera-away:     {np.mean(camera_away_retentions):.1f}% avg retention")

        # ANALYSIS 4: Model ranking by average retention
        print("\n" + "="*100)
        print("OVERALL MODEL RANKING (by average retention across all joints)")
        print("="*100)

        model_avg_retentions = []
        for model_key in ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo']:
            model_results = self.results[model_key]
            avg_retention = np.mean([r['retention_percentage'] for r in model_results])
            model_avg_retentions.append((model_key, avg_retention))

        # Sort by average retention
        model_avg_retentions.sort(key=lambda x: x[1], reverse=True)

        for rank, (model_key, avg_retention) in enumerate(model_avg_retentions, 1):
            model_name = self.models[model_key]['name']
            medal = ["🥇", "🥈", "🥉", "4th"][rank-1]
            print(f"{medal} {rank}. {model_name:<25}: {avg_retention:.1f}% average retention")

        # ANALYSIS 5: Statistical significance
        print("\n" + "="*100)
        print("KEY FINDINGS")
        print("="*100)

        # Find if M3 (DR) consistently beats others
        m3_results = self.results['M3_dr']
        m1_results = self.results['M1_baseline']

        m3_wins = sum(1 for i in range(len(self.joints_to_test))
                      if m3_results[i]['retention_percentage'] > m1_results[i]['retention_percentage'])

        print(f"\n✓ M3 (DR) outperforms M1 (Baseline) in {m3_wins}/{len(self.joints_to_test)} joint failures")

        # Check ankle_4 performance
        ankle_4_idx = self.joints_to_test.index('ankle_4')
        print(f"\n✓ Ankle_4 (hardest joint) retention rates:")
        for model_key in ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo']:
            model_name = self.models[model_key]['name']
            retention = self.results[model_key][ankle_4_idx]['retention_percentage']
            print(f"   {model_name:<25}: {retention:.1f}%")

        print("\n" + "="*100)

if __name__ == "__main__":
    evaluator = PerJointDeepDiveEvaluator()
    evaluator.evaluate_all_combinations()
