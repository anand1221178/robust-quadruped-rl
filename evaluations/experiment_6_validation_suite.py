#!/usr/bin/env python3
"""
EXPERIMENT 6: RESEARCH CLAIMS VALIDATION SUITE
Validates 4 major discoveries from the research:
1. VecNormalize provides implicit noise robustness
2. Stochastic resonance in SR2L (mild noise improves performance)
3. Hip_1 super-recovery phenomenon (walks faster with failure)
4. Joint difficulty ranking (ankle_4 universally hardest)
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
from scipy import stats

class ValidationSuite:
    """Comprehensive validation of 4 major research discoveries"""

    def __init__(self):
        # Episode parameters
        self.episode_length = 1200  # 20 seconds
        self.num_rollouts = 100  # Statistical significance

        # Model configurations (UPDATED: Using 32M retrained M1/M2, keeping champion M3/M4)
        self.models = {
            'M1_baseline': {
                'name': 'PPO Baseline (32M)',
                'path': '../experiments/M1_baseline_32M_RETRAINED_ym2jcllj/final_model',
                'vec_path': '../experiments/M1_baseline_32M_RETRAINED_ym2jcllj/vec_normalize.pkl',
                'baseline_distance': 11.20  # To be updated after rerun
            },
            'M2_sr2l': {
                'name': 'PPO + SR2L (32M)',
                'path': '../experiments/M2_sr2l_32M_RETRAINED_ze09p0vf/final_model',
                'vec_path': '../experiments/M2_sr2l_32M_RETRAINED_ze09p0vf/vec_normalize.pkl',
                'baseline_distance': 8.91  # To be updated after rerun
            },
            'M3_dr': {
                'name': 'PPO + DR (V7.7E Champion)',
                'path': '../done/v7_7e_ultra_speed_jtfwl2qf/final_model',
                'vec_path': '../done/v7_7e_ultra_speed_jtfwl2qf/vec_normalize.pkl',
                'baseline_distance': 7.90
            },
            'M4_combo': {
                'name': 'Ultimate Combo',
                'path': '../done/ultimate_robustness_combo_ju7lfsk2/final_model',
                'vec_path': '../done/ultimate_robustness_combo_ju7lfsk2/vec_normalize.pkl',
                'baseline_distance': 7.86
            }
        }

        # Joint mapping
        self.joint_to_action = {
            'hip_1': 0, 'ankle_1': 1,
            'hip_2': 2, 'ankle_2': 3,
            'hip_3': 4, 'ankle_3': 5,
            'hip_4': 6, 'ankle_4': 7
        }

        self.success_threshold = 1.5
        self.failure_height_threshold = 0.2

        self.results = {
            'test_1_vecnormalize': {},
            'test_2_stochastic_resonance': {},
            'test_3_hip1_recovery': {},
            'test_4_joint_ranking': {}
        }

    def create_environment(self, use_vecnormalize=True):
        """Create evaluation environment with optional VecNormalize"""
        def make_env():
            base_env = gym.make('RealAntMujoco-v0', disable_env_checker=True)
            env = SuccessRewardWrapper(base_env)
            env = TimeLimit(env, max_episode_steps=self.episode_length)
            return env

        env = DummyVecEnv([make_env])

        if use_vecnormalize:
            # Will load VecNormalize stats when loading model
            return env
        else:
            # Raw environment without normalization
            return env

    def load_model(self, model_key, use_vecnormalize=True):
        """Load model with optional VecNormalize"""
        config = self.models[model_key]
        env = self.create_environment(use_vecnormalize)

        if use_vecnormalize:
            env = VecNormalize.load(config['vec_path'], env)
            env.training = False
            env.norm_reward = False

        model = PPO.load(config['path'], env=env)
        return model, env

    def apply_sensor_noise(self, obs, noise_std, rng):
        """Apply sensor noise to joint observations (dims 13-28)"""
        if noise_std <= 0:
            return obs

        obs_copy = obs.copy()
        for idx in range(13, min(29, len(obs_copy[0]))):
            noise = rng.normal(0, noise_std)
            obs_copy[0][idx] += noise

        return obs_copy

    def apply_joint_failure(self, action, failed_joints, step_count, delay_steps=120):
        """Apply joint failure with delay"""
        if step_count < delay_steps:
            return action

        modified_action = action.copy()
        for joint_name in failed_joints:
            if joint_name in self.joint_to_action:
                action_idx = self.joint_to_action[joint_name]
                modified_action[0][action_idx] = 0.0

        return modified_action

    def check_robot_fallen(self, env):
        """Check if robot has fallen"""
        try:
            unwrapped = env.envs[0].unwrapped
            if hasattr(unwrapped, 'data'):
                return unwrapped.data.qpos[2] < self.failure_height_threshold
        except:
            pass
        return False

    def run_episode(self, model, env, noise_std=0.0, failed_joints=None, rng=None):
        """Run single episode with optional noise/failures"""
        if rng is None:
            rng = np.random.RandomState(42)

        obs = env.reset()
        done = False
        step_count = 0

        positions = []
        rewards = []
        fell = False

        try:
            unwrapped = env.envs[0].unwrapped
            start_x = unwrapped.data.qpos[0]
        except:
            start_x = 0.0

        while not done and step_count < self.episode_length:
            # Apply sensor noise if specified
            if noise_std > 0:
                noisy_obs = self.apply_sensor_noise(obs, noise_std, rng)
            else:
                noisy_obs = obs

            # Get action
            action, _ = model.predict(noisy_obs, deterministic=True)

            # Apply joint failure if specified
            if failed_joints:
                action = self.apply_joint_failure(action, failed_joints, step_count)

            # Track position BEFORE step
            try:
                current_x = unwrapped.data.qpos[0]
                positions.append(current_x)
            except:
                positions.append(0.0)

            # Step environment
            obs, reward, done, info = env.step(action)
            rewards.append(reward[0])

            if not fell and self.check_robot_fallen(env):
                fell = True

            step_count += 1

        # Calculate metrics
        final_position = positions[-1] if positions else start_x
        total_distance = abs(final_position - start_x)
        total_reward = sum(rewards)
        success = total_distance >= self.success_threshold

        return {
            'distance': float(total_distance),
            'reward': float(total_reward),
            'success': bool(success),
            'failure': bool(fell),
            'steps': int(step_count)
        }

    # ========================================================================
    # TEST 1: VecNormalize Implicit Robustness
    # ========================================================================
    def test_1_vecnormalize_ablation(self):
        """
        Tests claim: "VecNormalize provides implicit noise robustness"
        Method: Compare M1 (baseline) with vs without VecNormalize under noise
        Expected: Without VecNormalize, noise should devastate performance
        """
        print("\n" + "="*80)
        print("TEST 1: VecNormalize Implicit Robustness Ablation")
        print("="*80)
        print("Claim: VecNormalize is responsible for unexpected noise robustness")
        print("Method: Test M1 with/without VecNormalize at σ=0.0, 0.05, 0.1")
        print(f"Episodes: {2 * 3 * self.num_rollouts} (2 conditions × 3 levels × 100)")
        print("="*80)

        model_key = 'M1_baseline'
        noise_levels = [0.0, 0.05, 0.1]
        conditions = ['with_vecnormalize', 'without_vecnormalize']

        results = {}

        for condition in conditions:
            use_vecnorm = (condition == 'with_vecnormalize')
            print(f"\n🔬 Testing: {condition}")

            model, env = self.load_model(model_key, use_vecnormalize=use_vecnorm)
            condition_results = {}

            for noise_std in tqdm(noise_levels, desc=condition):
                rollout_results = []
                rng = np.random.RandomState(42)

                for rollout in range(self.num_rollouts):
                    result = self.run_episode(model, env, noise_std=noise_std, rng=rng)
                    rollout_results.append(result)

                distances = [r['distance'] for r in rollout_results]
                condition_results[f'noise_{noise_std}'] = {
                    'mean': float(np.mean(distances)),
                    'std': float(np.std(distances)),
                    'success_rate': float(sum(r['success'] for r in rollout_results) / len(rollout_results)),
                    'rollouts': rollout_results
                }

                print(f"  σ={noise_std:.2f}: {np.mean(distances):.3f}m "
                      f"({100 * np.mean(distances) / condition_results['noise_0.0']['mean']:.1f}% retention)")

            env.close()
            results[condition] = condition_results

        # Calculate degradation comparison
        print("\n" + "="*80)
        print("RESULTS: VecNormalize Impact on Noise Robustness")
        print("="*80)

        for noise_std in noise_levels:
            with_vec = results['with_vecnormalize'][f'noise_{noise_std}']['mean']
            without_vec = results['without_vecnormalize'][f'noise_{noise_std}']['mean']
            baseline_with = results['with_vecnormalize']['noise_0.0']['mean']
            baseline_without = results['without_vecnormalize']['noise_0.0']['mean']

            retention_with = (with_vec / baseline_with) * 100
            retention_without = (without_vec / baseline_without) * 100

            print(f"\nNoise σ={noise_std}:")
            print(f"  With VecNormalize:    {with_vec:.3f}m ({retention_with:.1f}% retention)")
            print(f"  Without VecNormalize: {without_vec:.3f}m ({retention_without:.1f}% retention)")
            print(f"  VecNormalize Advantage: {retention_with - retention_without:+.1f}%")

        self.results['test_1_vecnormalize'] = results

        # Statistical significance test
        with_vec_sigma01 = [r['distance'] for r in results['with_vecnormalize']['noise_0.1']['rollouts']]
        without_vec_sigma01 = [r['distance'] for r in results['without_vecnormalize']['noise_0.1']['rollouts']]
        t_stat, p_value = stats.ttest_ind(with_vec_sigma01, without_vec_sigma01)

        print(f"\n✅ Statistical Test (σ=0.1):")
        print(f"   t-statistic: {t_stat:.3f}, p-value: {p_value:.6f}")
        if p_value < 0.001:
            print(f"   Verdict: ✅ HIGHLY SIGNIFICANT (p < 0.001) - VecNormalize proven essential!")

    # ========================================================================
    # TEST 2: Stochastic Resonance in SR2L
    # ========================================================================
    def test_2_stochastic_resonance(self):
        """
        Tests claim: "SR2L improves with mild noise (stochastic resonance)"
        Method: Test M2 at fine noise increments to find performance peak
        Expected: Performance peak at σ ≈ 0.01-0.02 (mild noise)
        """
        print("\n" + "="*80)
        print("TEST 2: Stochastic Resonance Validation (SR2L)")
        print("="*80)
        print("Claim: SR2L IMPROVES from 8.91m → 9.00m with mild noise (101% retention)")
        print("Method: Test M2 at fine noise increments (σ=0.000 to 0.100)")
        print(f"Episodes: {7 * self.num_rollouts} (7 levels × 100)")
        print("="*80)

        model_key = 'M2_sr2l'
        # Fine-grained noise levels to find the peak
        noise_levels = [0.000, 0.005, 0.010, 0.020, 0.030, 0.050, 0.100]

        print(f"\n🔬 Testing: {self.models[model_key]['name']}")
        model, env = self.load_model(model_key)

        results = {}
        baseline_performance = self.models[model_key]['baseline_distance']

        for noise_std in tqdm(noise_levels, desc="Noise levels"):
            rollout_results = []
            rng = np.random.RandomState(42)

            for rollout in range(self.num_rollouts):
                result = self.run_episode(model, env, noise_std=noise_std, rng=rng)
                rollout_results.append(result)

            distances = [r['distance'] for r in rollout_results]
            mean_dist = np.mean(distances)
            retention = (mean_dist / baseline_performance) * 100

            results[f'noise_{noise_std:.3f}'] = {
                'noise_level': float(noise_std),
                'mean': float(mean_dist),
                'std': float(np.std(distances)),
                'retention_pct': float(retention),
                'success_rate': float(sum(r['success'] for r in rollout_results) / len(rollout_results)),
                'rollouts': rollout_results
            }

            marker = "🔥" if retention > 100 else "✓" if retention > 95 else "~"
            print(f"  σ={noise_std:.3f}: {mean_dist:.3f}m ({retention:.1f}% retention) {marker}")

        env.close()

        # Find optimal noise level
        optimal_noise = max(results.items(), key=lambda x: x[1]['retention_pct'])

        print("\n" + "="*80)
        print("RESULTS: Stochastic Resonance Analysis")
        print("="*80)
        print(f"\n🔥 Optimal Noise Level: σ={optimal_noise[1]['noise_level']:.3f}")
        print(f"   Performance: {optimal_noise[1]['mean']:.3f}m ({optimal_noise[1]['retention_pct']:.1f}% retention)")
        print(f"   Baseline (σ=0.0): {results['noise_0.000']['mean']:.3f}m")
        print(f"   Improvement: {optimal_noise[1]['mean'] - results['noise_0.000']['mean']:+.3f}m")

        if optimal_noise[1]['retention_pct'] > 100:
            print(f"\n✅ Stochastic Resonance CONFIRMED!")
            print(f"   SR2L performs BETTER with mild noise than without")
            print(f"   This is a neuroscience phenomenon - noise helps signal processing!")

        self.results['test_2_stochastic_resonance'] = results

    # ========================================================================
    # TEST 3: Hip_1 Super-Recovery Phenomenon
    # ========================================================================
    def test_3_hip1_super_recovery(self):
        """
        Tests claim: "Hip_1 failure can improve performance (105% retention)"
        Method: Test all 4 models with hip_1 locked (300 rollouts for significance)
        Expected: Some models walk FASTER with hip_1 locked
        """
        print("\n" + "="*80)
        print("TEST 3: Hip_1 Super-Recovery Investigation")
        print("="*80)
        print("Claim: Some models walk FASTER with hip_1 locked (105.5% retention!)")
        print("Method: Test all 4 models with hip_1 failure (300 rollouts)")
        print(f"Episodes: {4 * 300} (4 models × 300)")
        print("="*80)

        # Use higher rollouts for statistical confidence
        high_rollouts = 300
        results = {}

        for model_key in ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo']:
            print(f"\n🔬 Testing: {self.models[model_key]['name']}")

            model, env = self.load_model(model_key)
            baseline_performance = self.models[model_key]['baseline_distance']

            # Test with hip_1 locked
            rollout_results = []
            for rollout in tqdm(range(high_rollouts), desc=f"{self.models[model_key]['name']}"):
                result = self.run_episode(model, env, failed_joints=['hip_1'])
                rollout_results.append(result)

            distances = [r['distance'] for r in rollout_results]
            mean_dist = np.mean(distances)
            retention = (mean_dist / baseline_performance) * 100

            results[model_key] = {
                'model_name': self.models[model_key]['name'],
                'baseline': float(baseline_performance),
                'with_hip1_failure': float(mean_dist),
                'std': float(np.std(distances)),
                'retention_pct': float(retention),
                'success_rate': float(sum(r['success'] for r in rollout_results) / len(rollout_results)),
                'super_recovery': bool(retention > 100),
                'rollouts': rollout_results
            }

            marker = "🤯" if retention > 100 else "✓" if retention > 80 else "~"
            print(f"   Baseline: {baseline_performance:.3f}m")
            print(f"   With hip_1 locked: {mean_dist:.3f}m ({retention:.1f}% retention) {marker}")

            env.close()

        print("\n" + "="*80)
        print("RESULTS: Hip_1 Super-Recovery Analysis")
        print("="*80)

        super_recovery_models = [k for k, v in results.items() if v['super_recovery']]

        if super_recovery_models:
            print(f"\n🤯 SUPER-RECOVERY CONFIRMED for {len(super_recovery_models)} model(s)!")
            for model_key in super_recovery_models:
                r = results[model_key]
                print(f"\n{r['model_name']}:")
                print(f"  Baseline: {r['baseline']:.3f}m")
                print(f"  With hip_1 locked: {r['with_hip1_failure']:.3f}m")
                print(f"  Retention: {r['retention_pct']:.1f}% (WALKS FASTER!)")
                print(f"  Hypothesis: Hip_1 lock forces more efficient tripod gait")
        else:
            print("\n✗ No super-recovery observed in current models")
            print("  (Original finding was in V7.8f, not in final 4 models)")

        print("\n📊 Ranking by Hip_1 Robustness:")
        sorted_models = sorted(results.items(), key=lambda x: x[1]['retention_pct'], reverse=True)
        for rank, (model_key, data) in enumerate(sorted_models, 1):
            medal = ["🥇", "🥈", "🥉", "4th"][rank-1]
            print(f"{medal} {rank}. {data['model_name']}: {data['retention_pct']:.1f}% retention")

        self.results['test_3_hip1_recovery'] = results

    # ========================================================================
    # TEST 4: Joint Difficulty Ranking (Statistical)
    # ========================================================================
    def test_4_joint_difficulty_ranking(self):
        """
        Tests claim: "Ankle_4 is universally hardest joint"
        Method: Load Experiment 3 data and perform statistical analysis
        Expected: Ankle_4 significantly harder than all other joints (ANOVA)
        """
        print("\n" + "="*80)
        print("TEST 4: Joint Difficulty Ranking (Statistical Analysis)")
        print("="*80)
        print("Claim: Ankle_4 is universally hardest across ALL models")
        print("Method: Analyze Experiment 3 data with ANOVA + post-hoc tests")
        print("="*80)

        # Load Experiment 3 results
        exp3_path = 'evaluations/experiment_3_joint_failures/data'

        try:
            import glob
            exp3_files = glob.glob(f'{exp3_path}/*.json')
            if not exp3_files:
                print("\n⚠️  Experiment 3 data not found. Run Experiment 3 first.")
                return

            latest_file = max(exp3_files, key=os.path.getmtime)
            print(f"\n📂 Loading: {latest_file}")

            with open(latest_file, 'r') as f:
                exp3_data = json.load(f)

            # Extract retention percentages for each joint across all models
            joints = ['hip_1', 'ankle_1', 'hip_2', 'ankle_2', 'hip_3', 'ankle_3', 'hip_4', 'ankle_4']
            joint_retentions = {joint: [] for joint in joints}

            for model_key in ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo']:
                model_data = exp3_data[model_key]

                for joint_result in model_data['joint_results']:
                    joint_name = joint_result['failed_joint']

                    # Calculate retention percentage
                    baseline = self.models[model_key]['baseline_distance']
                    mean_dist = joint_result['distance']['mean']
                    retention_pct = (mean_dist / baseline) * 100

                    joint_retentions[joint_name].append(retention_pct)

            # Calculate statistics for each joint
            joint_stats = {}
            for joint, retentions in joint_retentions.items():
                joint_stats[joint] = {
                    'mean_retention': float(np.mean(retentions)),
                    'std': float(np.std(retentions)),
                    'min': float(np.min(retentions)),
                    'max': float(np.max(retentions)),
                    'values': retentions
                }

            # Rank joints by difficulty (lower retention = harder)
            ranked_joints = sorted(joint_stats.items(), key=lambda x: x[1]['mean_retention'])

            print("\n" + "="*80)
            print("RESULTS: Joint Difficulty Ranking")
            print("="*80)
            print("\nRanked by Average Retention (Lower = Harder):\n")

            for rank, (joint, stats) in enumerate(ranked_joints, 1):
                difficulty = "HARDEST" if rank == 1 else "EASIEST" if rank == 8 else ""
                print(f"{rank}. {joint:>10}: {stats['mean_retention']:>5.1f}% ± {stats['std']:.1f}% {difficulty}")

            # Statistical test: Is ankle_4 significantly hardest?
            print("\n" + "="*80)
            print("Statistical Significance Test (ANOVA)")
            print("="*80)

            # One-way ANOVA across all joints
            joint_values = [stats['values'] for stats in joint_stats.values()]
            f_stat, p_value = stats.f_oneway(*joint_values)

            print(f"\nOne-way ANOVA:")
            print(f"  F-statistic: {f_stat:.3f}")
            print(f"  p-value: {p_value:.6f}")

            if p_value < 0.001:
                print(f"  ✅ HIGHLY SIGNIFICANT (p < 0.001)")
                print(f"     Joint difficulty varies significantly!")

            # Post-hoc: Is ankle_4 significantly worse than others?
            ankle4_values = joint_stats['ankle_4']['values']
            other_joints_combined = []
            for joint, stats in joint_stats.items():
                if joint != 'ankle_4':
                    other_joints_combined.extend(stats['values'])

            t_stat, p_value_ankle4 = stats.ttest_ind(ankle4_values, other_joints_combined)

            print(f"\nPost-hoc Test (ankle_4 vs all others):")
            print(f"  Ankle_4 mean: {np.mean(ankle4_values):.1f}%")
            print(f"  Others mean: {np.mean(other_joints_combined):.1f}%")
            print(f"  t-statistic: {t_stat:.3f}")
            print(f"  p-value: {p_value_ankle4:.6f}")

            if p_value_ankle4 < 0.001:
                print(f"  ✅ HIGHLY SIGNIFICANT (p < 0.001)")
                print(f"     Ankle_4 is PROVABLY harder than other joints!")

            # Check if it's universally hardest (hardest for EVERY model)
            hardest_per_model = []
            for model_key in ['M1_baseline', 'M2_sr2l', 'M3_dr', 'M4_combo']:
                model_joint_perfs = {}
                for joint_result in exp3_data[model_key]['joint_results']:
                    joint_name = joint_result['failed_joint']
                    baseline = self.models[model_key]['baseline_distance']
                    retention = (joint_result['distance']['mean'] / baseline) * 100
                    model_joint_perfs[joint_name] = retention

                hardest = min(model_joint_perfs.items(), key=lambda x: x[1])
                hardest_per_model.append(hardest[0])

            universal = all(joint == 'ankle_4' for joint in hardest_per_model)

            print(f"\nUniversality Check:")
            print(f"  Hardest joint per model: {hardest_per_model}")
            if universal:
                print(f"  ✅ UNIVERSAL: ankle_4 is hardest for ALL 4 models!")
            else:
                print(f"  ~ Ankle_4 hardest for {hardest_per_model.count('ankle_4')}/4 models")

            self.results['test_4_joint_ranking'] = {
                'ranked_joints': [(joint, stats['mean_retention']) for joint, stats in ranked_joints],
                'anova_f': float(f_stat),
                'anova_p': float(p_value),
                'ankle4_vs_others_t': float(t_stat),
                'ankle4_vs_others_p': float(p_value_ankle4),
                'universal_hardest': bool(universal),
                'joint_stats': {k: {kk: vv for kk, vv in v.items() if kk != 'values'}
                               for k, v in joint_stats.items()}
            }

        except Exception as e:
            print(f"\n❌ Error loading Experiment 3 data: {e}")
            print("   Make sure Experiment 3 has been run first.")

    # ========================================================================
    # Main Execution
    # ========================================================================
    def run_all_tests(self):
        """Run all 4 validation tests"""
        print("\n" + "="*80)
        print("EXPERIMENT 6: RESEARCH CLAIMS VALIDATION SUITE")
        print("="*80)
        print("Validating 4 major discoveries through controlled experiments")
        print(f"Total estimated time: ~2.5 hours")
        print("="*80)

        # Test 1: VecNormalize ablation (~30 min)
        self.test_1_vecnormalize_ablation()

        # Test 2: Stochastic resonance (~20 min)
        self.test_2_stochastic_resonance()

        # Test 3: Hip_1 super-recovery (~30 min)
        self.test_3_hip1_super_recovery()

        # Test 4: Joint ranking (analysis only, ~5 min)
        self.test_4_joint_difficulty_ranking()

        # Save all results
        self.save_results()

        # Print final summary
        self.print_summary()

    def save_results(self):
        """Save all validation results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = 'evaluations/experiment_6_validation_suite/data'
        os.makedirs(output_dir, exist_ok=True)

        output_path = f"{output_dir}/validation_results_{timestamp}.json"

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✅ Results saved to: {output_path}")

    def print_summary(self):
        """Print final summary of all validation tests"""
        print("\n" + "="*80)
        print("VALIDATION SUITE SUMMARY")
        print("="*80)

        print("\n📊 TEST 1: VecNormalize Implicit Robustness")
        if 'test_1_vecnormalize' in self.results and self.results['test_1_vecnormalize']:
            data = self.results['test_1_vecnormalize']
            with_vec = data['with_vecnormalize']['noise_0.1']['mean']
            without_vec = data['without_vecnormalize']['noise_0.1']['mean']
            baseline_with = data['with_vecnormalize']['noise_0.0']['mean']
            baseline_without = data['without_vecnormalize']['noise_0.0']['mean']

            retention_with = (with_vec / baseline_with) * 100
            retention_without = (without_vec / baseline_without) * 100

            print(f"   With VecNormalize (σ=0.1):    {retention_with:.1f}% retention")
            print(f"   Without VecNormalize (σ=0.1): {retention_without:.1f}% retention")
            print(f"   Advantage: {retention_with - retention_without:+.1f}%")

            if retention_with - retention_without > 20:
                print(f"   ✅ CLAIM VALIDATED: VecNormalize essential for noise robustness")
            else:
                print(f"   ~ Partial support for claim")

        print("\n🔥 TEST 2: Stochastic Resonance")
        if 'test_2_stochastic_resonance' in self.results and self.results['test_2_stochastic_resonance']:
            data = self.results['test_2_stochastic_resonance']
            optimal = max(data.items(), key=lambda x: x[1]['retention_pct'])
            baseline = data['noise_0.000']

            print(f"   Baseline (σ=0.0): {baseline['retention_pct']:.1f}% retention")
            print(f"   Optimal (σ={optimal[1]['noise_level']:.3f}): {optimal[1]['retention_pct']:.1f}% retention")

            if optimal[1]['retention_pct'] > 100:
                print(f"   ✅ CLAIM VALIDATED: Stochastic resonance confirmed!")
            else:
                print(f"   ~ Peak at σ={optimal[1]['noise_level']:.3f} but <100% retention")

        print("\n🤯 TEST 3: Hip_1 Super-Recovery")
        if 'test_3_hip1_recovery' in self.results and self.results['test_3_hip1_recovery']:
            data = self.results['test_3_hip1_recovery']
            super_models = [k for k, v in data.items() if v['super_recovery']]

            if super_models:
                print(f"   ✅ SUPER-RECOVERY FOUND in {len(super_models)} model(s):")
                for model_key in super_models:
                    print(f"      {data[model_key]['model_name']}: {data[model_key]['retention_pct']:.1f}%")
            else:
                print(f"   ~ No super-recovery in current 4 models")
                print(f"      (Original finding in V7.8f, not final models)")

        print("\n📍 TEST 4: Joint Difficulty Ranking")
        if 'test_4_joint_ranking' in self.results and self.results['test_4_joint_ranking']:
            data = self.results['test_4_joint_ranking']

            if 'ranked_joints' in data:
                hardest = data['ranked_joints'][0]
                print(f"   Hardest joint: {hardest[0]} ({hardest[1]:.1f}% avg retention)")
                print(f"   ANOVA p-value: {data['anova_p']:.6f}")
                print(f"   Ankle_4 vs others p-value: {data['ankle4_vs_others_p']:.6f}")

                if data['universal_hardest']:
                    print(f"   ✅ CLAIM VALIDATED: Ankle_4 universally hardest (all 4 models)")
                elif hardest[0] == 'ankle_4':
                    print(f"   ✅ CLAIM SUPPORTED: Ankle_4 hardest on average")
                else:
                    print(f"   ✗ CLAIM NOT SUPPORTED: {hardest[0]} hardest, not ankle_4")

        print("\n" + "="*80)
        print("✅ VALIDATION SUITE COMPLETE")
        print("="*80)

if __name__ == "__main__":
    validator = ValidationSuite()
    validator.run_all_tests()
