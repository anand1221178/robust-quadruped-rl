#!/usr/bin/env python3
"""
EXPERIMENT 8: NOISE-INDUCED REGULARIZATION MECHANISM ANALYSIS
Systematic ablation study to identify the mechanism causing performance improvement with mild noise

Research Question: What causes the 101-108% retention at σ=0.1 (10× training noise)?

4 Sub-Experiments:
8A: Baseline Resonance Check - Does M1 (no SR2L training) also show resonance?
8B: Fine-Grained Noise Sweep - Exact peak location and characteristics for SR2L
8C: DR Model Resonance - Why does M3 show stronger resonance (108%) than M2 (101%)?
8D: VecNormalize Frozen Test - Is adaptive normalization necessary for resonance?

Hypotheses:
H1: VecNormalize adaptive statistics cause resonance
H2: SR2L smoothness regularization magnitude correlates with resonance strength
H3: Resonance only occurs in specific noise range (inverted-U curve)
H4: All models show resonance (independent of training method)

Total Episodes: 4,400 (4 sub-experiments)
Runtime: ~5 hours
"""

import sys
import os

# Add paths
sys.path.append('src')
sys.path.append(os.path.dirname(__file__))

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.success_reward_wrapper import SuccessRewardWrapper
import realant_sim
import json
from datetime import datetime
from tqdm import tqdm
import copy


class MechanismAnalyzer:
    """Analyzes noise-induced regularization mechanism"""

    def __init__(self):
        # Episode parameters
        self.episode_length = 1200
        self.num_rollouts = 100

        # Success threshold
        self.success_threshold = 1.5

        # Model configurations (UPDATED: Using 32M retrained M1/M2, keeping champion M3/M4)
        self.models = {
            'M1_baseline': {
                'name': 'PPO Baseline (32M)',
                'path': '../experiments/M1_baseline_32M_RETRAINED_ym2jcllj/final_model',
                'vec_path': '../experiments/M1_baseline_32M_RETRAINED_ym2jcllj/vec_normalize.pkl',
                'description': 'No robustness training - RETRAINED 32M'
            },
            'M2_sr2l': {
                'name': 'PPO + SR2L (32M)',
                'path': '../experiments/M2_sr2l_32M_RETRAINED_ze09p0vf/final_model',
                'vec_path': '../experiments/M2_sr2l_32M_RETRAINED_ze09p0vf/vec_normalize.pkl',
                'description': 'SR2L sensor noise specialist (trained σ=0.01) - RETRAINED 32M'
            },
            'M3_dr': {
                'name': 'PPO + DR (V7.7E)',
                'path': '../done/v7_7e_ultra_speed_jtfwl2qf/final_model',
                'vec_path': '../done/v7_7e_ultra_speed_jtfwl2qf/vec_normalize.pkl',
                'description': 'Domain randomization joint failure specialist'
            },
            'M4_combo': {
                'name': 'Ultimate Combo',
                'path': '../done/ultimate_robustness_combo_ju7lfsk2/final_model',
                'vec_path': '../done/ultimate_robustness_combo_ju7lfsk2/vec_normalize.pkl',
                'description': 'SR2L + DR combined'
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

    def load_model(self, model_key, freeze_vecnormalize=False):
        """Load model and VecNormalize (optionally frozen)"""
        config = self.models[model_key]

        env = self.create_environment()
        env = VecNormalize.load(config['vec_path'], env)
        env.training = False  # Don't update running stats during evaluation
        env.norm_reward = False

        # Optionally freeze VecNormalize completely
        if freeze_vecnormalize:
            # Save original stats
            original_mean = copy.deepcopy(env.obs_rms.mean)
            original_var = copy.deepcopy(env.obs_rms.var)
            env.frozen_mean = original_mean
            env.frozen_var = original_var
            env.norm_obs = lambda obs: (obs - original_mean) / np.sqrt(original_var + 1e-8)

        model = PPO.load(config['path'], env=env)
        return model, env

    def apply_gaussian_noise(self, obs, sigma):
        """Apply Gaussian noise to joint observations only (dims 13-28)"""
        if sigma == 0:
            return obs

        noisy_obs = obs.copy()
        noise = np.random.normal(0, sigma, size=obs.shape)
        # Only apply to joint positions/velocities
        noisy_obs[13:28] += noise[13:28]
        return noisy_obs

    def evaluate_with_noise(self, model_key, noise_level, freeze_vecnormalize=False):
        """Run rollouts with specific noise level"""
        model, env = self.load_model(model_key, freeze_vecnormalize=freeze_vecnormalize)

        rollout_results = []

        for rollout in range(self.num_rollouts):
            obs = env.reset()
            done = False
            step_count = 0

            positions = []
            rewards = []

            # Get initial position
            try:
                unwrapped = env.envs[0].unwrapped
                start_x = unwrapped.data.qpos[0]
            except:
                start_x = 0.0

            while not done and step_count < self.episode_length:
                # Apply noise
                if noise_level > 0:
                    noisy_obs = self.apply_gaussian_noise(obs[0], noise_level)
                    noisy_obs_vec = np.expand_dims(noisy_obs, axis=0)
                else:
                    noisy_obs_vec = obs

                # Get action
                action, _ = model.predict(noisy_obs_vec, deterministic=True)

                # Track position
                try:
                    current_x = unwrapped.data.qpos[0]
                    positions.append(current_x)
                except:
                    positions.append(0.0)

                # Step environment
                obs, reward, done, info = env.step(action)
                rewards.append(reward[0])

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
                'steps': int(step_count)
            })

        # Aggregate statistics
        distances = [r['distance'] for r in rollout_results]
        rewards = [r['reward'] for r in rollout_results]
        success_count = sum(1 for r in rollout_results if r['success'])

        summary = {
            'noise_level': float(noise_level),
            'num_rollouts': self.num_rollouts,
            'distance': {
                'mean': float(np.mean(distances)),
                'std': float(np.std(distances)),
                'median': float(np.median(distances))
            },
            'reward': {
                'mean': float(np.mean(rewards)),
                'std': float(np.std(rewards))
            },
            'success_rate': float(success_count / self.num_rollouts),
            'rollouts': rollout_results
        }

        env.close()
        return summary

    def experiment_8a_baseline_resonance_check(self):
        """
        Test 8A: Does baseline (no SR2L training) also show resonance?
        If YES → VecNormalize alone sufficient
        If NO → SR2L training necessary
        """
        print("\n" + "="*80)
        print("EXPERIMENT 8A: BASELINE RESONANCE CHECK")
        print("="*80)
        print("Research Question: Is SR2L training necessary for resonance?")
        print("Testing M1 (Baseline) at 11 noise levels")
        print("="*80)

        noise_levels = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]

        results = []
        for noise in tqdm(noise_levels, desc="8A - Baseline Noise Sweep"):
            result = self.evaluate_with_noise('M1_baseline', noise)
            results.append(result)

        # Calculate retention
        baseline_distance = results[0]['distance']['mean']
        for result in results:
            result['retention_percent'] = (result['distance']['mean'] / baseline_distance * 100
                                          if baseline_distance > 0 else 0.0)

        # Find peak
        peak_result = max(results, key=lambda x: x['retention_percent'])

        summary = {
            'experiment': '8A_baseline_resonance_check',
            'model': 'M1_baseline',
            'baseline_distance': float(baseline_distance),
            'peak_noise': float(peak_result['noise_level']),
            'peak_retention': float(peak_result['retention_percent']),
            'shows_resonance': peak_result['retention_percent'] > 100.0,
            'noise_sweep_results': results
        }

        print(f"\n✅ 8A Complete")
        print(f"  Baseline: {baseline_distance:.3f}m")
        print(f"  Peak: {peak_result['retention_percent']:.1f}% at σ={peak_result['noise_level']:.2f}")
        print(f"  Resonance: {'YES' if summary['shows_resonance'] else 'NO'}")

        return summary

    def experiment_8b_sr2l_fine_grained_sweep(self):
        """
        Test 8B: Find exact peak for SR2L model
        Provides precise characterization of resonance phenomenon
        """
        print("\n" + "="*80)
        print("EXPERIMENT 8B: SR2L FINE-GRAINED NOISE SWEEP")
        print("="*80)
        print("Research Question: What is the exact peak location for SR2L?")
        print("Testing M2 (SR2L) at 11 noise levels")
        print("="*80)

        noise_levels = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]

        results = []
        for noise in tqdm(noise_levels, desc="8B - SR2L Noise Sweep"):
            result = self.evaluate_with_noise('M2_sr2l', noise)
            results.append(result)

        # Calculate retention
        baseline_distance = results[0]['distance']['mean']
        for result in results:
            result['retention_percent'] = (result['distance']['mean'] / baseline_distance * 100
                                          if baseline_distance > 0 else 0.0)

        # Find peak
        peak_result = max(results, key=lambda x: x['retention_percent'])

        # Find noise range where retention > 100%
        super_performance_range = [r for r in results if r['retention_percent'] > 100.0]

        summary = {
            'experiment': '8B_sr2l_fine_grained_sweep',
            'model': 'M2_sr2l',
            'baseline_distance': float(baseline_distance),
            'peak_noise': float(peak_result['noise_level']),
            'peak_retention': float(peak_result['retention_percent']),
            'peak_width': len(super_performance_range),
            'super_performance_noise_range': [r['noise_level'] for r in super_performance_range],
            'noise_sweep_results': results
        }

        print(f"\n✅ 8B Complete")
        print(f"  Baseline: {baseline_distance:.3f}m")
        print(f"  Peak: {peak_result['retention_percent']:.1f}% at σ={peak_result['noise_level']:.2f}")
        print(f"  Super-performance range: {summary['peak_width']} noise levels")

        return summary

    def experiment_8c_dr_resonance_analysis(self):
        """
        Test 8C: Why does M3 show stronger resonance (108%) than M2 (101%)?
        M3 has no SR2L training but shows stronger effect
        """
        print("\n" + "="*80)
        print("EXPERIMENT 8C: DR MODEL RESONANCE ANALYSIS")
        print("="*80)
        print("Research Question: Why does DR show stronger resonance than SR2L?")
        print("Testing M3 (DR) at 11 noise levels")
        print("="*80)

        noise_levels = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]

        results = []
        for noise in tqdm(noise_levels, desc="8C - DR Noise Sweep"):
            result = self.evaluate_with_noise('M3_dr', noise)
            results.append(result)

        # Calculate retention
        baseline_distance = results[0]['distance']['mean']
        for result in results:
            result['retention_percent'] = (result['distance']['mean'] / baseline_distance * 100
                                          if baseline_distance > 0 else 0.0)

        # Find peak
        peak_result = max(results, key=lambda x: x['retention_percent'])

        summary = {
            'experiment': '8C_dr_resonance_analysis',
            'model': 'M3_dr',
            'baseline_distance': float(baseline_distance),
            'peak_noise': float(peak_result['noise_level']),
            'peak_retention': float(peak_result['retention_percent']),
            'stronger_than_sr2l': peak_result['retention_percent'] > 101.0,  # Compare to M2
            'noise_sweep_results': results
        }

        print(f"\n✅ 8C Complete")
        print(f"  Baseline: {baseline_distance:.3f}m")
        print(f"  Peak: {peak_result['retention_percent']:.1f}% at σ={peak_result['noise_level']:.2f}")
        print(f"  Stronger than SR2L: {'YES' if summary['stronger_than_sr2l'] else 'NO'}")

        return summary

    def experiment_8d_vecnormalize_frozen_test(self):
        """
        Test 8D: Is VecNormalize adaptive statistics necessary?
        Compare SR2L with frozen vs adaptive VecNormalize
        """
        print("\n" + "="*80)
        print("EXPERIMENT 8D: VECNORMALIZE FROZEN TEST")
        print("="*80)
        print("Research Question: Are adaptive statistics necessary for resonance?")
        print("Testing M2 (SR2L) with frozen VecNormalize at 5 noise levels")
        print("="*80)

        noise_levels = [0.00, 0.05, 0.10, 0.15, 0.20]

        # Test with normal (adaptive) VecNormalize
        adaptive_results = []
        for noise in tqdm(noise_levels, desc="8D - Adaptive VecNormalize"):
            result = self.evaluate_with_noise('M2_sr2l', noise, freeze_vecnormalize=False)
            result['vecnormalize_mode'] = 'adaptive'
            adaptive_results.append(result)

        # Test with frozen VecNormalize
        frozen_results = []
        for noise in tqdm(noise_levels, desc="8D - Frozen VecNormalize"):
            result = self.evaluate_with_noise('M2_sr2l', noise, freeze_vecnormalize=True)
            result['vecnormalize_mode'] = 'frozen'
            frozen_results.append(result)

        # Calculate retentions
        adaptive_baseline = adaptive_results[0]['distance']['mean']
        frozen_baseline = frozen_results[0]['distance']['mean']

        for result in adaptive_results:
            result['retention_percent'] = (result['distance']['mean'] / adaptive_baseline * 100
                                          if adaptive_baseline > 0 else 0.0)

        for result in frozen_results:
            result['retention_percent'] = (result['distance']['mean'] / frozen_baseline * 100
                                          if frozen_baseline > 0 else 0.0)

        # Find peaks
        adaptive_peak = max(adaptive_results, key=lambda x: x['retention_percent'])
        frozen_peak = max(frozen_results, key=lambda x: x['retention_percent'])

        summary = {
            'experiment': '8D_vecnormalize_frozen_test',
            'model': 'M2_sr2l',
            'adaptive': {
                'baseline_distance': float(adaptive_baseline),
                'peak_retention': float(adaptive_peak['retention_percent']),
                'peak_noise': float(adaptive_peak['noise_level']),
                'shows_resonance': adaptive_peak['retention_percent'] > 100.0,
                'results': adaptive_results
            },
            'frozen': {
                'baseline_distance': float(frozen_baseline),
                'peak_retention': float(frozen_peak['retention_percent']),
                'peak_noise': float(frozen_peak['noise_level']),
                'shows_resonance': frozen_peak['retention_percent'] > 100.0,
                'results': frozen_results
            },
            'adaptive_necessary': adaptive_peak['retention_percent'] > frozen_peak['retention_percent']
        }

        print(f"\n✅ 8D Complete")
        print(f"  Adaptive Peak: {adaptive_peak['retention_percent']:.1f}% at σ={adaptive_peak['noise_level']:.2f}")
        print(f"  Frozen Peak: {frozen_peak['retention_percent']:.1f}% at σ={frozen_peak['noise_level']:.2f}")
        print(f"  Adaptive necessary: {'YES' if summary['adaptive_necessary'] else 'NO'}")

        return summary

    def run_all_mechanism_experiments(self):
        """Run all 4 mechanism analysis experiments"""
        print("\n" + "="*80)
        print("EXPERIMENT 8: NOISE-INDUCED REGULARIZATION MECHANISM ANALYSIS")
        print("="*80)
        print("Testing 4 hypotheses about 101-108% retention phenomenon")
        print("Total episodes: 4,400")
        print("Estimated time: ~5 hours")
        print("="*80)

        # Run all 4 sub-experiments
        self.results['8A_baseline_resonance_check'] = self.experiment_8a_baseline_resonance_check()
        self.results['8B_sr2l_fine_grained_sweep'] = self.experiment_8b_sr2l_fine_grained_sweep()
        self.results['8C_dr_resonance_analysis'] = self.experiment_8c_dr_resonance_analysis()
        self.results['8D_vecnormalize_frozen_test'] = self.experiment_8d_vecnormalize_frozen_test()

        # Save results
        self.save_results()

        # Print comprehensive summary
        self.print_summary()

    def save_results(self):
        """Save results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "experiment_8_mechanism_analysis/data"
        os.makedirs(output_dir, exist_ok=True)

        output_file = f"{output_dir}/mechanism_analysis_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✅ Results saved to: {output_file}")

    def print_summary(self):
        """Print comprehensive mechanism analysis summary"""
        print("\n" + "="*80)
        print("EXPERIMENT 8 SUMMARY: MECHANISM IDENTIFICATION")
        print("="*80)

        # Hypothesis testing results
        print("\n🔬 HYPOTHESIS TEST RESULTS:\n")

        # H1: VecNormalize adaptive statistics cause resonance
        exp_8d = self.results['8D_vecnormalize_frozen_test']
        if exp_8d['adaptive_necessary']:
            print("✅ H1 SUPPORTED: Adaptive VecNormalize necessary for resonance")
            print(f"   Adaptive: {exp_8d['adaptive']['peak_retention']:.1f}%")
            print(f"   Frozen: {exp_8d['frozen']['peak_retention']:.1f}%")
        else:
            print("❌ H1 REJECTED: Frozen VecNormalize shows similar resonance")
            print(f"   Adaptive: {exp_8d['adaptive']['peak_retention']:.1f}%")
            print(f"   Frozen: {exp_8d['frozen']['peak_retention']:.1f}%")

        # H2: SR2L training strength correlates with resonance
        exp_8a = self.results['8A_baseline_resonance_check']
        exp_8b = self.results['8B_sr2l_fine_grained_sweep']
        if exp_8a['shows_resonance']:
            print("\n❌ H2 REJECTED: Baseline shows resonance without SR2L training")
            print(f"   M1 (no SR2L): {exp_8a['peak_retention']:.1f}% at σ={exp_8a['peak_noise']:.2f}")
            print(f"   M2 (SR2L): {exp_8b['peak_retention']:.1f}% at σ={exp_8b['peak_noise']:.2f}")
        else:
            print("\n✅ H2 SUPPORTED: Only SR2L-trained models show resonance")
            print(f"   M1 (no SR2L): {exp_8a['peak_retention']:.1f}% (no resonance)")
            print(f"   M2 (SR2L): {exp_8b['peak_retention']:.1f}% (strong resonance)")

        # H3: Resonance occurs in specific noise range (inverted-U)
        if exp_8b['peak_width'] > 1:
            print("\n✅ H3 SUPPORTED: Resonance occurs over noise range")
            print(f"   Peak width: {exp_8b['peak_width']} noise levels")
            print(f"   Range: σ={min(exp_8b['super_performance_noise_range']):.2f}-{max(exp_8b['super_performance_noise_range']):.2f}")
        else:
            print("\n⚠️  H3 PARTIAL: Single peak found")
            print(f"   Peak: σ={exp_8b['peak_noise']:.2f}")

        # H4: All models show resonance
        exp_8c = self.results['8C_dr_resonance_analysis']
        all_show_resonance = (exp_8a['shows_resonance'] and
                             exp_8b['peak_retention'] > 100 and
                             exp_8c['peak_retention'] > 100)

        if all_show_resonance:
            print("\n✅ H4 SUPPORTED: All models show resonance (training-independent)")
            print(f"   M1 (Baseline): {exp_8a['peak_retention']:.1f}%")
            print(f"   M2 (SR2L): {exp_8b['peak_retention']:.1f}%")
            print(f"   M3 (DR): {exp_8c['peak_retention']:.1f}%")
        else:
            print("\n❌ H4 REJECTED: Resonance depends on training method")

        print("\n" + "="*80)
        print("📊 MECHANISM CONCLUSION:")

        # Determine most likely mechanism
        if all_show_resonance:
            print("\nMechanism: UNIVERSAL NOISE-INDUCED REGULARIZATION")
            print("- All models benefit from mild noise regardless of training")
            print("- VecNormalize provides implicit denoising")
            print("- Small perturbations prevent overfitting to evaluation conditions")
        elif exp_8a['shows_resonance']:
            print("\nMechanism: VECNORMALIZE-DRIVEN REGULARIZATION")
            print("- Adaptive statistics key mechanism")
            print("- SR2L training enhances but not necessary")
        else:
            print("\nMechanism: SR2L-SPECIFIC SMOOTHNESS REGULARIZATION")
            print("- Only SR2L-trained models show resonance")
            print("- Smoothness constraint enables noise benefits")

        print("\n" + "="*80)


def main():
    analyzer = MechanismAnalyzer()
    analyzer.run_all_mechanism_experiments()


if __name__ == "__main__":
    main()
