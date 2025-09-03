#!/usr/bin/env python3
"""
Comprehensive Robustness Evaluation Suite
Tests all models against various failure modes and noise conditions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

try:
    from agents.ppo_sr2l import PPO_SR2L
    from envs.success_reward_wrapper import SuccessRewardWrapper
    from envs.target_walking_wrapper import TargetWalkingWrapper
    import realant_sim
except ImportError as e:
    print(f"Import warning: {e}")

class RobustnessEvaluationSuite:
    def __init__(self):
        self.models = {}
        self.results = {}
        
        # Test configurations
        self.noise_levels = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25]
        self.joint_failure_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.test_episodes = 20
        self.max_steps_per_episode = 1000
        
        # Failure modes to test
        self.failure_modes = {
            'sensor_noise': 'Gaussian noise added to observations',
            'joint_dropout': 'Random joint sensors set to zero',
            'joint_lock': 'Random joints locked in position',
            'joint_weakness': 'Random joints produce reduced torque',
            'delay_injection': 'Action delays simulating servo lag',
            'combined_failures': 'Multiple failure modes simultaneously'
        }
        
    def load_model(self, model_path: str, model_name: str, vec_normalize_path: Optional[str] = None):
        """Load a model for evaluation"""
        try:
            # Determine model type
            if 'sr2l' in model_path.lower():
                model = PPO_SR2L.load(model_path)
            else:
                model = PPO.load(model_path)
            
            # Create environment
            def make_env():
                _env = gym.make('RealAntMujoco-v0')
                _env = SuccessRewardWrapper(_env)
                _env = Monitor(_env)
                return _env
            
            env = DummyVecEnv([make_env])
            
            # Load normalization if available
            if vec_normalize_path and os.path.exists(vec_normalize_path):
                with open(vec_normalize_path, 'rb') as f:
                    vec_normalize = pickle.load(f)
                # CRITICAL: Set the venv attribute properly
                vec_normalize.venv = env
                env = vec_normalize
                env.training = False
            
            self.models[model_name] = {
                'model': model,
                'env': env,
                'path': model_path
            }
            
            print(f"✅ Loaded {model_name}: {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {str(e)}")
            return False
    
    def load_all_models(self):
        """Load all available models"""
        model_configs = [
            {
                'name': 'PPO Baseline',
                'path': 'done/ppo_baseline_ueqbjf2x/best_model/best_model.zip',
                'vec_path': 'done/ppo_baseline_ueqbjf2x/vec_normalize.pkl'
            },
            # Add other models as they become available
            # {
            #     'name': 'PPO + SR2L',
            #     'path': 'experiments/ppo_sr2l_corrected_*/best_model/best_model.zip',
            #     'vec_path': 'experiments/ppo_sr2l_corrected_*/vec_normalize.pkl'
            # },
            # {
            #     'name': 'PPO + DR',
            #     'path': 'experiments/ppo_dr_robust_*/best_model/best_model.zip',
            #     'vec_path': 'experiments/ppo_dr_robust_*/vec_normalize.pkl'
            # }
        ]
        
        loaded_models = 0
        for config in model_configs:
            if os.path.exists(config['path']):
                if self.load_model(config['path'], config['name'], config.get('vec_path')):
                    loaded_models += 1
            else:
                print(f"⏳ {config['name']} not yet available: {config['path']}")
        
        print(f"\\n📊 Loaded {loaded_models} models for evaluation")
        return loaded_models
    
    def inject_sensor_noise(self, obs: np.ndarray, noise_level: float) -> np.ndarray:
        """Inject Gaussian noise into observations"""
        if noise_level == 0:
            return obs
        
        noise = np.random.normal(0, noise_level, obs.shape)
        return obs + noise
    
    def simulate_joint_dropout(self, obs: np.ndarray, dropout_rate: float) -> np.ndarray:
        """Simulate joint sensor dropout"""
        if dropout_rate == 0:
            return obs
        
        # Handle vectorized observations properly
        joint_obs = obs.copy()
        
        # Check observation shape - it's likely (1, obs_dim) from VecEnv
        if len(joint_obs.shape) == 2:
            obs_dim = joint_obs.shape[1]
            # RealAnt has 29D observations
            # Joint angles are typically in indices 7-14, joint velocities in 15-22
            n_joints = min(8, (obs_dim - 7) // 2)  # Safe calculation
            
            for i in range(n_joints):
                if np.random.random() < dropout_rate:
                    # Dropout joint angle and velocity
                    if 7 + i < obs_dim:
                        joint_obs[0, 7 + i] = 0  # Joint angle
                    if 15 + i < obs_dim:
                        joint_obs[0, 15 + i] = 0  # Joint velocity
        else:
            # Fallback for 1D observations
            obs_dim = len(joint_obs)
            n_joints = min(8, (obs_dim - 7) // 2)
            
            for i in range(n_joints):
                if np.random.random() < dropout_rate:
                    if 7 + i < obs_dim:
                        joint_obs[7 + i] = 0
                    if 15 + i < obs_dim:
                        joint_obs[15 + i] = 0
        
        return joint_obs
    
    def simulate_joint_lock(self, action: np.ndarray, lock_rate: float) -> np.ndarray:
        """Simulate joint locking (action set to 0)"""
        if lock_rate == 0:
            return action
        
        locked_action = action.copy()
        for i in range(len(action)):
            if np.random.random() < lock_rate:
                locked_action[i] = 0  # Lock joint
        
        return locked_action
    
    def simulate_joint_weakness(self, action: np.ndarray, weakness_rate: float) -> np.ndarray:
        """Simulate joint weakness (reduced torque)"""
        if weakness_rate == 0:
            return action
        
        weak_action = action.copy()
        for i in range(len(action)):
            if np.random.random() < weakness_rate:
                weak_action[i] *= 0.3  # Reduce to 30% strength
        
        return weak_action
    
    def simulate_action_delay(self, action: np.ndarray, previous_action: np.ndarray, delay_rate: float) -> np.ndarray:
        """Simulate action delays"""
        if delay_rate == 0 or previous_action is None:
            return action
        
        # With some probability, return previous action (delay)
        if np.random.random() < delay_rate:
            return previous_action
        
        return action
    
    def run_episode_with_failures(self, model_name: str, failure_mode: str, 
                                 severity: float) -> Dict:
        """Run single episode with specified failure mode"""
        
        model_info = self.models[model_name]
        model = model_info['model']
        env = model_info['env']
        
        obs = env.reset()
        episode_reward = 0
        velocities = []
        actions_taken = []
        failure_events = 0
        previous_action = None
        
        for step in range(self.max_steps_per_episode):
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            original_action = action.copy()
            
            # Apply failure mode
            if failure_mode == 'sensor_noise':
                obs_with_failure = self.inject_sensor_noise(obs, severity)
                action, _ = model.predict(obs_with_failure, deterministic=True)
                
            elif failure_mode == 'joint_dropout':
                obs_with_failure = self.simulate_joint_dropout(obs, severity)
                action, _ = model.predict(obs_with_failure, deterministic=True)
                
            elif failure_mode == 'joint_lock':
                action = self.simulate_joint_lock(action, severity)
                
            elif failure_mode == 'joint_weakness':
                action = self.simulate_joint_weakness(action, severity)
                
            elif failure_mode == 'delay_injection':
                action = self.simulate_action_delay(action, previous_action, severity)
                
            elif failure_mode == 'combined_failures':
                # Apply multiple failure modes with reduced severity
                obs_with_failure = self.inject_sensor_noise(obs, severity * 0.5)
                action, _ = model.predict(obs_with_failure, deterministic=True)
                action = self.simulate_joint_weakness(action, severity * 0.3)
                action = self.simulate_action_delay(action, previous_action, severity * 0.2)
            
            # Count failure events
            if not np.allclose(action, original_action, atol=1e-6):
                failure_events += 1
            
            # Execute action
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            
            # Extract velocity - use the same successful method as research_demo_gui.py
            velocity = 0
            try:
                # Navigate through wrappers to get base env
                if hasattr(env, 'venv'):
                    base_env = env.venv.envs[0]
                else:
                    base_env = env.envs[0]
                
                # Get unwrapped env
                while hasattr(base_env, 'env'):
                    base_env = base_env.env
                
                # For MuJoCo envs, get x velocity directly from sim
                if hasattr(base_env, 'sim') and hasattr(base_env.sim, 'data'):
                    # Get center of mass velocity
                    velocity = base_env.sim.data.qvel[0]  # x-velocity (can be negative)
                    velocity = abs(velocity)  # Take absolute value
                else:
                    velocity = 0
            except Exception as e:
                velocity = 0
                if step == 0:  # Debug on first step
                    print(f"Debug - Velocity extraction failed: {e}")
            
            # Fallback methods if direct extraction failed
            if velocity == 0:
                # Method 1: Check info dict
                if len(info) > 0 and hasattr(info[0], 'get'):
                    velocity = info[0].get('speed', info[0].get('x_velocity', 0))
                
                # Method 2: Use reward as proxy (SuccessRewardWrapper)
                if velocity == 0 and reward[0] > 0:
                    velocity = reward[0] * 0.1  # Conservative scaling
            
            velocities.append(abs(velocity))
            actions_taken.append(action[0])
            
            previous_action = action
            
            if done[0]:
                break
        
        # Calculate metrics
        avg_velocity = np.mean(velocities) if velocities else 0
        smoothness = self.calculate_smoothness(actions_taken)
        stability = self.calculate_stability(velocities)
        
        return {
            'reward': episode_reward,
            'velocity': avg_velocity,
            'smoothness': smoothness,
            'stability': stability,
            'failure_events': failure_events,
            'steps': step + 1,
            'success': episode_reward > 50  # Adjust threshold as needed
        }
    
    def calculate_smoothness(self, actions: List[np.ndarray]) -> float:
        """Calculate action smoothness score"""
        if len(actions) < 2:
            return 1.0
        
        action_changes = []
        for i in range(len(actions) - 1):
            change = np.linalg.norm(actions[i+1] - actions[i])
            action_changes.append(change)
        
        mean_change = np.mean(action_changes)
        return 1.0 / (1.0 + mean_change)
    
    def calculate_stability(self, velocities: List[float]) -> float:
        """Calculate velocity stability (inverse of standard deviation)"""
        if len(velocities) < 2:
            return 1.0
        
        velocity_std = np.std(velocities)
        return 1.0 / (1.0 + velocity_std)
    
    def evaluate_model_robustness(self, model_name: str, failure_mode: str, 
                                 severity_levels: List[float]) -> Dict:
        """Evaluate model robustness across severity levels"""
        
        print(f"\\n🧪 Testing {model_name} - {failure_mode}")
        results = {'severity_levels': severity_levels}
        
        for metric in ['success_rate', 'avg_velocity', 'avg_smoothness', 
                      'avg_stability', 'avg_reward']:
            results[metric] = []
            results[f'{metric}_std'] = []
        
        for severity in severity_levels:
            print(f"  Severity {severity:.2f}: ", end="", flush=True)
            
            episode_results = []
            for episode in range(self.test_episodes):
                result = self.run_episode_with_failures(model_name, failure_mode, severity)
                episode_results.append(result)
                print(".", end="", flush=True)
            
            # Aggregate results
            success_rate = np.mean([r['success'] for r in episode_results]) * 100
            avg_velocity = np.mean([r['velocity'] for r in episode_results])
            avg_smoothness = np.mean([r['smoothness'] for r in episode_results])
            avg_stability = np.mean([r['stability'] for r in episode_results])
            avg_reward = np.mean([r['reward'] for r in episode_results])
            
            # Standard deviations
            velocity_std = np.std([r['velocity'] for r in episode_results])
            smoothness_std = np.std([r['smoothness'] for r in episode_results])
            stability_std = np.std([r['stability'] for r in episode_results])
            reward_std = np.std([r['reward'] for r in episode_results])
            
            results['success_rate'].append(success_rate)
            results['avg_velocity'].append(avg_velocity)
            results['avg_smoothness'].append(avg_smoothness)
            results['avg_stability'].append(avg_stability)
            results['avg_reward'].append(avg_reward)
            
            results['success_rate_std'].append(0)  # Success rate std calculated differently
            results['avg_velocity_std'].append(velocity_std)
            results['avg_smoothness_std'].append(smoothness_std)
            results['avg_stability_std'].append(stability_std)
            results['avg_reward_std'].append(reward_std)
            
            print(f" ✅ Success: {success_rate:.1f}%, Velocity: {avg_velocity:.3f} m/s")
        
        return results
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive robustness evaluation"""
        print("="*60)
        print("🚀 COMPREHENSIVE ROBUSTNESS EVALUATION SUITE")
        print("="*60)
        
        # Load models
        if self.load_all_models() == 0:
            print("❌ No models available for evaluation")
            return
        
        # Run evaluations
        self.results = {}
        
        for model_name in self.models.keys():
            self.results[model_name] = {}
            
            # Test sensor noise
            self.results[model_name]['sensor_noise'] = self.evaluate_model_robustness(
                model_name, 'sensor_noise', self.noise_levels)
            
            # Test joint failures
            self.results[model_name]['joint_dropout'] = self.evaluate_model_robustness(
                model_name, 'joint_dropout', [0.0, 0.1, 0.2, 0.3, 0.4])
            
            # Test joint locking
            self.results[model_name]['joint_lock'] = self.evaluate_model_robustness(
                model_name, 'joint_lock', [0.0, 0.1, 0.2, 0.3, 0.4])
            
            # Test combined failures
            self.results[model_name]['combined_failures'] = self.evaluate_model_robustness(
                model_name, 'combined_failures', [0.0, 0.1, 0.2, 0.3, 0.4])
        
        print("\\n✅ Evaluation complete!")
        return self.results
    
    def create_robustness_report(self, save_path: str = None):
        """Create comprehensive robustness report"""
        
        if not self.results:
            print("❌ No results to report. Run evaluation first.")
            return
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Comprehensive Robustness Evaluation Report', 
                    fontsize=20, fontweight='bold')
        
        # 1. Sensor Noise Robustness
        ax1 = fig.add_subplot(gs[0, :2])
        self.plot_failure_mode_comparison(ax1, 'sensor_noise', 'Sensor Noise Robustness')
        
        # 2. Joint Failure Robustness
        ax2 = fig.add_subplot(gs[0, 2:])
        self.plot_failure_mode_comparison(ax2, 'joint_dropout', 'Joint Dropout Robustness')
        
        # 3. Combined Failures
        ax3 = fig.add_subplot(gs[1, :2])
        self.plot_failure_mode_comparison(ax3, 'combined_failures', 'Combined Failure Robustness')
        
        # 4. Performance Degradation
        ax4 = fig.add_subplot(gs[1, 2:])
        self.plot_performance_degradation(ax4)
        
        # 5. Robustness Heatmap
        ax5 = fig.add_subplot(gs[2, :2])
        self.plot_robustness_heatmap(ax5)
        
        # 6. Summary Statistics
        ax6 = fig.add_subplot(gs[2, 2:])
        self.plot_summary_statistics(ax6)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Report saved to: {save_path}")
        else:
            plt.show()
        
        # Generate text report
        self.generate_text_report()
        
        return fig
    
    def plot_failure_mode_comparison(self, ax, failure_mode: str, title: str):
        """Plot comparison for specific failure mode"""
        
        for model_name, model_results in self.results.items():
            if failure_mode in model_results:
                data = model_results[failure_mode]
                severity_levels = data['severity_levels']
                success_rates = data['success_rate']
                
                ax.plot(severity_levels, success_rates, 'o-', linewidth=2, 
                       markersize=8, label=model_name)
        
        ax.set_xlabel('Failure Severity')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 100)
    
    def plot_performance_degradation(self, ax):
        """Plot performance degradation across failure modes"""
        
        failure_modes = ['sensor_noise', 'joint_dropout', 'joint_lock', 'combined_failures']
        
        for model_name in self.results.keys():
            degradations = []
            
            for failure_mode in failure_modes:
                if failure_mode in self.results[model_name]:
                    data = self.results[model_name][failure_mode]
                    baseline_perf = data['success_rate'][0]  # No failure
                    worst_perf = min(data['success_rate'])   # Worst failure
                    degradation = baseline_perf - worst_perf
                    degradations.append(degradation)
                else:
                    degradations.append(0)
            
            x = np.arange(len(failure_modes))
            ax.bar(x, degradations, alpha=0.7, label=model_name)
        
        ax.set_xlabel('Failure Mode')
        ax.set_ylabel('Performance Degradation (%)')
        ax.set_title('Performance Degradation by Failure Mode')
        ax.set_xticks(x)
        ax.set_xticklabels([fm.replace('_', '\\n') for fm in failure_modes])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_robustness_heatmap(self, ax):
        """Plot robustness heatmap across all conditions"""
        
        # Create matrix of robustness scores
        failure_modes = ['sensor_noise', 'joint_dropout', 'joint_lock', 'combined_failures']
        models = list(self.results.keys())
        
        robustness_matrix = []
        for model in models:
            row = []
            for failure_mode in failure_modes:
                if failure_mode in self.results[model]:
                    # Average success rate across severity levels
                    avg_success = np.mean(self.results[model][failure_mode]['success_rate'])
                    row.append(avg_success)
                else:
                    row.append(0)
            robustness_matrix.append(row)
        
        robustness_matrix = np.array(robustness_matrix)
        
        # Create heatmap
        im = ax.imshow(robustness_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        # Add labels
        ax.set_xticks(range(len(failure_modes)))
        ax.set_xticklabels([fm.replace('_', '\\n') for fm in failure_modes])
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(failure_modes)):
                text = ax.text(j, i, f'{robustness_matrix[i, j]:.0f}%',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Robustness Heatmap\\n(Average Success Rate)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Success Rate (%)')
    
    def plot_summary_statistics(self, ax):
        """Plot summary statistics"""
        ax.axis('off')
        
        # Calculate summary statistics
        summary_text = "ROBUSTNESS EVALUATION SUMMARY\\n\\n"
        
        for model_name in self.results.keys():
            summary_text += f"{model_name}:\\n"
            
            # Calculate overall robustness score
            all_success_rates = []
            for failure_mode in self.results[model_name].values():
                all_success_rates.extend(failure_mode['success_rate'])
            
            overall_robustness = np.mean(all_success_rates)
            robustness_std = np.std(all_success_rates)
            
            summary_text += f"  Overall Robustness: {overall_robustness:.1f}% ± {robustness_std:.1f}%\\n"
            summary_text += f"  Best Performance: {max(all_success_rates):.1f}%\\n"
            summary_text += f"  Worst Performance: {min(all_success_rates):.1f}%\\n"
            summary_text += "\\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3))
    
    def generate_text_report(self):
        """Generate detailed text report"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_file = f"robustness_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\\n")
            f.write("COMPREHENSIVE ROBUSTNESS EVALUATION REPORT\\n")
            f.write("="*80 + "\\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Test Episodes per Condition: {self.test_episodes}\\n")
            f.write(f"Max Steps per Episode: {self.max_steps_per_episode}\\n\\n")
            
            for model_name, model_results in self.results.items():
                f.write(f"MODEL: {model_name}\\n")
                f.write("-" * 40 + "\\n")
                
                for failure_mode, data in model_results.items():
                    f.write(f"\\nFailure Mode: {failure_mode}\\n")
                    f.write(f"Description: {self.failure_modes.get(failure_mode, 'Unknown')}\\n")
                    
                    severity_levels = data['severity_levels']
                    success_rates = data['success_rate']
                    velocities = data['avg_velocity']
                    
                    for i, severity in enumerate(severity_levels):
                        f.write(f"  Severity {severity:.2f}: Success {success_rates[i]:.1f}%, "
                               f"Velocity {velocities[i]:.3f} m/s\\n")
                
                f.write("\\n" + "="*40 + "\\n\\n")
        
        print(f"📄 Detailed report saved to: {report_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run comprehensive robustness evaluation')
    parser.add_argument('--episodes', type=int, default=20, 
                       help='Number of test episodes per condition')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum steps per episode')
    parser.add_argument('--save-report', type=str, 
                       help='Save path for the evaluation report')
    parser.add_argument('--models', type=str, nargs='+',
                       help='Specific models to evaluate (default: all available)')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = RobustnessEvaluationSuite()
    evaluator.test_episodes = args.episodes
    evaluator.max_steps_per_episode = args.max_steps
    
    # Run evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    if results:
        # Create report
        if args.save_report:
            save_path = args.save_report
        else:
            save_path = f"robustness_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        evaluator.create_robustness_report(save_path)
        
        print("\\n" + "="*60)
        print("🎉 ROBUSTNESS EVALUATION COMPLETE!")
        print("="*60)
        print(f"📊 Visual report: {save_path}")
        print("📄 Text report: robustness_report_*.txt")
        print("✅ Ready for research analysis!")
    
    else:
        print("❌ No evaluation results generated")

if __name__ == "__main__":
    main()