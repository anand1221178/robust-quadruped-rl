#!/usr/bin/env python3
"""
MASTER RUNNER: NEW EXPERIMENTS (2B, 3 Enhanced, 7)

Runs all 3 new experiments sequentially:
1. Experiment 2B: Extended Noise Types (7,200 episodes, ~12 hours)
2. Experiment 3: Joint Failures with Recovery Tracking (3,200 episodes, ~5 hours)
3. Experiment 7: Joint × Noise Ablation (12,800 episodes, ~18 hours)

Total: 23,200 episodes, ~35 hours runtime

Usage:
    python run_new_experiments.py              # Run all 3
    python run_new_experiments.py --exp 2b     # Run only Experiment 2B
    python run_new_experiments.py --exp 3      # Run only Experiment 3
    python run_new_experiments.py --exp 7      # Run only Experiment 7
    python run_new_experiments.py --list       # List experiments

Author: Anand Patel
Date: October 19, 2025
"""

import sys
import os
import argparse
import time
from datetime import datetime, timedelta
import traceback

# Add src to path
sys.path.append('src')

# Import experiment modules
from experiment_2b_extended_noise_types import ExtendedNoiseEvaluator
from experiment_3_joint_failures import JointFailureEvaluator
from experiment_7_joint_noise_ablation import JointNoiseAblationEvaluator


class ExperimentRunner:
    """Orchestrates execution of all new experiments"""

    def __init__(self):
        self.experiments = {
            '2b': {
                'name': 'Experiment 2B: Extended Noise Types',
                'evaluator_class': ExtendedNoiseEvaluator,
                'episodes': 7_200,
                'estimated_hours': 12,
                'description': 'Tests 3 noise types (Gaussian, Poisson, Salt-Pepper) across 6 SNR-matched levels'
            },
            '3': {
                'name': 'Experiment 3: Joint Failures (Enhanced)',
                'evaluator_class': JointFailureEvaluator,
                'episodes': 3_200,
                'estimated_hours': 5,
                'description': 'Tests 8 individual joint failures with recovery time tracking'
            },
            '7': {
                'name': 'Experiment 7: Joint × Noise Ablation',
                'evaluator_class': JointNoiseAblationEvaluator,
                'episodes': 12_800,
                'estimated_hours': 18,
                'description': 'Full factorial: 8 joints × 4 noise conditions with interaction effects'
            }
        }

        self.results_log = []

    def print_header(self):
        """Print startup banner"""
        print("=" * 80)
        print("NEW EXPERIMENTS MASTER RUNNER")
        print("=" * 80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

    def list_experiments(self):
        """List all available experiments"""
        print("\n" + "=" * 80)
        print("AVAILABLE EXPERIMENTS")
        print("=" * 80)

        total_episodes = 0
        total_hours = 0

        for exp_id, exp_info in self.experiments.items():
            print(f"\n{exp_id}. {exp_info['name']}")
            print(f"   Episodes: {exp_info['episodes']:,}")
            print(f"   Estimated time: {exp_info['estimated_hours']} hours")
            print(f"   Description: {exp_info['description']}")

            total_episodes += exp_info['episodes']
            total_hours += exp_info['estimated_hours']

        print("\n" + "-" * 80)
        print(f"TOTAL: {total_episodes:,} episodes, ~{total_hours} hours")
        print("=" * 80)

    def run_experiment(self, exp_id):
        """Run a single experiment with error handling"""
        if exp_id not in self.experiments:
            print(f"❌ Unknown experiment: {exp_id}")
            return False

        exp_info = self.experiments[exp_id]

        print("\n" + "=" * 80)
        print(f"STARTING: {exp_info['name']}")
        print("=" * 80)
        print(f"Episodes: {exp_info['episodes']:,}")
        print(f"Estimated time: {exp_info['estimated_hours']} hours")
        print(f"Expected completion: {(datetime.now() + timedelta(hours=exp_info['estimated_hours'])).strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        start_time = time.time()

        try:
            # Create evaluator instance
            evaluator = exp_info['evaluator_class']()

            # Run evaluation
            evaluator.run_all_evaluations()

            # Success
            elapsed_time = time.time() - start_time
            elapsed_hours = elapsed_time / 3600

            result = {
                'experiment_id': exp_id,
                'experiment_name': exp_info['name'],
                'status': 'SUCCESS',
                'elapsed_time_hours': elapsed_hours,
                'estimated_hours': exp_info['estimated_hours'],
                'efficiency': exp_info['estimated_hours'] / elapsed_hours if elapsed_hours > 0 else 0,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            self.results_log.append(result)

            print("\n" + "=" * 80)
            print(f"✅ EXPERIMENT {exp_id} COMPLETED SUCCESSFULLY")
            print("=" * 80)
            print(f"Elapsed time: {elapsed_hours:.2f} hours")
            print(f"Estimated time: {exp_info['estimated_hours']} hours")
            print(f"Efficiency: {result['efficiency']:.2f}x")
            print("=" * 80)

            return True

        except KeyboardInterrupt:
            print("\n" + "=" * 80)
            print(f"⚠️  EXPERIMENT {exp_id} INTERRUPTED BY USER")
            print("=" * 80)

            result = {
                'experiment_id': exp_id,
                'experiment_name': exp_info['name'],
                'status': 'INTERRUPTED',
                'elapsed_time_hours': (time.time() - start_time) / 3600,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            self.results_log.append(result)

            # Re-raise to allow outer handler to decide
            raise

        except Exception as e:
            elapsed_time = time.time() - start_time
            elapsed_hours = elapsed_time / 3600

            print("\n" + "=" * 80)
            print(f"❌ EXPERIMENT {exp_id} FAILED")
            print("=" * 80)
            print(f"Error: {str(e)}")
            print("\nTraceback:")
            traceback.print_exc()
            print("=" * 80)

            result = {
                'experiment_id': exp_id,
                'experiment_name': exp_info['name'],
                'status': 'FAILED',
                'error': str(e),
                'elapsed_time_hours': elapsed_hours,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            self.results_log.append(result)

            return False

    def run_all(self, exp_ids=None):
        """Run all experiments or specific subset"""
        if exp_ids is None:
            exp_ids = ['2b', '3', '7']

        self.print_header()

        # Show what will be run
        print("\n📋 EXECUTION PLAN:")
        total_episodes = 0
        total_hours = 0
        for exp_id in exp_ids:
            if exp_id in self.experiments:
                exp = self.experiments[exp_id]
                print(f"  {exp_id}. {exp['name']}")
                print(f"     → {exp['episodes']:,} episodes, ~{exp['estimated_hours']} hours")
                total_episodes += exp['episodes']
                total_hours += exp['estimated_hours']

        print(f"\n  TOTAL: {total_episodes:,} episodes, ~{total_hours} hours")
        print(f"  Expected completion: {(datetime.now() + timedelta(hours=total_hours)).strftime('%Y-%m-%d %H:%M:%S')}")

        # Confirm before starting
        print("\n" + "=" * 80)
        response = input("▶️  Ready to start? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("❌ Aborted by user")
            return

        # Run experiments
        overall_start = time.time()

        try:
            for i, exp_id in enumerate(exp_ids, 1):
                print(f"\n\n{'='*80}")
                print(f"EXPERIMENT {i}/{len(exp_ids)}")
                print(f"{'='*80}")

                success = self.run_experiment(exp_id)

                if not success:
                    print(f"\n⚠️  Experiment {exp_id} failed, but continuing to next...")

        except KeyboardInterrupt:
            print("\n\n" + "=" * 80)
            print("⚠️  MASTER RUNNER INTERRUPTED BY USER")
            print("=" * 80)

        finally:
            # Print final summary
            self.print_summary(overall_start)

    def print_summary(self, overall_start):
        """Print final summary of all experiments"""
        overall_elapsed = (time.time() - overall_start) / 3600

        print("\n\n" + "=" * 80)
        print("FINAL SUMMARY - ALL EXPERIMENTS")
        print("=" * 80)
        print(f"Total elapsed time: {overall_elapsed:.2f} hours")
        print(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        if not self.results_log:
            print("No experiments completed.")
            return

        print("\nRESULTS BY EXPERIMENT:")
        print("-" * 80)

        for result in self.results_log:
            status_icon = {
                'SUCCESS': '✅',
                'FAILED': '❌',
                'INTERRUPTED': '⚠️'
            }.get(result['status'], '?')

            print(f"\n{status_icon} {result['experiment_name']}")
            print(f"   Status: {result['status']}")
            print(f"   Elapsed: {result['elapsed_time_hours']:.2f} hours")

            if result['status'] == 'SUCCESS':
                print(f"   Efficiency: {result['efficiency']:.2f}x")
            elif result['status'] == 'FAILED':
                print(f"   Error: {result.get('error', 'Unknown')}")

        # Overall statistics
        successes = sum(1 for r in self.results_log if r['status'] == 'SUCCESS')
        failures = sum(1 for r in self.results_log if r['status'] == 'FAILED')
        interrupted = sum(1 for r in self.results_log if r['status'] == 'INTERRUPTED')

        print("\n" + "-" * 80)
        print(f"Success: {successes} | Failed: {failures} | Interrupted: {interrupted}")
        print("=" * 80)

        if successes == len(self.results_log):
            print("\n🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY! 🎉")
        elif successes > 0:
            print(f"\n⚠️  {successes}/{len(self.results_log)} experiments completed successfully")
        else:
            print("\n❌ No experiments completed successfully")

        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Run new robustness evaluation experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_new_experiments.py              Run all 3 experiments (~35 hours)
  python run_new_experiments.py --exp 2b     Run only Experiment 2B
  python run_new_experiments.py --exp 3      Run only Experiment 3 (with recovery tracking)
  python run_new_experiments.py --exp 7      Run only Experiment 7 (joint × noise)
  python run_new_experiments.py --list       List all experiments
        """
    )

    parser.add_argument(
        '--exp',
        type=str,
        choices=['2b', '3', '7'],
        help='Run specific experiment only (2b, 3, or 7)'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available experiments and exit'
    )

    args = parser.parse_args()

    runner = ExperimentRunner()

    if args.list:
        runner.list_experiments()
        return

    if args.exp:
        # Run single experiment
        runner.run_all(exp_ids=[args.exp])
    else:
        # Run all experiments
        runner.run_all()


if __name__ == "__main__":
    main()
