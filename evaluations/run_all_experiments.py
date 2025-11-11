#!/usr/bin/env python3
"""
ULTIMATE MASTER RUNNER: ALL 9 EXPERIMENTS
Runs everything in one go - start it and walk away!

Total: 42,400+ episodes, ~65 hours

Usage:
    python run_ALL_experiments.py                    # Run ALL 9 experiments
    python run_ALL_experiments.py --only 8           # Run only mechanism analysis
    python run_ALL_experiments.py --only 2b,3,7,8    # Run all new ones
    python run_ALL_experiments.py --skip 1,2,4,5,6   # Skip completed, run new
    python run_ALL_experiments.py --list             # List all

Author: Anand Patel
Date: October 20, 2025
"""

import sys
import os
import subprocess
import time
from datetime import datetime, timedelta
import json

print("Ultimate Experiment Runner initialized!")
print("This will run experiments as separate processes to avoid import conflicts.")
print("Each experiment will run its main() function independently.\n")

# Experiment definitions
EXPERIMENTS = {
    '1': {'name': 'Baseline Performance', 'script': 'experiment_1_baseline.py', 'hours': 0.67, 'episodes': 400},
    '2': {'name': 'Sensor Noise', 'script': 'experiment_2_sensor_noise.py', 'hours': 8, 'episodes': 4800},
    '2b': {'name': 'Extended Noise Types (NEW)', 'script': 'experiment_2b_extended_noise_types.py', 'hours': 12, 'episodes': 7200},
    '3': {'name': 'Joint Failures + Recovery (ENHANCED)', 'script': 'experiment_3_joint_failures.py', 'hours': 5, 'episodes': 3200},
    '4': {'name': 'Combined Stress', 'script': 'experiment_4_combined_stress.py', 'hours': 4, 'episodes': 2400},
    '5': {'name': 'Per-Joint Deep Dive', 'script': 'experiment_5_per_joint_deep_dive.py', 'hours': 8, 'episodes': 4800},
    '6': {'name': 'Validation Suite', 'script': 'experiment_6_validation_suite.py', 'hours': 4, 'episodes': 2400},
    '7': {'name': 'Joint × Noise Ablation (NEW)', 'script': 'experiment_7_joint_noise_ablation.py', 'hours': 18, 'episodes': 12800},
    '8': {'name': 'Mechanism Analysis (NEW)', 'script': 'experiment_8_mechanism_analysis.py', 'hours': 5, 'episodes': 4400}
}

def list_experiments():
    print("="*80)
    print("ALL EXPERIMENTS")
    print("="*80)
    total_eps, total_hrs = 0, 0
    for eid in sorted(EXPERIMENTS.keys()):
        e = EXPERIMENTS[eid]
        print(f"{eid}. {e['name']}")
        print(f"   Episodes: {e['episodes']:,} | Time: ~{e['hours']:.1f} hrs")
        total_eps += e['episodes']
        total_hrs += e['hours']
    print(f"\nTOTAL: {total_eps:,} episodes, ~{total_hrs:.0f} hours ({total_hrs/24:.1f} days)")
    print("="*80)

def run_experiment(eid):
    e = EXPERIMENTS[eid]
    print(f"\n{'='*80}")
    print(f"STARTING: {e['name']}")
    print(f"{'='*80}")
    print(f"Episodes: {e['episodes']:,} | Estimated: {e['hours']:.1f} hrs")
    completion = datetime.now() + timedelta(hours=e['hours'])
    print(f"Expected completion: {completion.strftime('%H:%M:%S')}")
    print("="*80)
    
    start = time.time()
    script_path = os.path.join(os.path.dirname(__file__), e['script'])
    
    try:
        result = subprocess.run([sys.executable, script_path])
        elapsed = (time.time() - start) / 3600
        
        if result.returncode == 0:
            print(f"\n{'='*80}")
            print(f" EXPERIMENT {eid} COMPLETE")
            print(f"Time: {elapsed:.2f} hrs (estimated: {e['hours']:.1f} hrs)")
            print("="*80)
            return True
        else:
            print(f"\n❌ Experiment {eid} failed (exit code {result.returncode})")
            return False
    except Exception as ex:
        print(f"\n❌ Experiment {eid} crashed: {ex}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run ALL experiments')
    parser.add_argument('--only', help='Run only these (e.g., "2b,3,7")')
    parser.add_argument('--skip', help='Skip these (e.g., "1,2,4,5,6")')
    parser.add_argument('--list', action='store_true', help='List experiments')
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
        return
    
    # Determine which to run
    if args.only:
        to_run = [x.strip() for x in args.only.split(',')]
    else:
        to_run = sorted(EXPERIMENTS.keys())
    
    if args.skip:
        skip = [x.strip() for x in args.skip.split(',')]
        to_run = [x for x in to_run if x not in skip]
    
    # Show plan
    print("\n" + "="*80)
    print("🚀 EXECUTION PLAN")
    print("="*80)
    total_eps, total_hrs = 0, 0
    for i, eid in enumerate(to_run, 1):
        e = EXPERIMENTS[eid]
        print(f"{i}. {e['name']} ({e['episodes']:,} eps, {e['hours']:.1f} hrs)")
        total_eps += e['episodes']
        total_hrs += e['hours']
    
    print(f"\nTOTAL: {total_eps:,} episodes, ~{total_hrs:.0f} hours")
    completion = datetime.now() + timedelta(hours=total_hrs)
    print(f"Expected done: {completion.strftime('%A %b %d at %H:%M')}")
    print("="*80)
    
    confirm = input(f"\n▶️  Run {len(to_run)} experiments (~{total_hrs:.0f} hrs)? Type YES: ").strip()
    if confirm != 'YES':
        print("❌ Aborted")
        return
    
    # Run them!
    start_time = time.time()
    results = []
    
    for i, eid in enumerate(to_run, 1):
        print(f"\n\n{'#'*80}")
        print(f"EXPERIMENT {i}/{len(to_run)}")
        print(f"{'#'*80}")
        
        success = run_experiment(eid)
        results.append((eid, EXPERIMENTS[eid]['name'], success))
        
        if not success:
            cont = input("\n⚠️  Continue to next? (yes/no): ").strip().lower()
            if cont not in ['yes', 'y']:
                break
    
    # Final summary
    total_time = (time.time() - start_time) / 3600
    print("\n\n" + "="*80)
    print("🎉 FINAL SUMMARY")
    print("="*80)
    print(f"Total runtime: {total_time:.2f} hours")
    print(f"\nResults:")
    for eid, name, success in results:
        icon = "" if success else "❌"
        print(f"  {icon} {eid}. {name}")
    
    successes = sum(1 for _, _, s in results if s)
    print(f"\nSuccess: {successes}/{len(results)}")
    print("="*80)

if __name__ == "__main__":
    main()
