#!/usr/bin/env python3
"""
Check what metrics are available in W&B runs
"""
import wandb

# W&B configuration
ENTITY = "anandpatel1221178-university-of-the-witswatersrand"
PROJECT = "robust-quadruped-rl"

# Run IDs to check
RUN_IDS = {
    'M1_baseline': 'ym2jcllj',
    'M2_sr2l': 'ze09p0vf',
    'M3_dr': 'jtfwl2qf',
    'M4_combo': 'ju7lfsk2'
}

print("="*80)
print("CHECKING W&B METRICS")
print("="*80)

api = wandb.Api()

for model_key, run_id in RUN_IDS.items():
    print(f"\n{model_key} ({run_id}):")
    print("-" * 40)

    try:
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")

        # Get run info
        print(f"  Name: {run.name}")
        print(f"  State: {run.state}")
        print(f"  Created: {run.created_at}")

        # Try to get history keys
        history = run.history()

        if len(history) > 0:
            print(f"  Data points: {len(history)}")
            print(f"  Available metrics:")
            for col in sorted(history.columns):
                if not col.startswith('_'):
                    print(f"    - {col}")
        else:
            print(f"  ⚠️  No history data found")

        # Check summary metrics
        print(f"  Summary keys:")
        for key in sorted(run.summary.keys()):
            if not key.startswith('_'):
                print(f"    - {key}")

    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "="*80)
print("DONE")
print("="*80)
