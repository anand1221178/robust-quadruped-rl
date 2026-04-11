#!/usr/bin/env python3
"""
Generate multi-seed configs for v2 experiments.
Creates 5 seed variants for each of M1-M4 (20 configs total).
All configs share identical base hyperparameters from configs/train/tuned.yaml.
"""

import yaml
import os
import copy

SEEDS = [42, 123, 456, 789, 999]
BASE_DIR = "configs/experiments_v2"

# Load base configs
def load_base(name):
    path = os.path.join(BASE_DIR, f"{name}.yaml")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def write_config(config, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  Created: {path}")

def generate_seeds(base_name, model_label):
    base = load_base(base_name)

    for seed in SEEDS:
        config = copy.deepcopy(base)
        config['seed'] = seed

        # Update experiment name
        config['experiment']['name'] = f"{base_name}_seed{seed}"
        config['experiment']['description'] = (
            config['experiment'].get('description', '') + f" (seed={seed})"
        )

        # Update tags
        tags = config.get('experiment', {}).get('tags', [])
        tags.append(f"seed{seed}")
        config['experiment']['tags'] = tags

        # Update wandb tags
        wandb_tags = config.get('logging', {}).get('wandb_tags', [])
        wandb_tags.append(f"seed{seed}")
        config['logging']['wandb_tags'] = wandb_tags

        out_path = os.path.join(BASE_DIR, "multiseed", f"{base_name}_seed{seed}.yaml")
        write_config(config, out_path)

if __name__ == "__main__":
    print("Generating v2 multi-seed configs...")
    print(f"Seeds: {SEEDS}")
    print(f"Output: {BASE_DIR}/multiseed/\n")

    models = [
        ("M1_baseline_v2", "M1"),
        ("M2_sr2l_v2", "M2"),
        ("M3_dr_v2", "M3"),
        ("M4_combo_v2", "M4"),
    ]

    for base_name, label in models:
        print(f"\n{label}: {base_name}")
        generate_seeds(base_name, label)

    total = len(SEEDS) * len(models)
    print(f"\nDone! Generated {total} configs.")
    print(f"\nSubmit all with:")
    print(f"  bash scripts/submit_v2_all.sh")
