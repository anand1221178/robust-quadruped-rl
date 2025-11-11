#!/usr/bin/env python3
"""
Generate 20 training configs for multi-seed experiment
4 models × 5 seeds = 20 training runs
"""

import yaml
from pathlib import Path

# Define the 4 models and 5 seeds
MODELS = {
    'M1_baseline': {
        'base_config': 'configs/experiments/ppo_baseline_32M.yaml',
        'description': 'Baseline PPO (no robustness training)',
        'total_timesteps': 32_000_000
    },
    'M2_sr2l': {
        'base_config': 'configs/experiments/ppo_sr2l_32M.yaml',
        'description': 'SR2L sensor noise robustness',
        'total_timesteps': 32_000_000
    },
    'M3_dr': {
        'base_config': 'configs/experiments/ppo_dr_32M.yaml',
        'description': 'Domain Randomization joint failure robustness',
        'total_timesteps': 32_000_000
    },
    'M4_combo': {
        'base_config': 'configs/experiments/ppo_combo_30M.yaml',
        'description': 'Combined SR2L + DR',
        'total_timesteps': 30_000_000
    }
}

SEEDS = [42, 123, 456, 789, 999]

def generate_configs():
    """Generate all 20 config files"""
    output_dir = Path('configs/multiseed_experiments')
    output_dir.mkdir(exist_ok=True, parents=True)

    generated_configs = []

    for model_key, model_info in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Generating configs for: {model_key}")
        print(f"Description: {model_info['description']}")
        print('='*60)

        # Check if base config exists
        base_path = Path(model_info['base_config'])
        if not base_path.exists():
            print(f"  ⚠️  Base config not found: {base_path}")
            print(f"  Creating minimal config from defaults...")
            base_config = create_minimal_config(model_key, model_info)
        else:
            with open(base_path, 'r') as f:
                base_config = yaml.safe_load(f)

        # Generate configs for each seed
        for seed in SEEDS:
            # Modify config for this seed
            config = base_config.copy()
            config['seed'] = seed
            config['total_timesteps'] = model_info['total_timesteps']

            # Update experiment name to include seed
            if 'experiment' not in config:
                config['experiment'] = {}
            config['experiment']['name'] = f"{model_key}_seed{seed}"
            config['experiment']['description'] = f"{model_info['description']} (seed={seed})"

            # Update W&B run name
            if 'logging' not in config:
                config['logging'] = {}
            config['logging']['wandb'] = True
            config['logging']['wandb_project'] = 'robust-quadruped-rl-multiseed'
            config['logging']['wandb_tags'] = [model_key, f'seed{seed}', 'multiseed_experiment']

            # Save config
            output_file = output_dir / f"{model_key}_seed{seed}.yaml"
            with open(output_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            generated_configs.append({
                'model': model_key,
                'seed': seed,
                'config_path': str(output_file),
                'timesteps': model_info['total_timesteps']
            })

            print(f"   Generated: {output_file.name} (seed={seed})")

    print(f"\n{'='*60}")
    print(f" Generated {len(generated_configs)} configs!")
    print('='*60)

    # Save manifest for tracking
    manifest_file = output_dir / 'training_manifest.yaml'
    with open(manifest_file, 'w') as f:
        yaml.dump({
            'total_runs': len(generated_configs),
            'models': list(MODELS.keys()),
            'seeds': SEEDS,
            'configs': generated_configs
        }, f, default_flow_style=False)

    print(f"\n📋 Manifest saved to: {manifest_file}")

    return generated_configs

def create_minimal_config(model_key, model_info):
    """Create minimal config if base doesn't exist"""
    config = {
        'defaults': ['/train/default', '/env/realant'],
        'experiment': {
            'name': model_key,
            'description': model_info['description']
        },
        'total_timesteps': model_info['total_timesteps'],
        'seed': 42,  # Will be overwritten
        'save_freq': 1_000_000,  # Save checkpoint every 1M steps
        'logging': {
            'wandb': True,
            'wandb_project': 'robust-quadruped-rl-multiseed'
        }
    }

    # Model-specific settings
    if model_key == 'M2_sr2l':
        config['sr2l'] = {
            'enabled': True,
            'lambda_smooth': 0.001,
            'perturbation_std': 0.01,
            'perturbation_dims': list(range(13, 29))  # Joint dims only
        }
    elif model_key == 'M3_dr':
        config['domain_randomization'] = {
            'enabled': True,
            'joint_failure_prob': 0.5,
            'max_failed_joints': 2,
            'curriculum': {
                'enabled': True,
                'phase_1_steps': 10_000_000,
                'phase_2_steps': 10_000_000,
                'phase_3_steps': 12_000_000
            }
        }
    elif model_key == 'M4_combo':
        config['sr2l'] = {
            'enabled': True,
            'lambda_smooth': 0.001,
            'perturbation_std': 0.01,
            'perturbation_dims': list(range(13, 29))
        }
        config['domain_randomization'] = {
            'enabled': True,
            'joint_failure_prob': 0.5,
            'max_failed_joints': 2,
            'curriculum': {
                'enabled': True,
                'phase_1_steps': 10_000_000,
                'phase_2_steps': 10_000_000,
                'phase_3_steps': 10_000_000
            }
        }

    return config

if __name__ == "__main__":
    configs = generate_configs()

    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Review generated configs in: configs/multiseed_experiments/")
    print("2. Launch all 20 training runs:")
    print("   bash scripts/launch_multiseed_training.sh")
    print("3. Monitor progress on W&B: robust-quadruped-rl-multiseed")
    print("4. After training: run paper/reconstruct_multiseed_curves.py")
    print("="*60)
