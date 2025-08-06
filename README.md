# Robust Quadruped Reinforcement Learning

Proactive Reinforcement Learning for Robust Quadruped Locomotion Under Limb Dropout and Sensor Noise.

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .  # Install project in development mode
```

3. Install MuJoCo (if not already installed):
- Follow instructions at: https://mujoco.org/

## Quick Start

### Local Testing
```bash
# Test environment
python src/train.py --config configs/experiments/ppo_baseline.yaml \
                   --override train.total_timesteps=10000

# Run ablation
python scripts/run_experiment.py --experiment ppo_dr --local
```

### Cluster Training
```bash
# Sync to cluster
./scripts/sync_to_cluster.sh

# Submit job
sbatch scripts/train_cluster.sh ppo_dr_sr2l
```

## Project Structure

- `configs/`: YAML configuration files
- `src/`: Source code (agents, environments, utilities)
- `scripts/`: Setup and utility scripts
- `experiments/`: Output directory for results
- `notebooks/`: Analysis and visualization

## Experiments

Run the following ablations:
1. `ppo_baseline`: PPO only
2. `ppo_dr`: PPO + Domain Randomization
3. `ppo_sr2l`: PPO + SR2L
4. `ppo_dr_sr2l`: PPO + DR + SR2L (full method)

## Citation

If you use this code, please cite:
```
@thesis{patel2025robust,
  title={Proactive Reinforcement Learning for Robust Quadruped Locomotion},
  author={Patel, Anand},
  year={2025},
  school={University of the Witwatersrand}
}
```
