#!/bin/bash

#SBATCH --job-name=PPO_v2
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --output=ppo_v2_%j.out
#SBATCH --error=ppo_v2_%j.err

echo "========================================================"
echo "PPO v2 (Tuned) Training on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Config: $1"
echo "Started at: $(date)"
echo "========================================================"

# Setup paths
export PATH="/usr/local/cuda-12.6/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH"

# Change to project directory
cd "/home-mscluster/panand/Research Proj/robust-quadruped-rl" || exit 1
echo "Working directory: $(pwd)"

# Show GPU info
echo -e "\nGPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv

# Use system Python
PYTHON_CMD="/usr/bin/python3"
echo -e "\nUsing system Python: $PYTHON_CMD"
$PYTHON_CMD --version

# Install packages
echo -e "\nInstalling packages..."
$PYTHON_CMD -m pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
$PYTHON_CMD -m pip install --user gymnasium mujoco dm-control
$PYTHON_CMD -m pip install --user stable-baselines3[extra]
$PYTHON_CMD -m pip install --user wandb pyyaml matplotlib pandas numpy scipy
$PYTHON_CMD -m pip install --user tensorboard tqdm rich

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY="2c6287ca2154b2592ecdd4f992f3a1a7fb7649fc"
export PYTHONPATH="${PWD}:${PWD}/src:${PYTHONPATH}"
export MUJOCO_GL=egl

# Get config path from argument
CONFIG_PATH=${1}

if [ -z "$CONFIG_PATH" ]; then
    echo "Error: No config path provided!"
    echo "Usage: sbatch scripts/train_v2_cluster.sh configs/experiments_v2/multiseed/M1_baseline_v2_seed42.yaml"
    exit 1
fi

# Run training
echo -e "\n========================================================"
echo "Starting training with config: ${CONFIG_PATH}"
echo "========================================================"

$PYTHON_CMD src/train.py --config ${CONFIG_PATH}

EXITCODE=$?

echo -e "\n========================================================"
echo "Job finished at: $(date)"
echo "Exit code: $EXITCODE"
echo "========================================================"

exit $EXITCODE
