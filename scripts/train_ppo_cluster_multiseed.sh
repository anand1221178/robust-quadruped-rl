#!/bin/bash

#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00

echo "========================================================"
echo "MULTI-SEED PPO Training on $(hostname)"
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
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Use system Python
PYTHON_CMD="/usr/bin/python3"
echo -e "\nUsing system Python: $PYTHON_CMD"
$PYTHON_CMD --version

# Install packages (same as before)
echo -e "\nInstalling packages..."
$PYTHON_CMD -m pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
$PYTHON_CMD -m pip install --user gymnasium mujoco stable-baselines3[extra]
$PYTHON_CMD -m pip install --user wandb pyyaml matplotlib pandas numpy scipy tensorboard

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY="2c6287ca2154b2592ecdd4f992f3a1a7fb7649fc"
export PYTHONPATH="${PWD}:${PWD}/src:${PYTHONPATH}"
export MUJOCO_GL=egl

# Get config name from argument
CONFIG_NAME=${1}

if [ -z "$CONFIG_NAME" ]; then
    echo "❌ Error: No config name provided!"
    echo "Usage: sbatch train_ppo_cluster_multiseed.sh <config_name>"
    exit 1
fi

# Run training
echo -e "\n========================================================"
echo "Starting training with config: ${CONFIG_NAME}"
echo "========================================================"

$PYTHON_CMD src/train.py \
    --config configs/multiseed_experiments/${CONFIG_NAME}.yaml

EXITCODE=$?

echo -e "\n========================================================"
echo "Job finished at: $(date)"
echo "Exit code: $EXITCODE"
echo "========================================================"

exit $EXITCODE
