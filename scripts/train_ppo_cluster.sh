#!/bin/bash

#SBATCH --job-name=PPO_Train
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=ppo_%j.out
#SBATCH --error=ppo_%j.err

echo "========================================================"
echo "PPO Training on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
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

# Clean up broken venv
rm -rf venv_gpu

# Use system Python directly
PYTHON_CMD="/usr/bin/python3"
echo -e "\nUsing system Python: $PYTHON_CMD"
$PYTHON_CMD --version

# Install pip for user if needed
echo -e "\nEnsuring pip is available..."
$PYTHON_CMD -m ensurepip --user 2>/dev/null || {
    echo "Installing pip manually..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    $PYTHON_CMD get-pip.py --user
    rm get-pip.py
}

# Upgrade pip
echo -e "\nUpgrading pip..."
$PYTHON_CMD -m pip install --user --upgrade pip

# Install all packages to user directory
echo -e "\nInstalling packages to user directory..."
$PYTHON_CMD -m pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
$PYTHON_CMD -m pip install --user gymnasium mujoco dm-control
$PYTHON_CMD -m pip install --user stable-baselines3[extra]  # This installs tqdm and rich
$PYTHON_CMD -m pip install --user wandb pyyaml matplotlib pandas numpy scipy
$PYTHON_CMD -m pip install --user tensorboard imageio
# Explicitly install tqdm and rich just to be sure
$PYTHON_CMD -m pip install --user tqdm rich

# Verify installations
echo -e "\nVerifying installations..."
$PYTHON_CMD -c "import torch; print(f'PyTorch: {torch.__version__}')" || echo "PyTorch not found"
$PYTHON_CMD -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')" || echo "Gymnasium not found"
$PYTHON_CMD -c "import stable_baselines3; print(f'SB3: {stable_baselines3.__version__}')" || echo "SB3 not found"
$PYTHON_CMD -c "import tqdm; print(f'tqdm: installed')" || echo "tqdm not found"
$PYTHON_CMD -c "import rich; print(f'rich: installed')" || echo "rich not found"

# Verify GPU access
echo -e "\nVerifying GPU access..."
$PYTHON_CMD -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY="2c6287ca2154b2592ecdd4f992f3a1a7fb7649fc"
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Add user site-packages to Python path
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:${PYTHONPATH}"

# Set MuJoCo to use software rendering (no display on cluster)
export MUJOCO_GL=egl

# Run training
echo -e "\n========================================================"
echo "Starting PPO training..."
echo "========================================================"

EXPERIMENT=${1:-ppo_baseline}  # Default to ppo_baseline if no argument

# Run training
$PYTHON_CMD src/train.py \
    --config configs/experiments/${EXPERIMENT}.yaml \
    --override logging.wandb_entity="anandpatel1221178-university-of-the-witswatersrand" \
    --override logging.wandb_project="robust-quadruped-rl"

EXITCODE=$?

echo -e "\n========================================================"
echo "Job finished at: $(date)"
echo "Exit code: $EXITCODE"
echo "========================================================"

exit $EXITCODE