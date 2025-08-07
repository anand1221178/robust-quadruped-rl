#!/bin/bash
#SBATCH --job-name=ppo_baseline
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --mail-user=your-email@wits.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL

# Print job info
echo "Starting job on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"

# Load required modules (adjust based on your cluster)
module load python/3.9
module load cuda/11.8  # Even for CPU, some packages need CUDA

# Create a virtual environment if it doesn't exist
if [ ! -d "venv_cluster" ]; then
    echo "Creating virtual environment..."
    python -m venv venv_cluster
fi

# Activate virtual environment
source venv_cluster/bin/activate

# Install dependencies if needed
echo "Checking dependencies..."
pip install --upgrade pip

# Install required packages
pip install torch --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8 version
pip install gymnasium stable-baselines3 wandb pyyaml tqdm matplotlib pandas
pip install mujoco dm-control

# Verify GPU is available
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}'); print(f'GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Export WANDB API key
export WANDB_API_KEY="2c6287ca2154b2592ecdd4f992f3a1a7fb7649fc"

# Set experiment name
EXPERIMENT="ppo_baseline"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Run training
echo "Starting training..."
python src/train.py \
    --config configs/experiments/${EXPERIMENT}.yaml \
    --override logging.wandb_entity="anandpatel1221178-university-of-the-witswatersrand" \
    --override logging.wandb_project="robust-quadruped-rl-cluster"

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code $?"
fi

echo "Job finished at $(date)"
