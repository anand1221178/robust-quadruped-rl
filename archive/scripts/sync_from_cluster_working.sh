#!/bin/bash

# Sync from cluster - handles git and large files separately
# Usage: ./scripts/sync_from_cluster.sh

# Configuration
CLUSTER_USER="panand"
CLUSTER_HOST="mscluster-login1.ms.wits.ac.za"
CLUSTER_PROJECT_PATH="/home-mscluster/panand/Research Proj/robust-quadruped-rl"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Robust Quadruped RL - Cluster Sync Script ===${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "setup.py" ] || [ ! -d "src" ]; then
    echo -e "${RED}Error: Must run from project root directory${NC}"
    echo "Please cd to your robust-quadruped-rl directory first"
    exit 1
fi

# Step 1: Stash any local changes to prevent conflicts
echo -e "${YELLOW}Step 1: Stashing local changes...${NC}"
git stash push -m "Auto-stash before cluster sync $(date)"

# Step 2: Pull latest code from git
echo -e "${YELLOW}Step 2: Pulling latest code from GitHub...${NC}"
git pull origin master || {
    echo -e "${RED}Git pull failed. Resolve conflicts manually, then run this script again.${NC}"
    git stash pop
    exit 1
}

# Step 3: Sync ONLY essential experiment files
echo -e "${YELLOW}Step 3: Syncing experiment results from cluster...${NC}"
echo "Syncing only essential files (configs, models, evaluation results)..."

# Create experiments directory if it doesn't exist
mkdir -p experiments

# First, get list of experiment directories
ssh "${CLUSTER_USER}@${CLUSTER_HOST}" "cd '${CLUSTER_PROJECT_PATH}' && ls -d experiments/ppo_*/ 2>/dev/null" | while read -r exp_dir; do
    echo "Syncing $exp_dir"
    
    # Create the experiment directory locally
    mkdir -p "$exp_dir"
    
    # Sync only specific files from each experiment
    rsync -avz --progress \
        --include='config.yaml' \
        --include='final_model.zip' \
        --include='best_model/' \
        --include='best_model/best_model.zip' \
        --include='vec_normalize.pkl' \
        --include='eval/' \
        --include='eval/*' \
        --include='*.png' \
        --include='*.mp4' \
        --exclude='*' \
        "${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_PROJECT_PATH}/${exp_dir}" \
        "$(dirname "$exp_dir")/"
done

# Step 4: Sync any new scripts
echo -e "${YELLOW}Step 4: Syncing scripts...${NC}"
rsync -avz --progress \
    "${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_PROJECT_PATH}/scripts/" \
    "./scripts/"

# Step 5: Pop stashed changes back
echo -e "${YELLOW}Step 5: Restoring local changes...${NC}"
git stash pop 2>/dev/null || echo "No stashed changes to restore"

# Step 6: Summary
echo ""
echo -e "${GREEN}=== Sync Complete! ===${NC}"
echo ""
echo "What was synced:"
echo "  ✓ Latest code from GitHub"
echo "  ✓ Experiment configs (config.yaml)"
echo "  ✓ Final trained models (final_model.zip, best_model.zip)"
echo "  ✓ Normalization stats (vec_normalize.pkl)"
echo "  ✓ Evaluation results (eval/)"
echo "  ✓ Any generated videos or plots"
echo ""
echo "What was NOT synced:"
echo "  ✗ W&B run files (.wandb)"
echo "  ✗ Checkpoints (intermediate saves)"
echo "  ✗ TensorBoard logs"
echo "  ✗ Large checkpoint directories"
echo ""
echo -e "${GREEN}You can now run:${NC}"
echo "  python scripts/test_trained_model.py"
echo ""

# Optional: Show latest experiments
echo -e "${YELLOW}Latest experiments found locally:${NC}"
ls -la experiments/ | grep "ppo_" | tail -5

# Show total size synced
echo ""
echo -e "${YELLOW}Total size of experiments:${NC}"
du -sh experiments/ 2>/dev/null || echo "No experiments found"