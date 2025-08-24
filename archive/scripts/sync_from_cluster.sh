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

# Step 3: Sync experiment results (excluding huge files)
echo -e "${YELLOW}Step 3: Syncing experiment results from cluster...${NC}"
echo "This may take a few minutes..."

# Create experiments directory if it doesn't exist
mkdir -p experiments

# Sync experiments, excluding very large files
rsync -avz --progress \
    --exclude='*.wandb' \
    --exclude='checkpoints/' \
    --exclude='tensorboard/' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='wandb/' \
    "${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_PROJECT_PATH}/experiments/" \
    "./experiments/"

# Step 4: Sync models separately (only .zip files)
echo -e "${YELLOW}Step 4: Syncing trained models...${NC}"
rsync -avz --progress \
    --include='*/' \
    --include='*.zip' \
    --include='*.pkl' \
    --exclude='*' \
    "${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_PROJECT_PATH}/experiments/" \
    "./experiments/"

# Step 5: Sync any new scripts
echo -e "${YELLOW}Step 5: Syncing scripts...${NC}"
rsync -avz --progress \
    "${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_PROJECT_PATH}/scripts/" \
    "./scripts/"

# Step 6: Pop stashed changes back
echo -e "${YELLOW}Step 6: Restoring local changes...${NC}"
git stash pop 2>/dev/null || echo "No stashed changes to restore"

# Step 7: Summary
echo ""
echo -e "${GREEN}=== Sync Complete! ===${NC}"
echo ""
echo "What was synced:"
echo "  ✓ Latest code from GitHub"
echo "  ✓ Experiment configs and results" 
echo "  ✓ Trained models (.zip files)"
echo "  ✓ Scripts"
echo ""
echo "What was NOT synced (too large):"
echo "  ✗ W&B run files (.wandb)"
echo "  ✗ Checkpoints (intermediate saves)"
echo "  ✗ TensorBoard logs"
echo ""
echo -e "${GREEN}You can now run:${NC}"
echo "  python scripts/test_trained_model.py"
echo ""

# Optional: Show latest experiments
echo -e "${YELLOW}Latest experiments found:${NC}"
ls -la experiments/ | grep "ppo_" | tail -5