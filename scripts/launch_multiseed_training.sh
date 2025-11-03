#!/bin/bash
#
# Launch all 20 multi-seed training runs
# 4 models × 5 seeds = 20 jobs
#
# Estimated time: ~40 hours per run = ~800 GPU-hours total
#

echo "========================================================"
echo "MULTI-SEED TRAINING LAUNCH"
echo "========================================================"
echo "This will submit 20 training jobs to the cluster:"
echo "  - M1 (Baseline) × 5 seeds"
echo "  - M2 (SR2L) × 5 seeds"
echo "  - M3 (DR) × 5 seeds"
echo "  - M4 (Combined) × 5 seeds"
echo ""
echo "Estimated compute: ~800 GPU-hours"
echo "========================================================"
echo ""

# Change to project root
cd "$(dirname "$0")/.." || exit 1

# Models and seeds
MODELS=("M1_baseline" "M2_sr2l" "M3_dr" "M4_combo")
SEEDS=(42 123 456 789 999)

SUBMITTED_JOBS=0
TOTAL_JOBS=$((${#MODELS[@]} * ${#SEEDS[@]}))

echo "📋 Submitting $TOTAL_JOBS jobs..."
echo ""

# Submit all jobs
for model in "${MODELS[@]}"; do
    echo "Starting $model runs..."

    for seed in "${SEEDS[@]}"; do
        CONFIG_NAME="${model}_seed${seed}"

        # Check if config exists
        if [ ! -f "configs/multiseed_experiments/${CONFIG_NAME}.yaml" ]; then
            echo "  ⚠️  Config not found: ${CONFIG_NAME}.yaml - skipping"
            continue
        fi

        # Submit job
        JOB_ID=$(sbatch \
            --job-name="${CONFIG_NAME}" \
            --output="logs/multiseed/${CONFIG_NAME}_%j.out" \
            --error="logs/multiseed/${CONFIG_NAME}_%j.err" \
            scripts/train_ppo_cluster_multiseed.sh "${CONFIG_NAME}" \
            | awk '{print $4}')

        if [ $? -eq 0 ]; then
            echo "  ✅ Submitted: ${CONFIG_NAME} (Job ID: ${JOB_ID})"
            SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
        else
            echo "  ❌ Failed: ${CONFIG_NAME}"
        fi

        # Small delay to avoid overwhelming scheduler
        sleep 0.5
    done

    echo ""
done

echo "========================================================"
echo "SUBMISSION COMPLETE"
echo "========================================================"
echo "Jobs submitted: $SUBMITTED_JOBS / $TOTAL_JOBS"
echo ""
echo "📊 Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f logs/multiseed/*.out"
echo ""
echo "🔬 Track on W&B:"
echo "  https://wandb.ai/anandpatel1221178-university-of-the-witswatersrand/robust-quadruped-rl-multiseed"
echo ""
echo "⏱️  Estimated completion:"
echo "  ~40 hours (if runs happen in parallel)"
echo "  ~800 hours (if sequential)"
echo "========================================================"

# Create tracking file
mkdir -p logs/multiseed
echo "Launched $SUBMITTED_JOBS jobs at $(date)" > logs/multiseed/launch_timestamp.txt
echo "Models: ${MODELS[*]}" >> logs/multiseed/launch_timestamp.txt
echo "Seeds: ${SEEDS[*]}" >> logs/multiseed/launch_timestamp.txt
