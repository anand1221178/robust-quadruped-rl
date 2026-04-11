#!/bin/bash
# Submit all v2 multi-seed training jobs to the cluster
# Usage: bash scripts/submit_v2_all.sh
#
# This submits 20 jobs total: 4 models x 5 seeds
# Each job: ~36-47 hours on RTX 8000
# Total compute: ~850 GPU-hours

echo "========================================================"
echo "Submitting V2 Multi-Seed Training Jobs"
echo "4 models x 5 seeds = 20 jobs"
echo "========================================================"

SEEDS=(42 123 456 789 999)
MODELS=(M1_baseline_v2 M2_sr2l_v2 M3_dr_v2 M4_combo_v2)

COUNT=0
for MODEL in "${MODELS[@]}"; do
    echo -e "\n--- ${MODEL} ---"
    for SEED in "${SEEDS[@]}"; do
        CONFIG="configs/experiments_v2/multiseed/${MODEL}_seed${SEED}.yaml"
        if [ -f "$CONFIG" ]; then
            JOB_NAME="${MODEL}_s${SEED}"
            echo "  Submitting: ${JOB_NAME}"
            sbatch --job-name="${JOB_NAME}" scripts/train_v2_cluster.sh "${CONFIG}"
            COUNT=$((COUNT + 1))
        else
            echo "  WARNING: Config not found: ${CONFIG}"
            echo "  Run: python scripts/generate_v2_multiseed_configs.py"
        fi
    done
done

echo -e "\n========================================================"
echo "Submitted ${COUNT} jobs"
echo "Monitor with: squeue -u \$USER"
echo "========================================================"
