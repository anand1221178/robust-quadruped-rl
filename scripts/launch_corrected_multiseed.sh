#!/bin/bash
#
# Launch all 10 CORRECTED multiseed experiments (M3 + M4)
# Total: 5 seeds × 2 models = 10 training runs
# Estimated time: ~36 hours per run = 360 GPU-hours total
#
# Usage: bash scripts/launch_corrected_multiseed.sh

set -e

echo "🚀 LAUNCHING CORRECTED MULTISEED EXPERIMENTS 🚀"
echo "================================================"
echo ""
echo "Models: M3_dr_corrected (5 seeds) + M4_combo_corrected (5 seeds)"
echo "Total runs: 10"
echo "Estimated time: ~36 hours per run"
echo ""
echo "FIXES APPLIED:"
echo "   Correct parameter names (joint_dropout_prob, max_dropped_joints)"
echo "   Phase configs at top level (not nested)"
echo "   Proper CurriculumDRWrapper selection"
echo "   3-phase curriculum: Clean → Single failures → Multiple failures"
echo ""

# Check if train_ppo_cluster.sh exists
if [ ! -f "scripts/train_ppo_cluster.sh" ]; then
    echo "❌ Error: scripts/train_ppo_cluster.sh not found!"
    exit 1
fi

# Track submitted jobs
declare -a job_ids=()

echo "📋 SUBMITTING M3 (DR with curriculum) - 5 seeds"
echo "-----------------------------------------------"
for seed in 42 123 456 789 999; do
    config="M3_dr_corrected_seed${seed}"
    echo "  Submitting: ${config}"
    job_id=$(sbatch --parsable scripts/train_ppo_cluster.sh ${config})
    job_ids+=("${job_id}")
    echo "    ✓ Job ID: ${job_id}"
    sleep 1  # Avoid overwhelming scheduler
done

echo ""
echo "📋 SUBMITTING M4 (SR2L + DR with curriculum) - 5 seeds"
echo "------------------------------------------------------"
for seed in 42 123 456 789 999; do
    config="M4_combo_corrected_seed${seed}"
    echo "  Submitting: ${config}"
    job_id=$(sbatch --parsable scripts/train_ppo_cluster.sh ${config})
    job_ids+=("${job_id}")
    echo "    ✓ Job ID: ${job_id}"
    sleep 1  # Avoid overwhelming scheduler
done

echo ""
echo " ALL 10 JOBS SUBMITTED SUCCESSFULLY!"
echo "========================================"
echo ""
echo "Job IDs: ${job_ids[@]}"
echo ""
echo "📊 Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f logs/slurm-JOBID.out"
echo ""
echo "⏱️  Estimated completion: ~36 hours (if all run in parallel)"
echo "💾 Expected output: experiments/M3_dr_corrected_seed*/final_model.zip"
echo "                    experiments/M4_combo_corrected_seed*/final_model.zip"
echo ""
echo "🎯 These corrected models will have:"
echo "  • Actual 3-phase curriculum progression"
echo "  • Proper joint failure rates (0% → 10% → 20%)"
echo "  • CurriculumDRWrapper properly applied"
echo "  • Logged curriculum_phase will show 1 → 2 → 3 progression"
echo ""
echo "Good luck! 🍀"
