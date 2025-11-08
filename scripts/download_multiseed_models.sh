#!/bin/bash
# Download all 20 completed multi-seed training runs from cluster
# Uses the newer Nov 4 runs (the Nov 3 runs hit job limits and were restarted)

REMOTE_USER="panand"
REMOTE_HOST="146.141.21.100"
REMOTE_BASE="/home-mscluster/panand/Research\ Proj/robust-quadruped-rl/experiments"
LOCAL_BASE="./experiments"

echo "=== Downloading M1 (Baseline) - 5 seeds ==="
rsync -avz --progress --exclude='wandb/' --exclude='*.wandb' --exclude='tensorboard/' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/M1_baseline_seed42_rap72nn2" "${LOCAL_BASE}/"

rsync -avz --progress --exclude='wandb/' --exclude='*.wandb' --exclude='tensorboard/' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/M1_baseline_seed123_ef86brh7" "${LOCAL_BASE}/"

rsync -avz --progress --exclude='wandb/' --exclude='*.wandb' --exclude='tensorboard/' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/M1_baseline_seed456_s39mzd6e" "${LOCAL_BASE}/"

rsync -avz --progress --exclude='wandb/' --exclude='*.wandb' --exclude='tensorboard/' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/M1_baseline_seed789_b6pielh9" "${LOCAL_BASE}/"

rsync -avz --progress --exclude='wandb/' --exclude='*.wandb' --exclude='tensorboard/' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/M1_baseline_seed999_uzo3vstk" "${LOCAL_BASE}/"

echo ""
echo "=== Downloading M2 (SR2L) - 5 seeds ==="
rsync -avz --progress --exclude='wandb/' --exclude='*.wandb' --exclude='tensorboard/' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/M2_sr2l_seed42_8qknaqel" "${LOCAL_BASE}/"

rsync -avz --progress --exclude='wandb/' --exclude='*.wandb' --exclude='tensorboard/' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/M2_sr2l_seed123_wgn4flak" "${LOCAL_BASE}/"

rsync -avz --progress --exclude='wandb/' --exclude='*.wandb' --exclude='tensorboard/' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/M2_sr2l_seed456_qjwh15rq" "${LOCAL_BASE}/"

rsync -avz --progress --exclude='wandb/' --exclude='*.wandb' --exclude='tensorboard/' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/M2_sr2l_seed789_g9loing3" "${LOCAL_BASE}/"

rsync -avz --progress --exclude='wandb/' --exclude='*.wandb' --exclude='tensorboard/' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/M2_sr2l_seed999_p977kpg4" "${LOCAL_BASE}/"

echo ""
echo "=== Downloading M3 (DR) - 5 seeds (Nov 4 completions) ==="
rsync -avz --progress --exclude='wandb/' --exclude='*.wandb' --exclude='tensorboard/' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/M3_dr_seed42_epnckzy2" "${LOCAL_BASE}/"

rsync -avz --progress --exclude='wandb/' --exclude='*.wandb' --exclude='tensorboard/' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/M3_dr_seed123_yxt6lw5f" "${LOCAL_BASE}/"

rsync -avz --progress --exclude='wandb/' --exclude='*.wandb' --exclude='tensorboard/' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/M3_dr_seed456_ijv5yz5g" "${LOCAL_BASE}/"

rsync -avz --progress --exclude='wandb/' --exclude='*.wandb' --exclude='tensorboard/' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/M3_dr_seed789_howl2uf6" "${LOCAL_BASE}/"

rsync -avz --progress --exclude='wandb/' --exclude='*.wandb' --exclude='tensorboard/' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/M3_dr_seed999_vzzp16pn" "${LOCAL_BASE}/"

echo ""
echo "=== Downloading M4 (Combo) - 5 seeds (Nov 4 completions) ==="
rsync -avz --progress --exclude='wandb/' --exclude='*.wandb' --exclude='tensorboard/' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/M4_combo_seed42_p0n7r1gw" "${LOCAL_BASE}/"

rsync -avz --progress --exclude='wandb/' --exclude='*.wandb' --exclude='tensorboard/' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/M4_combo_seed123_06fqvwf0" "${LOCAL_BASE}/"

rsync -avz --progress --exclude='wandb/' --exclude='*.wandb' --exclude='tensorboard/' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/M4_combo_seed456_osu6bibl" "${LOCAL_BASE}/"

rsync -avz --progress --exclude='wandb/' --exclude='*.wandb' --exclude='tensorboard/' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/M4_combo_seed789_pfcyh0d8" "${LOCAL_BASE}/"

rsync -avz --progress --exclude='wandb/' --exclude='*.wandb' --exclude='tensorboard/' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/M4_combo_seed999_71xtry0k" "${LOCAL_BASE}/"

echo ""
echo "=== Download complete! ==="
echo "Downloaded 20 models (5 seeds × 4 model types)"
echo ""
echo "Next steps:"
echo "1. Run reconstruction script: python paper/reconstruct_multiseed_curves.py"
echo "2. This will generate averaged learning curves with error bands showing seed variance"
echo "3. Update Figure 1 in paper with new multi-seed learning curves"
