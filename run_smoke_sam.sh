#!/bin/bash
#SBATCH --job-name=smoke-sam
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# SAM convergence smoke test on a 20M-param tiny model.
# Purpose: verify the SAM wrapper (--use-sam, --sam-rho) does not diverge or NaN
# and that val loss decreases monotonically over a short run, BEFORE committing
# any L40S-hours to a full 125M SAM run.
#
# Runs four 1-epoch jobs back-to-back at the smoke-test arch (n_layer=2, n_head=2,
# n_embd=128) so each finishes in ~3-5 min. Total wall-clock: ~20 min.
#
# Acceptance criteria (manual check):
#   - All four runs print "Min val Loss" < initial val loss (~10.83)
#   - SAM runs are not catastrophically worse than non-SAM at this scale
#   - No NaN, no exception, exit code 0
#   - SAM per-step dt is ~2x non-SAM per-step dt (sanity-check that SAM actually
#     does the second forward+backward)

set -euo pipefail

cd ~/cs229/optimizer-regularization-interactions
mkdir -p slurm_logs

source .venv/bin/activate

echo "=== Job info ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Started: $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "==============="

COMMON=(
  --n_layer 2 --n_head 2 --n_embd 128
  --device-batch-size 4 --total-batch-size 8192
  --num-epochs 1
  --weight-decay 0.1 --dropout 0.1
)

echo ">>> [1/4] AdamW baseline (no SAM)"
WANDB_MODE=offline python train.py "${COMMON[@]}" \
  --optimizer adamw \
  --run "smoke-adamw-${SLURM_JOB_ID}"

echo ">>> [2/4] AdamW + SAM (rho=0.05)"
WANDB_MODE=offline python train.py "${COMMON[@]}" \
  --optimizer adamw --use-sam --sam-rho 0.05 \
  --run "smoke-adamw-sam-${SLURM_JOB_ID}"

echo ">>> [3/4] Muon baseline (no SAM)"
WANDB_MODE=offline python train.py "${COMMON[@]}" \
  --optimizer muon \
  --run "smoke-muon-${SLURM_JOB_ID}"

echo ">>> [4/4] Muon + SAM (rho=0.05)"
WANDB_MODE=offline python train.py "${COMMON[@]}" \
  --optimizer muon --use-sam --sam-rho 0.05 \
  --run "smoke-muon-sam-${SLURM_JOB_ID}"

echo "Finished: $(date -Iseconds)"
