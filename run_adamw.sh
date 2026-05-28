#!/bin/bash
#SBATCH --job-name=layer1-adamw
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Layer-1 comparison (MILESTONE / 2-epoch): AdamW for matrix params, same regularization
# config as run_muon.sh (WD=1.6, dropout=0.1). Matrix-AdamW LR defaults to 1e-3
# (scaled by --lr_multiplier=0.25 -> 2.5e-4 effective peak).
# Expected wall-clock: ~40 min on L40S.

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

WANDB_MODE=offline python train.py \
  --n_layer 12 --n_head 12 --n_embd 768 \
  --num-epochs 2 \
  --weight-decay 1.6 --dropout 0.1 \
  --optimizer adamw \
  --run "layer1-adamw-2ep-${SLURM_JOB_ID}"

echo "Finished: $(date -Iseconds)"
