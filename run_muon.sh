#!/bin/bash
#SBATCH --job-name=layer1-muon
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Layer-1 baseline (MILESTONE / 2-epoch): Muon for matrix params, at slowrun record-#1's
# recommended hyperparameters (WD=1.6, dropout=0.1, n_layer=12, n_head=12, n_embd=768).
# Reduced to 2 epochs for milestone deadline. Full 12-epoch run is a separate followup.
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
  --optimizer muon \
  --run "layer1-muon-2ep-${SLURM_JOB_ID}"

echo "Finished: $(date -Iseconds)"
