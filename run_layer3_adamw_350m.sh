#!/bin/bash
#SBATCH --job-name=layer3-adamw-350m
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Layer-3 scaling: AdamW @ 350M params at AdamW's empirical best (WD, dropout)
# from the Layer-2 sweep: WD=0.03, dropout=0.0.
#
# Arch: n_layer=24, n_head=16, n_embd=1024 (~340M params + ~50M value-embed → ~390M total).
# Tokens: same 100M FineWeb train / 10M val split (compute-constrained, not data-constrained).
# Seed: 1337 (avoids the seed=42 outlier identified at WD=0.03,dropout=0 in the sweep).
#
# Expected wall-clock: ~95-110 min on L40S (2.8x the 125M-2ep run, ~linear in params).
# Walltime budget 3h gives slack for compile time + the bigger model.
#
# Memory: 350M at device_batch=4 fits comfortably on L40S 46GB. If it OOMs, halve
# --device-batch-size and grad-accum will compensate (slight wall-clock hit).

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
  --n_layer 24 --n_head 16 --n_embd 1024 \
  --num-epochs 2 \
  --weight-decay 0.03 --dropout 0.0 \
  --optimizer adamw \
  --seed 1337 \
  --run "layer3-adamw-350m-${SLURM_JOB_ID}"

echo "Finished: $(date -Iseconds)"
