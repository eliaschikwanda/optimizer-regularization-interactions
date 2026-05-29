#!/bin/bash
#SBATCH --job-name=layer2-adamw-sam-wd01
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Layer-2 SAM sanity at full 125M scale: AdamW + SAM (rho=0.05) @ WD=0.1, 2 epochs.
# Purpose: verify SAM behaves the same at 125M as it did in the 20M smoke test
# BEFORE committing to the 96-run Layer-2 sweep. Specifically:
#   - convergence (val loss < AdamW baseline of 4.638, or at least within noise)
#   - no NaN / divergence on the longer run
#   - wall-clock ~2x non-SAM AdamW@WD=0.1 (~34 min) → expect ~68 min
# If this lands cleanly, the sweep is safe to launch. If it diverges or stagnates,
# investigate before burning ~64 GPU-hours on the full grid.
#
# Walltime budget 2.5h gives generous headroom past the 70-min estimate.

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
  --weight-decay 0.1 --dropout 0.1 \
  --optimizer adamw \
  --use-sam --sam-rho 0.05 \
  --run "layer2-adamw-sam-wd01-2ep-${SLURM_JOB_ID}"

echo "Finished: $(date -Iseconds)"
