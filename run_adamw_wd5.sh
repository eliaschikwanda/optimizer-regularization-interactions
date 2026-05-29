#!/bin/bash
#SBATCH --job-name=layer1-adamw-wd5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Layer-1 WD-curve probe: AdamW @ WD=0.5 — fills in the middle of the AdamW WD scan
# between 0.1 (val 4.638) and 1.6 (val 5.024). Combined with WD={0.03, 0.1, 1.6} this
# gives 4 points to locate AdamW's optimal regularization at 125M / 2 epochs.

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
  --weight-decay 0.5 --dropout 0.1 \
  --optimizer adamw \
  --run "layer1-adamw-wd5-2ep-${SLURM_JOB_ID}"

echo "Finished: $(date -Iseconds)"
