#!/bin/bash
#SBATCH --job-name=layer1-adamw-wd01
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Layer-1 retuned-WD probe: AdamW with weight_decay=0.1 (vs. 1.6 baseline).
# Tests Kim et al.'s prediction that AdamW wants much lower WD than slowrun's default,
# and whether retuning closes the Muon-vs-AdamW gap at WD=1.6 (Muon 4.108 vs AdamW 5.024).
# Same arch / dropout / batch / epochs as run_adamw.sh — only WD differs.
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
  --weight-decay 0.1 --dropout 0.1 \
  --optimizer adamw \
  --run "layer1-adamw-wd01-2ep-${SLURM_JOB_ID}"

echo "Finished: $(date -Iseconds)"
