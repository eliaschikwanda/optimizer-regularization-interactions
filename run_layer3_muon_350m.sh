#!/bin/bash
#SBATCH --job-name=layer3-muon-350m
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Layer-3 scaling: Muon @ 350M params at Muon's empirical best (WD, dropout)
# from the Layer-2 sweep: WD=0.30, dropout=0.0.
#
# Matched-arch counterpart to run_layer3_adamw_350m.sh — gives the AdamW-vs-Muon
# gap at 350M, the single most important sentence in the paper's "scaling" section.
#
# Arch: n_layer=24, n_head=16, n_embd=1024 (~340M params + ~50M value-embed → ~390M total).
# Seed: 1337 (matched to AdamW counterpart for paired comparison).
# Expected wall-clock: ~95-110 min on L40S.

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
  --weight-decay 0.30 --dropout 0.0 \
  --optimizer muon \
  --seed 1337 \
  --run "layer3-muon-350m-${SLURM_JOB_ID}"

echo "Finished: $(date -Iseconds)"
