#!/bin/bash
#SBATCH --job-name=layer2-muon-sam-wd01
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Layer-2 SAM sanity at full 125M scale: Muon + SAM (rho=0.05) @ WD=0.1, 2 epochs.
# Matched counterpart to run_adamw_sam_wd01.sh. Higher-risk of the two: SAM's
# perturb-step happens BEFORE Muon's orthogonalization on the second pass, so the
# perturbed-point gradient still goes through Newton-Schulz/Polar-Express. The
# 20M smoke test passed cleanly (val 5.281 vs Muon baseline 5.274, no NaN), but
# divergence is more likely with longer training and a non-trivial Muon update.
#
# If this run NaNs or stagnates while AdamW+SAM works: switch to per-param-group
# SAM (split rho/grad-norm per group) or scale rho down for the Muon group.
# Document the result either way — it's a finding.
#
# Walltime budget 2.5h gives generous headroom past the ~70-min estimate.

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
  --optimizer muon \
  --use-sam --sam-rho 0.05 \
  --run "layer2-muon-sam-wd01-2ep-${SLURM_JOB_ID}"

echo "Finished: $(date -Iseconds)"
