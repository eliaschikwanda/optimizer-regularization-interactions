#!/bin/bash
#SBATCH --job-name=layer3-muon-sam-350m
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Layer-3 scaling: Muon + SAM @ 350M params at Muon+SAM's empirical best
# (WD, dropout) from the Layer-2 sweep: WD=0.30, dropout=0.0, rho=0.05.
#
# Highest-risk run of Layer 3: SAM's grad perturbation interacts with Muon's
# orthogonalization on the second pass at a larger scale. The 125M sanity
# (val 3.986, no NaN, ~2.0x wall-clock) suggests it'll behave, but worth
# watching for divergence given the longer training time and bigger gradients.
#
# Cost: SAM doubles per-step wall-clock. Expected ~190-220 min on L40S.
# Walltime budget 5h gives generous slack.

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
  --use-sam --sam-rho 0.05 \
  --seed 1337 \
  --run "layer3-muon-sam-350m-${SLURM_JOB_ID}"

echo "Finished: $(date -Iseconds)"
