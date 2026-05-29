#!/bin/bash
#SBATCH --job-name=l2-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-96%6
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

# Layer-2 sweep: 96 runs over (optimizer x WD x dropout x seed).
# One SLURM array job, one cell per array task. Reads sweep/configs.csv to
# get the config for $SLURM_ARRAY_TASK_ID.
#
# --array=1-96%6 caps concurrent tasks at 6 (one per available oat GPU).
# Adjust the cap if FarmShare is empty/busy: %4 = polite, %12 = greedy.
#
# Wall-clock estimate:
#   Non-SAM cells (48 runs): ~35 min each
#   SAM cells     (48 runs): ~70 min each
#   Total compute: 48*35 + 48*70 = 5040 GPU-min = 84 GPU-hours
#   With 6 parallel slots: 84/6 = 14 hours real-time. Plan: launch Friday night.
#
# Time limit per task: 2.5h gives slack past the 70-min SAM cost.
#
# Failed tasks can be resubmitted individually:
#   sbatch --array=37,42,55 sweep/run_sweep.sh

set -euo pipefail

cd ~/cs229/optimizer-regularization-interactions
mkdir -p slurm_logs sweep/results

source .venv/bin/activate

TASK=${SLURM_ARRAY_TASK_ID}
CSV=sweep/configs.csv

# sed -n "$((TASK+1))p" skips the header row, so config rows are 1-indexed
ROW=$(sed -n "$((TASK+1))p" "$CSV")
if [[ -z "$ROW" ]]; then
  echo "ERROR: no row for task $TASK in $CSV"
  exit 1
fi

# Parse CSV (idx,name,optimizer,use_sam,sam_rho,wd,dropout,seed)
IFS=',' read -r IDX NAME OPT USE_SAM SAM_RHO WD DROPOUT SEED <<< "$ROW"

# Sanity check that the IDX in the row matches the TASK we asked for
if [[ "$IDX" != "$TASK" ]]; then
  echo "ERROR: row idx=$IDX but task=$TASK — CSV out of sync"
  exit 1
fi

SAM_ARGS=""
if [[ "$USE_SAM" == "1" ]]; then
  SAM_ARGS="--use-sam --sam-rho ${SAM_RHO}"
fi

echo "=== Sweep task ${TASK}/96 ==="
echo "Job ID:    ${SLURM_JOB_ID} (array ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})"
echo "Node:      $(hostname)"
echo "Run name:  ${NAME}"
echo "Config:    opt=${OPT} use_sam=${USE_SAM} rho=${SAM_RHO} wd=${WD} dropout=${DROPOUT} seed=${SEED}"
echo "Started:   $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "==============="

WANDB_MODE=offline python train.py \
  --n_layer 12 --n_head 12 --n_embd 768 \
  --num-epochs 2 \
  --weight-decay "${WD}" --dropout "${DROPOUT}" \
  --optimizer "${OPT}" \
  --seed "${SEED}" \
  ${SAM_ARGS} \
  --run "${NAME}-${SLURM_JOB_ID}"

# Append the final val loss to a sweep-wide results file (atomic-ish)
LAST_LOG="slurm_logs/l2-sweep-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
VAL_LOSS=$(grep "Min val Loss" "${LAST_LOG}" | tail -1 | awk '{print $NF}')
VAL_BPB=$(grep "Min val BPB" "${LAST_LOG}" | tail -1 | awk '{print $NF}')
TRAIN_TIME=$(grep "Total training time" "${LAST_LOG}" | tail -1 | awk '{print $NF}')
echo "${IDX},${NAME},${OPT},${USE_SAM},${WD},${DROPOUT},${SEED},${VAL_LOSS},${VAL_BPB},${TRAIN_TIME}" \
  >> sweep/results/summary.csv

echo "Finished:  $(date -Iseconds)"
