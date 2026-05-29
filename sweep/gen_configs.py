#!/usr/bin/env python3
"""Generate sweep/configs.csv for the Layer-2 SLURM array job.

Grid: 4 optimizer configs x 4 WD x 2 dropout x 3 seeds = 96 runs.

Each row: idx,name,optimizer,use_sam,sam_rho,wd,dropout,seed

run_sweep.sh reads this CSV indexed by $SLURM_ARRAY_TASK_ID (1-based, matching
sed -n "${TASK}p" which skips the header).

Re-run any time you change the grid:
    python sweep/gen_configs.py > sweep/configs.csv
"""

import sys
import itertools

OPTIMIZERS = [
    # (label, --optimizer, --use-sam flag)
    ("adamw",     "adamw", False),
    ("muon",      "muon",  False),
    ("adamw-sam", "adamw", True),
    ("muon-sam",  "muon",  True),
]
WDS      = [0.03, 0.1, 0.3, 1.0]
DROPOUTS = [0.0, 0.1]
SEEDS    = [42, 1337, 2718]
SAM_RHO  = 0.05

def main():
    print("idx,name,optimizer,use_sam,sam_rho,wd,dropout,seed")
    idx = 0
    for (label, opt, use_sam), wd, dr, seed in itertools.product(
        OPTIMIZERS, WDS, DROPOUTS, SEEDS,
    ):
        idx += 1
        wd_tag = f"wd{str(wd).replace('.', '')}"
        dr_tag = f"dr{str(dr).replace('.', '')}"
        name = f"sweep-{label}-{wd_tag}-{dr_tag}-s{seed}"
        print(f"{idx},{name},{opt},{int(use_sam)},{SAM_RHO},{wd},{dr},{seed}")

if __name__ == "__main__":
    main()
