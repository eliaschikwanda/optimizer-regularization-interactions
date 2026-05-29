# Layer-1: optimizer x weight_decay scan, 125M params, 2 epochs, 100M FineWeb tokens

All runs: dropout=0.1, total_batch=524288 tokens, 1x L40S on FarmShare oat.
Val loss measured on the 10M FineWeb val split (slowrun's pinned split).

| optimizer | wd   | val_loss | val_bpb  | train_time | job_id  |
|-----------|------|----------|----------|------------|---------|
| AdamW     | 0.03 | 4.640952 | 1.508338 | 34.64m     | 1582823 |
| AdamW     | 0.1  | 4.637892 | 1.507347 | 33.76m     | 1582290 |
| AdamW     | 0.5  | 4.806669 | 1.562213 | 34.48m     | 1582824 |
| AdamW     | 1.6  | 5.024189 | 1.632876 | 34.20m     | 1565112 |
| Muon      | 0.1  | 3.943272 | 1.281570 | 34.23m     | 1582291 |
| Muon      | 1.6  | 4.108166 | 1.335155 | 34.41m     | (local) |

## Key numbers

- **AdamW WD curve:** flat in [0.03, 0.1], climbs sharply past 0.5. Minimum at WD=0.1 (4.638). WD=0.03 ties within noise (Δ=0.003).
- **Muon WD curve:** weakly decreasing — 4.108 at WD=1.6 → 3.943 at WD=0.1. Muon prefers lower WD, but is far less sensitive than AdamW (Δ=0.16 vs Δ=0.39 over the same WD range).
- **Best-vs-best gap:** Muon (0.1) − AdamW (0.1) = 3.943 − 4.638 = **−0.695**. Muon's advantage survives WD retuning.
- **Gap at slowrun's recommended WD=1.6:** 4.108 − 5.024 = −0.916. Of that, **24% (0.221) is attributable to WD mistuning of AdamW**; the rest is a real optimizer-level effect.

## Findings

1. **Ranking does not invert.** Muon beats AdamW at every WD value tested, including each optimizer's best.
2. **Kim et al.'s "30× higher WD than standard" does not transfer to this scale.** At 125M / 100M tokens / 2 epochs, AdamW's optimum is the textbook 0.1, not ~3.0.
3. **Slowrun's recommended WD=1.6 is suboptimal for both optimizers at this scale.** It hurts AdamW more than Muon, exaggerating Muon's apparent edge by ~0.22 loss.
4. **Muon's edge is genuine, not a regularization artifact.** The residual 0.70 gap at matched optimal WD is what Layer 2 (SAM) needs to attack.
