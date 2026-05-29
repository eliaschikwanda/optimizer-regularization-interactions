# Layer-1 2x2: optimizer x weight_decay, 125M params, 2 epochs, 100M FineWeb tokens

| optimizer | wd  | val_loss | val_bpb | train_time | job_id  |
|-----------|-----|----------|---------|------------|---------|
| AdamW     | 1.6 | 5.024189 | 1.632876| 34.20m     | 1565112 |
| AdamW     | 0.1 | 4.637892 | 1.507347| 33.76m     | 1582290 |
| Muon      | 1.6 | 4.108166 | 1.335155| 34.41m     | (local) |
| Muon      | 0.1 | 3.943272 | 1.281570| 34.23m     | 1582291 |
