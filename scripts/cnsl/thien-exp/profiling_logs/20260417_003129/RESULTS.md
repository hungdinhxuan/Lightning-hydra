## Sweep results (`/dev/shm` dataset) — 2026-04-17 00:31:29

### Setup

- Script: `scripts/cnsl/thien-exp/2_debug.sh`
- Fixed knobs (defaults):
  - `LIMIT_TRAIN_BATCHES=100`, `LIMIT_VAL_BATCHES=20`, `VAL_CHECK_INTERVAL=1.0`
  - `BATCH_SIZE=16`, `OMP_THREADS=16`
- Swept knobs:
  - `NUM_WORKERS ∈ {8,16,24}`
  - `PIN_MEMORY ∈ {true,false}`
- Raw logs are in this folder:
  - `W08_pinT.log`, `W16_pinT.log`, `W24_pinT.log`, `W16_pinF.log`, `W24_pinF.log`

### Key numbers from Lightning `FIT Profiler Report`

All times are seconds.

| Run | `run_training_batch` mean | `run_training_batch` total (100) | `train_dataloader_next` mean | `train_dataloader_next` total (100) | `val_next` mean | `val_next` total (22) | `save_checkpoint` total |
|---|---:|---:|---:|---:|---:|---:|---:|
| W=8, pin=T | 0.36250 | 36.250 | 0.055759 | 5.5759 | 0.28523 | 6.2752 | 5.0720 |
| W=16, pin=T | 0.37286 | 37.286 | 0.062610 | 6.2610 | 0.29439 | 6.4765 | 3.3728 |
| W=24, pin=T | 0.36309 | 36.309 | 0.059776 | 5.9776 | 0.38836 | 8.5438 | 3.3912 |
| W=16, pin=F | 0.38175 | 38.175 | 0.057124 | 5.7124 | 0.32874 | 7.2322 | 3.3486 |
| W=24, pin=F | **0.36041** | **36.041** | 0.062776 | 6.2776 | 0.38636 | 8.5000 | 3.6831 |

### Conclusions (for this sweep)

- **Fastest training loop** in this sweep is **`NUM_WORKERS=24` + `PIN_MEMORY=false`** (`run_training_batch` mean **0.360s**).
- **DataLoader wait (`train_dataloader_next`) did not monotonically improve** with more workers or with `pin_memory=true`.
  - Best `train_dataloader_next` in this sweep is **W=8, pin=T** (mean **0.0558s**), but that run is not the fastest overall on `run_training_batch`.
- Validation (`val_next`) varies quite a bit between runs (likely noise / caching / scheduling effects), so for next iteration you may want to reduce validation noise when comparing pure step-time.

### Recommended next action

- Use this as the default for follow-up compute-bound checks:
  - `NUM_WORKERS=24 PIN_MEMORY=false bash scripts/cnsl/thien-exp/2_debug.sh`

