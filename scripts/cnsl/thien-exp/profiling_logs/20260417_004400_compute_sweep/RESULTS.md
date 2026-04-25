## Compute sweep results — 2026-04-17 00:44:00

### Setup

- Script: `scripts/cnsl/thien-exp/2_debug.sh`
- Fixed knobs:
  - `NUM_WORKERS=24`, `PIN_MEMORY=false`
  - `LIMIT_TRAIN_BATCHES=100`, `LIMIT_VAL_BATCHES=20`, `VAL_CHECK_INTERVAL=1.0`
  - `OMP_THREADS=16`
- Swept knob:
  - `BATCH_SIZE ∈ {16, 20, 24}`

### Notes about failed attempts

- Initial `BS20` and `BS24` runs failed due to OOM because they were accidentally launched in parallel and competed for VRAM.
- Valid numbers below are from sequential reruns:
  - `BS20_W24_pinF_rerun.log`
  - `BS24_W24_pinF_rerun.log`

### Key numbers from profiler

All times are seconds.

| Run | `run_training_batch` mean | `run_training_batch` total (100) | `train_dataloader_next` mean | `val_next` mean | `save_checkpoint` total | Approx samples/s (`batch_size / run_training_batch_mean`) |
|---|---:|---:|---:|---:|---:|---:|
| BS=16 | **0.36597** | **36.597** | **0.066758** | **0.38930** | 3.3369 | **43.72** |
| BS=20 (rerun) | 0.42151 | 42.151 | 0.10149 | 0.52929 | 4.8526 | 47.45 |
| BS=24 (rerun) | 0.49834 | 49.834 | 0.089771 | 0.58781 | 2.8608 | 48.16 |

### Interpretation

- Larger batch sizes (`20`, `24`) increase per-step time significantly and also increase `val_next` cost.
- Throughput in samples/s rises with batch size, but not proportionally; latency and validation overhead become much worse.
- For **debug/profiling loops** where responsiveness and stable step-time are preferred, `BS=16` is the best operating point.

### Recommended next step

- Keep:
  - `NUM_WORKERS=24`
  - `PIN_MEMORY=false`
  - `BATCH_SIZE=16` (for debug/profiling)
- Move to validation/checkpoint overhead isolation:
  - compare `VAL_CHECK_INTERVAL=1.0` vs `0.5` with same settings above.

