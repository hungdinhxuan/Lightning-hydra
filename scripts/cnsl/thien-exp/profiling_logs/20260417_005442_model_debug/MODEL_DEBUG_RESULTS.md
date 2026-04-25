## Model debug (ordered) — 2026-04-17 00:54:42

### Scope and order

We followed an ordered model-debug flow with fixed data-loader settings:

- Fixed: `NUM_WORKERS=24`, `PIN_MEMORY=false`, `BATCH_SIZE=16`
- Fixed: `LIMIT_TRAIN_BATCHES=100`, `LIMIT_VAL_BATCHES=20`, `VAL_CHECK_INTERVAL=1.0`
- Ordered runs:
  1. Baseline (`bf16-mixed`, aug `[RawBoost12, none]`)
  2. No augmentation (`bf16-mixed`, aug `["none"]`)
  3. Precision ablation (`fp32` via `++trainer.precision=32`)

Logs:

- `01_baseline.log`
- `02_no_aug.log`
- `03_fp32.log`

### Key profiler metrics

All times are seconds.

| Run | `run_training_batch` mean | `optimizer_step` mean | `training_step` mean | `backward` mean | `train_dataloader_next` mean | `val_next` mean |
|---|---:|---:|---:|---:|---:|---:|
| 01 baseline (bf16 + aug) | 0.36919 | 0.36909 | 0.18648 | 0.17755 | 0.075779 | 0.42888 |
| 02 no-aug (bf16) | 0.37057 | 0.37047 | 0.18722 | 0.17849 | 0.071630 | 0.45708 |
| 03 fp32 | 0.45787 | 0.45776 | 0.20737 | 0.24588 | 0.070012 | 0.50730 |

### What this says about the model

- **No-aug vs baseline is almost identical on compute path** (`run_training_batch`, `optimizer_step`, `training_step`, `backward`), so augmentation is not the primary driver of current slowdown.
- **FP32 is much slower than bf16**:
  - `run_training_batch`: `0.36919 -> 0.45787` (**+24.0%**)
  - `backward`: `0.17755 -> 0.24588` (**+38.5%**)
  - `training_step`: `0.18648 -> 0.20737` (**+11.2%**)
- `train_dataloader_next` remains around `~0.07s` across runs, while compute-side metrics move a lot with precision.

### Conclusion

The dominant bottleneck in current settings is **model compute path** (forward/backward/optimizer), not data augmentation logic.

### Recommended next steps (model-focused)

1. Keep `bf16-mixed` for training/debug (`fp32` clearly regresses speed).
2. Profile operator-level hotspots inside model forward/backward (XLS-R + conformer blocks) using a short `torch.profiler` trace (50-100 steps).
3. Then run one structural ablation on model complexity (for example fewer conformer encoders or smaller `emb_size`) to measure speed/accuracy tradeoff directly.

## Torch profiler step (100 steps)

Run:

- `04_torch_profiler.log` with override `trainer.profiler=pytorch`
- Settings kept the same as baseline (`NUM_WORKERS=24`, `PIN_MEMORY=false`, `BATCH_SIZE=16`, `bf16-mixed`)

### Hot blocks/layers (from `FIT Profiler Report`)

- `fairseq wav2vec2` path dominates module-level CPU total:
  - `Wav2Vec2...`: `CPU total 458.644ms` (`63.64%`)
  - `Transformer...`: `CPU total 423.397ms` (`58.75%`)
- Front-end + linear-heavy math is prominent:
  - `aten::linear`: `CPU total 234.435ms`, `CUDA total 178.971ms`
  - `aten::addmm`: `CUDA total 71.462ms`
  - `aten::cudnn_convolution`: `CUDA total 41.097ms`
  - `aten::layer_norm`: `CUDA total 35.804ms`
- Data/type/copy overhead is non-trivial:
  - `aten::copy_`: `CUDA 87.247ms` (`33.48%` of self CUDA in table row)
  - `aten::to` / `aten::_to_copy`: `CUDA total 56.957ms`

### Torch profiler totals

- `Self CPU time total: 720.699ms`
- `Self CUDA time total: 260.439ms`

### Interpretation update

- The hotspot is now clearly localized to the **XLS-R (wav2vec2) encoder stack** and its dense ops (`linear/addmm/layer_norm`) during forward/backward.
- `aten::copy_` and `to/_to_copy` appearing high suggests there is also meaningful tensor movement/casting overhead in the step path.
- This confirms earlier conclusion: bottleneck is in model compute path, not primarily in data loading/augmentation.

### Next model-debug actions (targeted)

1. Keep `bf16-mixed`; do not switch to fp32 for training speed.
2. Inspect and reduce avoidable tensor casts/copies in model forward path (especially around feature extraction and view handling).

## Speed-up trial: steps 1, 2, 3

Requested trial:

1. Reduce copy/cast overhead in model path.
2. Keep `bf16-mixed`.
3. Try `torch.compile`.

### Code changes applied

- Step 1:
  - Removed `x.requires_grad = True` from `src/models/components/xlsr_conformertcm_baseline.py` forward.
  - Avoided unconditional cast in `src/models/base/mdt_module.py`:
    - now cast to float32 only when `x.dtype != torch.float32`.
- Step 2:
  - Kept `+trainer.precision=bf16-mixed` unchanged in runs.
- Step 3:
  - Added `model.compile_model` flag and compile hook in `src/models/v2/xlsr_conformertcm_mdt_module.py`.
  - Compile is applied in `on_fit_start` (after adapter/checkpoint load) to avoid state-dict key mismatch.

### Benchmark runs (same setup: W24, pin=false, BS16, 100 train steps)

| Run | Meaning | `run_training_batch` mean | `optimizer_step` mean | `train_dataloader_next` mean |
|---|---|---:|---:|---:|
| `01_baseline.log` | old baseline (before step-1 edits) | 0.36919s | 0.36909s | 0.075779s |
| `05_postfix_bf16.log` | after step-1 edits + bf16 | **0.33196s** | **0.33185s** | 0.087654s |
| `06_postfix_bf16_compile.log` | step-1 edits + bf16 + compile | 9.4351s | 9.4350s | 0.061668s |

### Outcome

- Step 1 + Step 2 helped training-step speed:
  - `run_training_batch`: `0.36919 -> 0.33196` (**~10.1% faster**).
- Step 3 (`torch.compile`) is **not suitable** for this current workload:
  - `run_training_batch` exploded to `9.4351s` due heavy compile/runtime overhead (very likely dynamic-shape recompilation behavior).
  - Therefore, for now keep `compile_model: false`.

### Recommendation after this trial

- Keep the step-1 edits and continue with `bf16-mixed`.
- Do not enable `torch.compile` for this model/config unless we first constrain dynamic shapes and verify compile cache hit behavior.

