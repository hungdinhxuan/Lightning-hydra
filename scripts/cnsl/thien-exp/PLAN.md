# Training Bottleneck Investigation Plan

## 1) Goals

- Identify the slowest component in the training pipeline (`data loading`, `GPU compute`, `validation`, `logging`, `I/O`).
- Collect quantitative evidence (`time per step`, `GPU utilization`, `data wait time`) instead of relying on intuition.
- Select the top 2-3 optimizations with the highest expected impact and validate them with short A/B runs.

## 2) Scope

- Main script: `scripts/cnsl/thien-exp/2_debug.sh`
- Main config: `configs/experiment/thien_exp/xlsr_conformertcm_mdt_lora.yaml`
- Do not change the model architecture during the bottleneck isolation phase.

## 3) KPIs and Decision Thresholds

- `step_time_ms`: average time per training step.
- `gpu_util_%`: average GPU utilization from `nvidia-smi dmon`.
- `gpu_mem_GB`: GPU memory usage.
- `data_wait_ratio`: fraction of time spent waiting for data, if available from the profiler.
- `val_overhead_%`: fraction of total wall-clock time spent in validation.

Heuristics for bottleneck classification:
- `gpu_util_% < 70%` with unstable `step_time` suggests a data/input bottleneck.
- High `gpu_util_%`, high `gpu_mem_GB`, and stable `step_time` suggest a compute/model bottleneck.
- A large wall-clock increase when decreasing `val_check_interval` suggests a validation bottleneck.

## 4) Execution Principles

- Run in two phases: `dry-run profiling` followed by `confirmation on a longer run`.
- Change only one variable at a time to avoid confounding effects.
- Save the full command, logs, and metric snapshots for every test.

## 5) Phase A - Baseline (Required)

Goal: capture a clean performance snapshot of the current setup.

1. Run the baseline for 300-500 steps using the current script:
   - `bash scripts/cnsl/thien-exp/2_debug.sh`
2. Collect system metrics in parallel:
   - `nvidia-smi dmon -s pucvmt -d 1`
   - CPU/RAM/I/O if needed: `pidstat -durh 1 -p <train_pid>`
3. Record:
   - time per step from the training CSV log
   - validation duration for each evaluation event
   - throughput in samples per second
4. Save artifacts to a timestamped directory, for example:
   - `outputs/profiling/<YYYYMMDD_HHMM>/baseline/`

Deliverable:
- One baseline metric table plus the top 3 bottleneck hypotheses.

## 6) Phase B - Bottleneck Isolation (Micro-Experiments)

### B1. Check for a DataLoader bottleneck

Run the following sequentially for 150-300 steps each:
- Test W1: `num_workers=8`
- Test W2: `num_workers=16` (baseline)
- Test W3: `num_workers=24` if CPU capacity allows
- Test P1: `pin_memory=false`
- Test P2: `pin_memory=true` (baseline)
- Test T1: `OMP_NUM_THREADS=8` vs `16` (baseline)

Conclude a DataLoader bottleneck if:
- Increasing `num_workers` or adjusting CPU threads reduces `step_time` by more than 8-10%.

### B2. Check for a Validation bottleneck

The current script uses:
- `++trainer.val_check_interval=0.5`

Run:
- V1: `val_check_interval=1.0`
- V2: `val_check_interval=0.25`

Compare:
- wall-clock time for the same number of training steps
- percentage of total time spent in validation

Conclude a Validation bottleneck if:
- Changing `val_check_interval` significantly changes total wall-clock time while training-step time stays similar.

### B3. Check for Logging or Checkpoint overhead

Run:
- L1: keep `logger=csv` (baseline debug mode)
- L2: disable non-essential logging and keep only minimal metrics

If the framework allows it, reduce step-level logging frequency as well.

Conclude a Logging bottleneck if:
- Reducing or disabling logging clearly decreases `step_time`.

### B4. Check for a Compute bottleneck

Run:
- C1: batch size 16 (baseline)
- C2: batch size 20 or 24 if VRAM allows
- C3: a smaller batch size such as 12 for comparison

Evaluate:
- how throughput and `gpu_util_%` change as batch size changes

Conclude a Compute bottleneck if:
- GPU utilization stays high, throughput scales with batch size until near the VRAM limit, and data wait is low.

## 7) Phase C - Deep Profiling (Only if Needed)

If Phase B is still inconclusive:
- Use `torch.profiler` for 50-100 steps to break down:
  - DataLoader wait
  - forward/backward time
  - optimizer step
  - validation hooks
- Optionally run `nsys profile` on a short run to inspect the kernel timeline.

Deliverable:
- Top 5 operators or blocks by total runtime, plus targeted optimization ideas.

## 8) Optimization Priority After the Bottleneck Is Found

1. Data path:
   - Tune `num_workers`, `OMP_NUM_THREADS`, and `pin_memory`.
   - Verify that the dataset under `/dev/shm` is being used consistently and not falling back to disk I/O.
2. Validation:
   - Adjust `val_check_interval`, or move validation to epoch boundaries if acceptable.
3. Compute:
   - Increase batch size up to a stable VRAM limit.
   - Consider gradient accumulation if you need to preserve the effective batch size.
4. Logging:
   - Reduce logging and checkpoint frequency during the dry-debug phase.

## 9) Artifacts to Save for Every Run

- Full launch command with overrides.
- Effective merged config.
- Training logs (CSV, TensorBoard, or W&B if used).
- `nvidia-smi dmon` snapshots.
- A summary table of key metrics (`step_time`, throughput, utilization, validation overhead).

## 10) Definition of Done

This plan is complete when:
- There is a baseline comparison table plus at least 4 micro-tests covering DataLoader, Validation, Compute, and Logging.
- The primary bottleneck, or top 2 bottlenecks, is identified using metrics.
- There is a prioritized optimization list with expected gains in percent.
- At least one post-optimization confirmation run shows measurable improvement.
