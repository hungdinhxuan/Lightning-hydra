# Persistent Benchmark Service Usage

This service keeps one benchmark model loaded inside a long-running worker and runs queued benchmark jobs sequentially. It is additive: the old `scripts/benchmark_py/benchmark.py` command remains the fallback path.

## When to Use

Use the service when running multiple benchmark folders against the same model, adapter, config, and GPU. The first job loads the model; later jobs with the same runtime signature reuse the resident model.

Use the legacy command when debugging a single dataset, comparing output compatibility, or when the worker path fails.

## Start Worker

From repo root:

```bash
python scripts/benchmark_py/service_worker.py \
  --queue-dir .benchmark_service_queue \
  --reload-on-change
```

Recommended long-running tmux session:

```bash
tmux new -s benchmark-worker
python scripts/benchmark_py/service_worker.py \
  --queue-dir .benchmark_service_queue \
  --reload-on-change
```

Options:

- `--queue-dir`: file queue directory. Default: `.benchmark_service_queue`
- `--poll-seconds`: idle polling interval. Default: `5`
- `--reload-on-change`: reload resident model if later job uses a different config/model/adapter/runtime signature
- `--once`: process one queued job and exit
- `--health-path`: explicit health JSON path. Default: `<queue-dir>/health.json`

Without `--reload-on-change`, the worker rejects jobs whose model/config/adapter signature differs from the loaded runtime.

## Submit Job

Submit command mirrors the legacy benchmark CLI:

```bash
python scripts/benchmark_py/service_submit.py \
  --queue-dir .benchmark_service_queue \
  -g 0 \
  -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer \
  -b data/May_2026_benchmark \
  -m /path/to/base_or_merged_model.pth \
  -r logs/results/May_2026_benchmark \
  -n 12May2026_lora_from_29April26_xlsr_conformertcm_mdt-conf-3 \
  -a /path/to/adapter/checkpoint \
  -z 128 \
  --precision bf16-mixed \
  --missing-protocol-label auto
```

Any trailing Hydra override is passed through:

```bash
python scripts/benchmark_py/service_submit.py ... ++trainer.precision=bf16-mixed ++data.num_workers=8
```

Extra args must start with `+` or `++`, same rule as the legacy benchmark wrapper.

## Required Inputs

- `-g, --gpu`: GPU id or MIG UUID
- `-c, --config`: experiment config path, same as legacy `experiment=...`
- `-b, --benchmark-folder`: benchmark root containing dataset subfolders
- `-m, --model-path`: base or merged model path
- `-r, --results-folder`: output root
- `-n, --comment`: run name

Optional inputs:

- `-a, --adapter-paths`: adapter or LoRA checkpoint/path
- `-l, --is-ln`: Lightning checkpoint loading flag. Default: `true`
- `-s, --random-start`: random crop/start flag. Default: `true`
- `-t, --trim-length`: trim length. Default: `64000`
- `-z, --batch-size`: batch size. Default: `128`
- `--precision`: adds precision intent to runtime signature and injects `++trainer.precision=<value>` if no precision override already exists
- `--eval-config`: optional eval YAML
- `--missing-protocol-label`: `ask`, `skip`, `auto`, `spoof`, or `bonafide`. Default for service submit: `skip`

## Monitor Worker

Health file:

```bash
cat .benchmark_service_queue/health.json
```

Queue directories:

```bash
find .benchmark_service_queue -maxdepth 2 -type f | sort
```

Meaning:

- `pending/`: submitted, not started
- `running/`: currently claimed by worker
- `done/`: completed successfully
- `failed/`: job failed or had at least one dataset failure

Per-job metadata:

```text
<results-folder>/<run-name>/service_metadata/<job_id>.json
```

Metadata includes `job_id`, dataset path, config/model/adapter paths, GPU id, worker PID, git commit, status, command equivalent, and runtime signature.

## Output Layout

Benchmark artifacts remain legacy-compatible:

```text
<results-folder>/<run-name>/
  summary.txt
  summary_detailed.txt
  summary_details.jsonl
  <dataset>_<normalized_config>_<run-name>.txt
  merged_scores_<normalized_config>_<run-name>.txt
  merged_protocol_<normalized_config>_<run-name>.txt
  service_metadata/
    <job_id>.json
```

## Runtime Reuse Rules

The worker reuses a loaded model only when these fields match:

- GPU id
- config path
- model path
- adapter path
- `is_ln`
- precision
- extra Hydra overrides

If they differ:

- with `--reload-on-change`: worker reloads model, then runs job
- without `--reload-on-change`: job fails with runtime mismatch

## One-Shot Mode

Useful for smoke testing queue behavior:

```bash
python scripts/benchmark_py/service_worker.py \
  --queue-dir .benchmark_service_queue \
  --once
```

Exit codes:

- `0`: one job processed
- `2`: no queued job found

## Rollback

Run old path directly:

```bash
python scripts/benchmark_py/benchmark.py \
  -g 0 \
  -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer \
  -b data/May_2026_benchmark \
  -m /path/to/base_or_merged_model.pth \
  -r logs/results/May_2026_benchmark \
  -n 12May2026_lora_from_29April26_xlsr_conformertcm_mdt-conf-3 \
  -a /path/to/adapter/checkpoint \
  -z 128 \
  --missing-protocol-label auto
```

## Validation Checklist

1. Start worker.
2. Submit two jobs with same model/config/adapter.
3. Confirm `.benchmark_service_queue/health.json` has same runtime signature and `load_count` remains `1`.
4. Compare output folder with legacy command on one small benchmark folder.
5. If mismatch appears, keep using legacy command and inspect `<run-name>/service_metadata/<job_id>.json`.
