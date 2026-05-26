# Persistent Inference Service Engineering Spec

## Background

The current benchmark workflow is optimized for experimentation speed, but it reloads the model repeatedly for each dataset run. In the current manual flow, a dataset is placed into a benchmark folder and a benchmark script is executed with explicit model, adapter, config, and output paths. This works well for development, but repeated model initialization creates avoidable overhead when multiple datasets are evaluated sequentially.[1][2]

The immediate engineering goal is not to build a full production inference platform. The goal is to introduce a **persistent benchmark inference service** that keeps a model resident on a selected GPU and reuses it across multiple benchmark jobs, while preserving the existing benchmark logic and output format as much as possible.[3][1][2]

This spec assumes the codebase is still under active development, so full model versioning and hard immutability are not yet required. Instead, the design prioritizes low-risk extensibility, compatibility with the existing benchmark path, and safe rollback to the current script-based workflow.[4]

## Problem Statement

The current benchmark flow incurs repeated costs that are orthogonal to the benchmark itself:

- Model loading is repeated for each dataset run.
- CUDA initialization and precision/runtime setup may repeat unnecessarily.
- The workflow is manual and therefore prone to inconsistency in run metadata capture.[4]
- The current execution mode does not separate long-lived model state from per-dataset execution state.[1][2]

The system needs a way to run multiple benchmark datasets against the same loaded model instance on a fixed GPU without repeatedly reinitializing the full model stack.[1][2]

## Goals

### Primary goals

- Keep a model resident on a specific GPU across multiple benchmark jobs.[3][2]
- Avoid repeated model load for each dataset benchmark run.[1][2]
- Reuse the current benchmark logic with minimal behavioral changes.
- Add new code around the existing benchmark path rather than rewriting it.[4]
- Preserve current outputs, metrics, and evaluation behavior as much as possible.
- Support safe fallback to the current script-based benchmark execution.

### Secondary goals

- Add minimal job metadata tracking for reproducibility during development.[4]
- Support controlled model reload when model or adapter artifacts change.
- Create a path toward future benchmark automation without requiring it now.[5][6]

## Non-Goals

The first implementation is explicitly **not** intended to provide:

- Full model registry or formal model version management.[4]
- Multi-tenant inference serving.
- Public production API serving.
- Autoscaling or elastic scheduling infrastructure.[3]
- Online low-latency serving guarantees.
- A full dashboard product.
- Immediate replacement of the existing benchmark CLI path.

## Design Principles

### Additive change, not invasive rewrite

The implementation must introduce a new layer around the existing benchmark code. Existing benchmark scripts remain valid and runnable throughout the migration.

### Legacy-compatible execution core

The current benchmark path is treated as the execution baseline. New service code should call into the existing logic through thin wrappers or extracted reusable functions rather than reimplementing benchmark behavior.

### One pain point first

The first version should optimize for the dominant bottleneck: repeated model loading. It should not attempt to solve every benchmark orchestration problem in the same milestone.[1][2]

### Explicit rollback path

At every stage, the team must be able to run the old benchmark command directly if the new service path fails.

## Proposed Architecture

### High-level architecture

The recommended first implementation is a **single-model persistent benchmark worker**:

- One worker process is pinned to one GPU.[3][1]
- The worker loads one model stack into memory at startup.[2]
- Jobs are submitted to the worker using a file-based queue, a small API, or a simple internal queue.
- For each job, the worker runs the existing benchmark inference/evaluation path using the already-loaded model.
- The worker writes results in a format compatible with the current benchmark output.

This architecture addresses repeated model initialization without forcing a production-grade serving layer.[3][1][2]

### Logical components

| Component | Purpose | Required in v1 |
|---|---|---|
| `legacy_adapter` | Wrap existing benchmark code into reusable interfaces | Yes |
| `model_runtime` | Load and retain model on GPU, expose reload behavior | Yes |
| `benchmark_worker` | Long-lived process that executes benchmark jobs sequentially | Yes |
| `job_schema` | Structured representation of dataset benchmark requests | Yes |
| `result_writer` | Emit metrics and artifacts in legacy-compatible format | Yes |
| `submit_cli` | Submit jobs to the worker | Yes |
| `health` | Provide worker liveness/readiness introspection | Recommended |
| `server` | Optional lightweight HTTP control plane | Optional |
| `run_registry` | Minimal metadata persistence for runs | Recommended |

## Execution Model

### Worker lifecycle

The persistent worker lifecycle should be:

1. Start worker process.
2. Select and pin target GPU.
3. Load config, model, adapter, checkpoint, and runtime options.
4. Run a warmup pass to validate readiness.
5. Enter idle state and wait for benchmark jobs.
6. Receive benchmark job.
7. Build dataset loader and run inference/evaluation using the already-loaded model.
8. Write outputs and metadata.
9. Return to idle state.
10. On command, reload model or restart worker.

This separation ensures long-lived model state and short-lived per-job state remain independent.[1][2]

### Job execution semantics

The worker should process jobs sequentially in the first version. This simplifies state management and reduces the risk of concurrency bugs in a legacy codebase not originally designed for multi-job residency.

Each job should be isolated at the dataset-run level:

- Dataset-specific temporary state must not leak into subsequent jobs.
- Logging handlers must not accumulate between runs.
- Output directories must remain job-specific.
- Per-job exceptions must not corrupt the long-lived model state.

## Compatibility Strategy

The compatibility strategy is the most important part of the design.

### Existing benchmark logic remains authoritative

The new service must delegate the following behaviors to the existing benchmark implementation whenever possible:

- Audio preprocessing
- Dataset construction
- Batch collation
- Forward pass behavior
- Postprocessing and score generation
- Metric aggregation
- Result serialization

### Adapter-based integration

If the current benchmark code is monolithic, the implementation should add a thin adapter layer instead of rewriting benchmark logic. Example adapter responsibilities:

- Convert submitted jobs into legacy argument/config structures.
- Call legacy dataset-building functions.
- Call benchmark inference loops with an already-loaded model handle.
- Call or reuse existing output-writing routines.

## Recommended Internal Interfaces

The following internal interfaces are recommended. Names may vary, but the boundaries should be preserved.

```python
class BenchmarkJob:
    job_id: str
    dataset_path: str
    result_dir: str
    run_name: str
    batch_size: int
    precision: str
    extra_overrides: dict

class LoadedBenchmarkModel:
    def predict_batch(self, batch):
        ...

class BenchmarkRunner:
    def run_job(self, job: BenchmarkJob):
        ...
```

The critical distinction is that `LoadedBenchmarkModel` is created once per worker lifecycle, while `run_job()` is invoked once per dataset run.[1][2]

## Data Flow

### Control flow

1. A user or upstream script submits a `BenchmarkJob`.
2. The worker validates the job payload.
3. The worker uses the adapter layer to construct dataset loaders and per-job runtime settings.
4. The worker runs benchmark inference using the already-loaded model.
5. The result writer emits output artifacts.
6. Metadata is recorded for traceability.

### Output flow

The service should preserve the current benchmark artifacts wherever possible. At minimum, it should write:

- Aggregate metric summaries
- Per-file or per-sample scores when currently supported
- Run metadata
- Error state if the job fails

## Job Schema

A practical v1 schema could look like this:

```json
{
  "job_id": "uuid-or-timestamp",
  "dataset_path": "data/May_2026_benchmark",
  "result_dir": "logs/results/May_2026_benchmark",
  "run_name": "12May2026_lora_from_29April26_xlsr_conformertcm_mdt-conf-3",
  "batch_size": 128,
  "precision": "bf16-mixed",
  "gpu_id": 0,
  "config_path": "cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer",
  "model_path": "/NAS1_pretrained_lab/29April26_xlsr_conformertcm_mdt_lora_merged.pth",
  "adapter_path": "/data/hungdx/lighning-hydra-train-runs/runs/2026-05-12_22-15-02/checkpoints/epoch_003.ckpt",
  "extra_overrides": {
    "trainer.precision": "bf16-mixed"
  }
}
```

This schema is deliberately close to the existing command-line workflow so that migration cost remains low.

## Submission Modes

### Recommended first mode: CLI submit

The easiest low-risk path is to add a submission CLI that serializes jobs and pushes them to the worker.

Example:

```bash
python service_submit.py \
  --dataset data/May_2026_benchmark \
  --result logs/results/May_2026_benchmark \
  --run-name 12May2026_lora_from_29April26_xlsr_conformertcm_mdt-conf-3
```

This approach avoids introducing HTTP concerns too early.

### Optional second mode: lightweight API

If a control plane is desirable, a small API may expose:

- `POST /jobs`
- `GET /jobs/{id}`
- `POST /reload`
- `GET /health`

This API should remain thin and must not embed benchmark logic directly.

## Model Residency and Reload Semantics

### Residency model

The loaded model should remain resident in a dedicated worker process on a fixed GPU until one of the following occurs:

- A reload command is issued.
- A fatal runtime error requires process restart.
- A deployment/update event replaces the worker.

### Reload policy

Because the codebase is still under development and formal versioning is not yet stable, reloading should be explicit and controllable. A reload operation should:

1. Drain or reject new jobs temporarily.
2. Finish or abort the current job safely.
3. Release current model state.
4. Load the new model stack.
5. Run warmup.
6. Resume accepting jobs.

## Metadata Strategy

Although the system is pre-versioning, each run should still capture enough metadata to support traceability.[4]

Recommended run metadata:

- `job_id`
- `dataset_path`
- `run_name`
- `config_path`
- `model_path`
- `adapter_path`
- `extra_overrides`
- `gpu_id`
- `worker_pid`
- `worker_started_at`
- `started_at`
- `finished_at`
- `status`
- `git_commit`
- `command_equivalent`

This metadata is sufficient for debugging and run comparison during active development.[4]

## Failure Handling

### Failure model

Per-job failure should be isolated from worker lifecycle whenever possible.

A dataset job may fail because of:

- Corrupt input data
- Unexpected preprocessing edge cases
- CUDA out-of-memory events
- Runtime exceptions in metric logic

The worker should handle these without immediately requiring a full rewrite of the process lifecycle.

### Failure handling policy

- A job failure must produce an explicit failed status and error log.
- The worker should remain alive unless the error compromises model/runtime integrity.
- A configurable restart path should exist for repeated failures.
- Model state should not be silently reused after a suspected corrupted runtime state.

## Performance Expectations

The main expected gain is removal of repeated model load overhead between dataset runs.[1][2]

Performance improvements are expected in:

- Total elapsed time across multiple benchmark runs
- Reduced startup overhead per dataset
- Better utilization of a designated GPU for sequential benchmark jobs

The first version does **not** optimize for maximum throughput across multiple GPUs. It optimizes for lower friction and lower repeated initialization cost.[3][1]

## Testing Strategy

Testing must focus on compatibility, not only correctness.

### 1. Golden output compatibility test

Take a known small dataset and run it through:

- the existing benchmark CLI path
- the new worker path

Compare:

- aggregate metrics
- per-sample outputs where available
- sample counts
- major derived scores

Minor numeric tolerance may be allowed if runtime ordering or precision behavior differs.

### 2. Reuse test

Submit two sequential jobs against the same worker and verify the model is loaded only once.

### 3. Reload test

Reload to a new model or adapter and verify:

- worker readiness returns successfully
- model signature changes
- subsequent jobs run with the new model state

### 4. Failure isolation test

Submit a malformed or failing dataset job and verify:

- the job fails cleanly
- the worker remains recoverable
- subsequent valid jobs can still execute if runtime state is intact

### 5. Fallback test

Verify the old benchmark command remains operational throughout implementation and rollout.

## Migration Plan

### Phase 1: code archaeology

The assigned agent should first map the current codebase and identify:

- entrypoint
- model load path
- dataset loop path
- metric/output path
- configuration flow
- hidden global state or side effects

No new service code should be written before this map is complete.

### Phase 2: extract or wrap reusable boundaries

The agent should then create a compatibility layer around the legacy benchmark implementation. This may involve:

- function extraction
- wrapper classes
- adapter utilities

The old CLI path must still work unchanged.

### Phase 3: implement worker

Implement a single-GPU sequential worker that uses the adapter layer.

### Phase 4: submit path and metadata

Add CLI-based submission and metadata capture.

### Phase 5: compatibility validation

Run old-path versus new-path comparisons on representative datasets.

### Phase 6: soft rollout

Use the worker path experimentally while retaining the legacy path as the default fallback.

## Rollback Plan

If the persistent worker path proves unstable, the system should immediately revert to the current benchmark command without requiring artifact or codebase rollback.

Rollback mechanism:

- Keep the current benchmark CLI path unchanged.
- Do not remove legacy entrypoints during initial rollout.
- Treat the service as additive until repeated equivalence tests pass.

## Suggested Repository Layout

```text
scripts/
  benchmark_py/
    benchmark.py
    service_submit.py

src/
  benchmark_service/
    __init__.py
    schemas.py
    worker.py
    model_runtime.py
    legacy_adapter.py
    result_writer.py
    health.py
    server.py
    queue.py

tests/
  test_legacy_compat.py
  test_worker_reuse_model.py
  test_reload_model.py
```

## Risks and Mitigations

| Risk | Description | Mitigation |
|---|---|---|
| Hidden global state | Legacy code may mutate globals or process-wide state | Audit startup and per-job boundaries before service implementation |
| Tight CLI coupling | Legacy logic may be embedded inside `main()` | Introduce adapter layer rather than wide refactor |
| Output drift | New path may accidentally change metric/output behavior | Use golden compatibility tests |
| Runtime state leakage | Long-lived worker may accumulate logger handlers, tensors, or stale config | Add per-job cleanup and bounded restart policy |
| GPU instability | Worker pinned to an unstable GPU may reduce reliability | Require explicit GPU pinning and health checks |
| Over-engineering | Too many platform features added too early | Restrict v1 to persistent worker scope |

## Acceptance Criteria

The implementation is acceptable only if all of the following hold:

- The legacy benchmark script still runs successfully.
- Two sequential benchmark jobs can run without reloading the model.
- Primary metrics remain compatible with the legacy path.
- The worker can be restarted or reloaded safely.
- Minimal metadata is captured for each run.[4]
- A fallback path to the old command remains available.
- Smoke tests and compatibility tests are included.

## Recommended Agent Workflow

A coding agent assigned to this project should follow this order strictly:

1. Read and map the existing benchmark codebase.
2. Produce a short architecture note describing entrypoint, load path, inference path, and output path.
3. Identify safe hook points for additive integration.
4. Implement adapter layer.
5. Implement single-GPU persistent worker.
6. Implement submit path and metadata capture.
7. Add regression and compatibility tests.
8. Validate with representative datasets.

The agent should be explicitly instructed not to replace or rewrite benchmark logic unless no compatible integration point exists.

## Example Implementation Prompt for the Agent

```md
Read the current benchmark codebase and implement a persistent benchmark worker that keeps one model resident on a specified GPU across multiple benchmark jobs. The implementation must minimize changes to existing benchmark logic. Treat the current benchmark script as the authoritative execution path and wrap it through an adapter layer. Preserve output compatibility, add job metadata capture, support explicit model reload, and keep the current CLI workflow fully functional as a fallback. Work incrementally: first map the codebase, then identify safe integration points, then implement the worker, then add compatibility tests.
```

## Final Recommendation

The correct engineering move is to build a **persistent benchmark inference worker**, not a full inference platform. This directly addresses the repeated model-load bottleneck while preserving the current benchmark implementation as the source of truth.[1][2] The safest path is additive: wrap the legacy benchmark code, keep the old CLI path alive, and validate compatibility through side-by-side tests before expanding the service further.[4]