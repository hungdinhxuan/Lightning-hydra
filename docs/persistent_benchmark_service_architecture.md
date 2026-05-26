# Persistent Benchmark Service Architecture

Usage guide: `docs/persistent_benchmark_service_usage.md`

## Existing Path Map

- Entrypoint: `scripts/benchmark_py/benchmark.py`
- Per-dataset loop: `get_subdirectories()` then `process_dataset()`
- Model execution command: `scripts/benchmark_py/execution.py::execute_benchmark()`
- Legacy model launch: `python src/train.py experiment=... ++train=False ++test=True ...`
- Score output: model `test_step()` writes `model.score_save_path`
- Metric/output path: `evaluate_results()`, `calculate_pooled_eer()`, and `create_merged_protocol()`

## Service Hook

`process_dataset()` now accepts optional `execute_benchmark_fn`. Existing CLI uses same subprocess executor. Persistent worker injects `LoadedBenchmarkModel.execute_benchmark()` instead.

## Resident Runtime

`src/benchmark_service/model_runtime.py` composes Hydra config once, instantiates model once, and runs each dataset with:

- fresh datamodule
- fresh trainer/callback/logger instances
- same resident model object
- updated per-job `score_save_path`, `data_dir`, `protocol_path`, batch size, trim length, and random start

## Worker Flow

1. `service_submit.py` writes `BenchmarkJob` JSON into file queue.
2. `service_worker.py` pops jobs sequentially.
3. Worker loads model from first job runtime signature.
4. Each job runs legacy benchmark orchestration through `LegacyBenchmarkAdapter`.
5. Metadata lands in `<results>/<run_name>/service_metadata/<job_id>.json`.
6. Health lands in `<queue_dir>/health.json`.

## Rollback

Legacy command remains valid:

```bash
python scripts/benchmark_py/benchmark.py -g 0 -c CONFIG -b BENCHMARK_DIR -m MODEL -r RESULTS -n RUN
```
