import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from benchmark_py.protocol import parse_protocol_line

from src.benchmark_service.legacy_adapter import LegacyBenchmarkAdapter
from src.benchmark_service.queue import FileJobQueue
from src.benchmark_service.schemas import BenchmarkJob, RuntimeConfig
from src.benchmark_service.worker import PersistentBenchmarkWorker


class FakeRuntime:
    def __init__(self):
        self.load_count = 0
        self.runtime_config = None
        self.executions = []

    def load(self, runtime_config):
        self.runtime_config = runtime_config
        self.load_count += 1

    def matches(self, runtime_config):
        return self.runtime_config == runtime_config

    def execute_benchmark(self, config):
        if self.runtime_config is None:
            self.load(RuntimeConfig(
                gpu_id=config.gpu_number,
                config_path=config.yaml_config,
                model_path=config.base_model_path,
                adapter_path=config.adapter_paths,
                is_ln=config.is_base_model_path_ln,
                extra_overrides=[],
            ))
        self.executions.append(config)
        config.score_save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config.protocol_path, "r", encoding="utf-8") as src, open(
            config.score_save_path, "w", encoding="utf-8"
        ) as dst:
            for line in src:
                parsed = parse_protocol_line(line)
                if parsed:
                    rel_path, _, _ = parsed
                    dst.write(f"{rel_path} 0.1 0.9\n")
        return True


def make_benchmark_root(tmp_path):
    root = tmp_path / "benchmark"
    for name in ["dataset_a", "dataset_b"]:
        dataset = root / name
        dataset.mkdir(parents=True)
        (dataset / "protocol.txt").write_text(
            "sample_1.wav eval bonafide\nsample_2.wav eval spoof\n",
            encoding="utf-8",
        )
    return root


def make_job(tmp_path, benchmark_root):
    return BenchmarkJob(
        dataset_path=str(benchmark_root),
        result_dir=str(tmp_path / "results"),
        run_name="run_a",
        config_path="cnsl/example",
        model_path="/tmp/model.ckpt",
        gpu_id="0",
        adapter_path="/tmp/adapter",
        batch_size=2,
        missing_protocol_label="skip",
    )


def patch_heavy_legacy(monkeypatch):
    import benchmark_py.benchmark as legacy_benchmark

    monkeypatch.setattr(
        legacy_benchmark,
        "evaluate_results",
        lambda score_file, protocol_file, summary_file, dataset_name, *args, **kwargs: (
            Path(summary_file).open("a", encoding="utf-8").write(f"{dataset_name} | 0 | 0 | 0\n")
            or True
        ),
    )
    monkeypatch.setattr(legacy_benchmark, "calculate_pooled_eer", lambda *args, **kwargs: None)
    monkeypatch.setattr(legacy_benchmark, "create_merged_protocol", lambda *args, **kwargs: None)


def test_job_schema_roundtrip_and_command(tmp_path):
    benchmark_root = make_benchmark_root(tmp_path)
    job = make_job(tmp_path, benchmark_root)
    payload = json.loads(job.to_json())

    restored = BenchmarkJob.from_dict(payload)

    assert restored.job_id == job.job_id
    assert restored.command_equivalent()[:2] == ["python", "scripts/benchmark_py/benchmark.py"]
    assert "-b" in restored.command_equivalent()


def test_job_rejects_non_hydra_extra_args(tmp_path):
    benchmark_root = make_benchmark_root(tmp_path)
    job = make_job(tmp_path, benchmark_root)
    job.extra_overrides = ["--bad-flag"]

    with pytest.raises(ValueError, match="Hydra overrides"):
        job.validate()


def test_legacy_adapter_runs_two_datasets_with_injected_executor(tmp_path, monkeypatch):
    patch_heavy_legacy(monkeypatch)
    benchmark_root = make_benchmark_root(tmp_path)
    job = make_job(tmp_path, benchmark_root)
    runtime = FakeRuntime()

    success = LegacyBenchmarkAdapter(runtime.execute_benchmark).run(job)

    assert success is True
    assert len(runtime.executions) == 2
    result_dir = Path(job.result_dir) / job.run_name
    assert (result_dir / f"dataset_a_cnsl_example_{job.run_name}.txt").exists()
    assert (result_dir / f"dataset_b_cnsl_example_{job.run_name}.txt").exists()


def test_worker_reuses_loaded_runtime_for_sequential_jobs(tmp_path, monkeypatch):
    patch_heavy_legacy(monkeypatch)
    benchmark_root = make_benchmark_root(tmp_path)
    queue = FileJobQueue(tmp_path / "queue")
    job_a = make_job(tmp_path, benchmark_root)
    job_b = make_job(tmp_path, benchmark_root)
    job_b.run_name = "run_b"
    queue.submit(job_a)
    queue.submit(job_b)

    worker = PersistentBenchmarkWorker(queue=queue, reload_on_change=False, poll_seconds=0.01)
    fake_runtime = FakeRuntime()
    fake_runtime.load(RuntimeConfig.from_job(job_a))
    worker.runtime = fake_runtime

    assert worker.run_once() is True
    assert worker.run_once() is True
    assert fake_runtime.load_count == 1
    assert len(fake_runtime.executions) == 4
    assert not list(queue.pending_dir.glob("*.json"))
    assert len(list(queue.done_dir.glob("*.json"))) == 2
