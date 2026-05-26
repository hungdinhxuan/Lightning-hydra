"""Single-model persistent benchmark worker."""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from src.benchmark_service.health import build_health_payload, write_health
from src.benchmark_service.legacy_adapter import LegacyBenchmarkAdapter
from src.benchmark_service.queue import FileJobQueue
from src.benchmark_service.result_writer import ResultWriter
from src.benchmark_service.schemas import BenchmarkJob, RuntimeConfig, utc_now_iso

if TYPE_CHECKING:
    from src.benchmark_service.model_runtime import LoadedBenchmarkModel


class PersistentBenchmarkWorker:
    """Sequential worker that keeps one benchmark model resident."""

    def __init__(
        self,
        queue: FileJobQueue,
        initial_runtime: Optional[RuntimeConfig] = None,
        reload_on_change: bool = False,
        poll_seconds: float = 5.0,
        health_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.queue = queue
        self.reload_on_change = reload_on_change
        self.poll_seconds = poll_seconds
        self.health_path = Path(health_path) if health_path else queue.queue_dir / "health.json"
        self.worker_started_at = utc_now_iso()
        self.result_writer = ResultWriter(worker_started_at=self.worker_started_at)
        self.runtime: Optional["LoadedBenchmarkModel"] = None
        if initial_runtime:
            self.runtime = self._new_runtime(initial_runtime)
            self.runtime.load()
        self.current_job_id: Optional[str] = None
        self.status = "idle"
        self._write_health()

    def _write_health(self) -> None:
        runtime_config = self.runtime.runtime_config if self.runtime else None
        load_count = self.runtime.load_count if self.runtime else 0
        write_health(
            self.health_path,
            build_health_payload(
                worker_started_at=self.worker_started_at,
                runtime_config=runtime_config,
                status=self.status,
                current_job_id=self.current_job_id,
                load_count=load_count,
            ),
        )

    @staticmethod
    def _new_runtime(runtime_config: RuntimeConfig) -> "LoadedBenchmarkModel":
        from src.benchmark_service.model_runtime import LoadedBenchmarkModel

        return LoadedBenchmarkModel(runtime_config)

    def _ensure_runtime(self, runtime_config: RuntimeConfig) -> "LoadedBenchmarkModel":
        if self.runtime is None:
            self.runtime = self._new_runtime(runtime_config)
            self.runtime.load()
            return self.runtime
        if self.runtime.matches(runtime_config):
            return self.runtime
        if not self.reload_on_change:
            raise ValueError(
                "job runtime differs from loaded worker runtime; "
                "restart worker, issue explicit reload, or use --reload-on-change"
            )
        self.runtime.reload(runtime_config)
        return self.runtime

    def reload(self, runtime_config: RuntimeConfig) -> None:
        if self.runtime is None:
            self.runtime = self._new_runtime(runtime_config)
            self.runtime.load()
        else:
            self.runtime.reload(runtime_config)
        self._write_health()

    def run_job(self, job: BenchmarkJob) -> bool:
        runtime_config = RuntimeConfig.from_job(job)
        runtime = self._ensure_runtime(runtime_config)
        adapter = LegacyBenchmarkAdapter(runtime.execute_benchmark)
        started_at = utc_now_iso()
        self.current_job_id = job.job_id
        self.status = "running"
        self._write_health()
        try:
            success = adapter.run(job)
        except Exception as exc:
            finished_at = utc_now_iso()
            error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.result_writer.write(
                job=job,
                status="failed",
                started_at=started_at,
                finished_at=finished_at,
                runtime_config=runtime_config,
                error=error,
            )
            self.status = "idle"
            self.current_job_id = None
            self._write_health()
            raise

        finished_at = utc_now_iso()
        self.result_writer.write(
            job=job,
            status="success" if success else "failed",
            started_at=started_at,
            finished_at=finished_at,
            runtime_config=runtime_config,
            error=None if success else "one or more datasets failed",
        )
        self.status = "idle"
        self.current_job_id = None
        self._write_health()
        return success

    def run_once(self) -> bool:
        running_path = self.queue.pop_next()
        if running_path is None:
            return False
        job = BenchmarkJob.from_json_file(running_path)
        try:
            success = self.run_job(job)
        except Exception:
            self.queue.mark_failed(running_path)
            raise
        if success:
            self.queue.mark_done(running_path)
        else:
            self.queue.mark_failed(running_path)
        return True

    def run_forever(self) -> None:
        while True:
            try:
                ran_job = self.run_once()
            except KeyboardInterrupt:
                self.status = "stopped"
                self._write_health()
                raise
            except Exception:
                traceback.print_exc()
                self.status = "idle"
                self.current_job_id = None
                self._write_health()
                ran_job = True
            if not ran_job:
                time.sleep(self.poll_seconds)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent benchmark worker")
    parser.add_argument("--queue-dir", default=".benchmark_service_queue")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--reload-on-change", action="store_true")
    parser.add_argument("--once", action="store_true", help="Process one queued job and exit")
    parser.add_argument("--health-path", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    worker = PersistentBenchmarkWorker(
        queue=FileJobQueue(args.queue_dir),
        reload_on_change=args.reload_on_change,
        poll_seconds=args.poll_seconds,
        health_path=args.health_path,
    )
    if args.once:
        return 0 if worker.run_once() else 2
    worker.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
