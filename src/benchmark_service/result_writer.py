"""Metadata writer for persistent benchmark jobs."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from src.benchmark_service.schemas import BenchmarkJob, RuntimeConfig, utc_now_iso


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


class ResultWriter:
    """Write per-job service metadata next to benchmark result folder."""

    def __init__(self, worker_started_at: str, metadata_subdir: str = "service_metadata") -> None:
        self.worker_started_at = worker_started_at
        self.metadata_subdir = metadata_subdir

    def metadata_path(self, job: BenchmarkJob) -> Path:
        return Path(job.result_dir) / job.run_name / self.metadata_subdir / f"{job.job_id}.json"

    def write(
        self,
        job: BenchmarkJob,
        status: str,
        started_at: str,
        finished_at: str,
        runtime_config: RuntimeConfig,
        error: Optional[str] = None,
    ) -> Path:
        path = self.metadata_path(job)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "job_id": job.job_id,
            "dataset_path": job.dataset_path,
            "run_name": job.run_name,
            "config_path": job.config_path,
            "model_path": job.model_path,
            "adapter_path": job.adapter_path,
            "extra_overrides": job.extra_overrides,
            "gpu_id": job.gpu_id,
            "worker_pid": os.getpid(),
            "worker_started_at": self.worker_started_at,
            "started_at": started_at,
            "finished_at": finished_at,
            "status": status,
            "git_commit": _git_commit(),
            "command_equivalent": job.command_equivalent(),
            "runtime_signature": runtime_config.signature(),
        }
        if error:
            payload["error"] = error
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path
