"""File-based job queue for persistent benchmark worker."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Union

from src.benchmark_service.schemas import BenchmarkJob


class FileJobQueue:
    """Tiny durable queue using pending/running/done/failed directories."""

    def __init__(self, queue_dir: Union[str, Path]) -> None:
        self.queue_dir = Path(queue_dir)
        self.pending_dir = self.queue_dir / "pending"
        self.running_dir = self.queue_dir / "running"
        self.done_dir = self.queue_dir / "done"
        self.failed_dir = self.queue_dir / "failed"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        for path in [self.pending_dir, self.running_dir, self.done_dir, self.failed_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def submit(self, job: BenchmarkJob) -> Path:
        job.validate()
        tmp_path = self.pending_dir / f".{job.job_id}.json.tmp"
        final_path = self.pending_dir / f"{job.job_id}.json"
        job.write_json(tmp_path)
        os.replace(tmp_path, final_path)
        return final_path

    def iter_pending(self) -> Iterable[Path]:
        yield from sorted(self.pending_dir.glob("*.json"))

    def pop_next(self) -> Optional[Path]:
        for pending_path in self.iter_pending():
            running_path = self.running_dir / pending_path.name
            try:
                os.replace(pending_path, running_path)
            except FileNotFoundError:
                continue
            return running_path
        return None

    def mark_done(self, running_path: Union[str, Path]) -> Path:
        running_path = Path(running_path)
        done_path = self.done_dir / running_path.name
        os.replace(running_path, done_path)
        return done_path

    def mark_failed(self, running_path: Union[str, Path]) -> Path:
        running_path = Path(running_path)
        failed_path = self.failed_dir / running_path.name
        os.replace(running_path, failed_path)
        return failed_path
