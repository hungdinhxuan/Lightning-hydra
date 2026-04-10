#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVENT_RESULT_READY = "RESULT_READY"
EVENT_FAILED = "FAILED"
EVENT_NEED_DECISION = "NEED_DECISION"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def repo_root_from_queue(queue_path: Path) -> Path:
    # .../plans/<ts>/queue.jsonl -> repo root is parents[2]
    return queue_path.resolve().parents[2]


def log_suggests_oom(log_path: Path, tail_bytes: int = 256_000) -> bool:
    if not log_path.is_file():
        return False
    try:
        data = log_path.read_bytes()
        if len(data) > tail_bytes:
            data = data[-tail_bytes:]
        text = data.decode("utf-8", errors="replace")
    except OSError:
        return False
    return "OutOfMemoryError" in text or "CUDA out of memory" in text


def append_event(events_file: Path, payload: dict[str, Any]) -> None:
    with events_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def job_is_training(rec: dict[str, Any]) -> bool:
    """Has a live PID to poll (training or eval actually started)."""
    return rec.get("status") == "running" and rec.get("pid") is not None


def slot_reserved(rec: dict[str, Any]) -> bool:
    """GPU slot is taken: active training, or pending row already allocated a gpu_id."""
    if job_is_training(rec):
        return True
    return rec.get("status") == "pending" and rec.get("gpu_id") is not None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-file", default="plans/01042026/queue.jsonl")
    parser.add_argument("--events-file", default="plans/01042026/events.jsonl")
    parser.add_argument("--gpu-id", default=None)
    args = parser.parse_args()

    queue_path = Path(args.queue_file)
    if not queue_path.exists():
        print("[monitor_jobs] queue file not found")
        return 0

    events_path = Path(args.events_file)
    records = [json.loads(line) for line in queue_path.read_text().splitlines() if line.strip()]
    changed = False
    # Exit 10 after this run if we emitted FAILED or NEED_DECISION (cron/agent hook).
    need_agent = False

    # 1) Check running jobs and emit completion/failure events.
    for rec in records:
        if rec.get("status") != "running":
            continue
        pid = rec.get("pid")
        if pid is None:
            # Monitor used to set status=running before nohup recorded a pid — recover instead of FAILED.
            rec["status"] = "pending"
            rec["updated_at"] = now_iso()
            changed = True
            continue

        if not pid_exists(int(pid)):
            rec["updated_at"] = now_iso()
            log_rel = rec.get("log_file")
            log_abs: Path | None = None
            if isinstance(log_rel, str) and log_rel.strip():
                log_abs = repo_root_from_queue(queue_path) / log_rel
            oom = bool(log_abs and log_suggests_oom(log_abs))
            if oom:
                rec["status"] = "failed"
                rec["failure_reason"] = "OOM"
                append_event(
                    events_path,
                    {
                        "trial_id": rec.get("trial_id", "unknown"),
                        "event": EVENT_FAILED,
                        "reason": "process exited with OOM (log)",
                        "timestamp": now_iso(),
                    },
                )
                need_agent = True
            else:
                rec["status"] = "done"
                append_event(
                    events_path,
                    {
                        "trial_id": rec.get("trial_id", "unknown"),
                        "event": EVENT_RESULT_READY,
                        "reason": "process exited",
                        "timestamp": now_iso(),
                    },
                )
            changed = True

    # 2) If no slot is reserved, optionally allocate GPU for one unallocated pending job.
    if not any(slot_reserved(r) for r in records):
        pending = next(
            (r for r in records if r.get("status") == "pending" and r.get("gpu_id") is None),
            None,
        )
        if pending is not None and args.gpu_id is not None:
            pending["gpu_id"] = args.gpu_id
            pending["updated_at"] = now_iso()
            append_event(
                events_path,
                {
                    "trial_id": pending.get("trial_id", "unknown"),
                    "event": EVENT_NEED_DECISION,
                    "reason": "dispatch ready on selected gpu; launch run_trial.py",
                    "gpu_id": args.gpu_id,
                    "timestamp": now_iso(),
                },
            )
            need_agent = True
            changed = True

    if changed:
        with queue_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print("[monitor_jobs] updated queue and emitted events")
    else:
        print("[monitor_jobs] no queue change")

    if need_agent:
        print("[monitor_jobs] AGENT_TRIGGER exit=10 (FAILED or NEED_DECISION emitted this run)")
        return 10
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

