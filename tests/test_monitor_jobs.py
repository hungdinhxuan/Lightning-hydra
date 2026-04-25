"""Tests for scripts/monitor_jobs.py (trial-experiment-monitoring queue monitor)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import monitor_jobs  # noqa: E402


def _run_main(queue: Path, events: Path, gpu_id: str | None = None) -> int:
    old = sys.argv
    try:
        argv = ["monitor_jobs", "--queue-file", str(queue), "--events-file", str(events)]
        if gpu_id is not None:
            argv += ["--gpu-id", gpu_id]
        sys.argv = argv
        return monitor_jobs.main()
    finally:
        sys.argv = old


def test_job_is_training_and_slot_reserved() -> None:
    assert monitor_jobs.job_is_training({"status": "running", "pid": 123})
    assert not monitor_jobs.job_is_training({"status": "running"})
    assert not monitor_jobs.job_is_training({"status": "pending", "pid": 1})

    assert monitor_jobs.slot_reserved({"status": "pending", "gpu_id": "MIG-abc"})
    assert not monitor_jobs.slot_reserved({"status": "pending"})
    assert monitor_jobs.slot_reserved({"status": "running", "pid": 1})


def test_repo_root_from_queue(tmp_path: Path) -> None:
    q = tmp_path / "plans" / "01042026" / "queue.jsonl"
    q.parent.mkdir(parents=True)
    q.write_text("{}\n", encoding="utf-8")
    assert monitor_jobs.repo_root_from_queue(q) == tmp_path.resolve()


def test_log_suggests_oom(tmp_path: Path) -> None:
    log = tmp_path / "t.log"
    log.write_text("ok\n", encoding="utf-8")
    assert not monitor_jobs.log_suggests_oom(log)
    log.write_text("CUDA out of memory at step\n", encoding="utf-8")
    assert monitor_jobs.log_suggests_oom(log)
    log.write_text("torch.OutOfMemoryError: boom\n", encoding="utf-8")
    assert monitor_jobs.log_suggests_oom(log)


def test_main_recovers_running_without_pid(tmp_path: Path) -> None:
    plans = tmp_path / "plans" / "01042026"
    plans.mkdir(parents=True)
    queue = plans / "queue.jsonl"
    events = plans / "events.jsonl"
    rec = {
        "trial_id": "t1",
        "status": "running",
        "gpu_id": "MIG-x",
        "pid": None,
    }
    queue.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    events.write_text("", encoding="utf-8")

    assert _run_main(queue, events) == 0
    rows = [json.loads(l) for l in queue.read_text().splitlines() if l.strip()]
    assert rows[0]["status"] == "pending"
    assert rows[0]["gpu_id"] == "MIG-x"


def test_main_allocates_first_pending_with_gpu_id(tmp_path: Path) -> None:
    plans = tmp_path / "plans" / "01042026"
    plans.mkdir(parents=True)
    queue = plans / "queue.jsonl"
    events = plans / "events.jsonl"
    queue.write_text(
        json.dumps({"trial_id": "t1", "status": "pending", "gpu_id": None}) + "\n",
        encoding="utf-8",
    )
    events.write_text("", encoding="utf-8")

    rc = _run_main(queue, events, gpu_id="MIG-test")
    assert rc == 10
    rows = [json.loads(l) for l in queue.read_text().splitlines() if l.strip()]
    assert rows[0]["status"] == "pending"
    assert rows[0]["gpu_id"] == "MIG-test"
    ev_lines = [l for l in events.read_text().splitlines() if l.strip()]
    assert len(ev_lines) == 1
    ev = json.loads(ev_lines[0])
    assert ev["event"] == monitor_jobs.EVENT_NEED_DECISION
    assert ev["trial_id"] == "t1"


def test_main_marks_done_when_pid_dead(tmp_path: Path) -> None:
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    dead_pid = proc.pid
    proc.terminate()
    proc.wait(timeout=10)

    plans = tmp_path / "plans" / "01042026"
    plans.mkdir(parents=True)
    queue = plans / "queue.jsonl"
    events = plans / "events.jsonl"
    rec = {
        "trial_id": "t1",
        "status": "running",
        "pid": dead_pid,
        "log_file": "plans/01042026/logs/x.log",
    }
    queue.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    log_rel = plans / "logs"
    log_rel.mkdir(parents=True)
    (log_rel / "x.log").write_text("finished ok\n", encoding="utf-8")
    events.write_text("", encoding="utf-8")

    assert _run_main(queue, events) == 10
    rows = [json.loads(l) for l in queue.read_text().splitlines() if l.strip()]
    assert rows[0]["status"] == "done"
    ev = json.loads(events.read_text().splitlines()[0])
    assert ev["event"] == monitor_jobs.EVENT_RESULT_READY


def test_main_no_dispatch_when_slot_reserved(tmp_path: Path) -> None:
    plans = tmp_path / "plans" / "01042026"
    plans.mkdir(parents=True)
    queue = plans / "queue.jsonl"
    events = plans / "events.jsonl"
    queue.write_text(
        json.dumps({"trial_id": "t1", "status": "pending", "gpu_id": "MIG-1"})
        + "\n"
        + json.dumps({"trial_id": "t2", "status": "pending", "gpu_id": None})
        + "\n",
        encoding="utf-8",
    )
    events.write_text("", encoding="utf-8")

    rc = _run_main(queue, events, gpu_id="MIG-2")
    assert rc == 0
    rows = [json.loads(l) for l in queue.read_text().splitlines() if l.strip()]
    assert rows[1]["gpu_id"] is None
    assert events.read_text().strip() == ""
