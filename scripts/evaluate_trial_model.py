#!/usr/bin/env python3
"""Run 10% benchmark eval for a completed test10pct trial (queue-driven)."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
CKPT_LINE_RE = re.compile(
    r"Created averaged checkpoint:\s*(\S+)",
    re.MULTILINE,
)
PROTO_10PCT_MARKER = Path("plans/01042026/protocol_subsets_10pct/protocol.txt")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_queue(queue_path: Path) -> list[dict[str, Any]]:
    lines = queue_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def find_trial(records: list[dict[str, Any]], trial_id: str) -> dict[str, Any]:
    for rec in records:
        if rec.get("trial_id") == trial_id:
            return rec
    raise SystemExit(f"[evaluate_trial_model] trial_id not in queue: {trial_id}")


def extract_averaged_ckpt(log_path: Path) -> str:
    if not log_path.is_file():
        raise SystemExit(f"[evaluate_trial_model] log file not found: {log_path}")
    raw = log_path.read_text(encoding="utf-8", errors="replace")
    clean = ANSI_RE.sub("", raw)
    matches = CKPT_LINE_RE.findall(clean)
    if not matches:
        raise SystemExit(
            f"[evaluate_trial_model] no 'Created averaged checkpoint:' line in {log_path}"
        )
    return matches[-1].strip()


def append_event(events_path: Path, payload: dict[str, Any]) -> None:
    with events_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def ensure_protocol_subsets_10pct(root: Path, dry_run: bool) -> None:
    marker = root / PROTO_10PCT_MARKER
    if marker.is_file():
        print(f"[evaluate_trial_model] protocol subsets OK: {marker}")
        return
    script = root / "plans/01042026/build_protocol_subsets_10pct.sh"
    if not script.is_file():
        raise SystemExit(f"[evaluate_trial_model] missing {script}")
    cmd = ["bash", str(script)]
    print(f"[evaluate_trial_model] building protocol subsets: {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=root, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--queue-file", default="plans/01042026/queue.jsonl")
    parser.add_argument("--events-file", default="plans/01042026/events.jsonl")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions only; do not run eval or append events",
    )
    args = parser.parse_args()

    tid = args.trial_id
    if "test10pct" not in tid:
        raise SystemExit(
            "[evaluate_trial_model] only trial_id containing 'test10pct' is supported"
        )

    root = repo_root()
    queue_path = Path(args.queue_file)
    if not queue_path.is_absolute():
        queue_path = (Path.cwd() / queue_path).resolve()
    events_path = Path(args.events_file)
    if not events_path.is_absolute():
        events_path = (Path.cwd() / events_path).resolve()

    records = load_queue(queue_path)
    rec = find_trial(records, tid)

    log_rel = rec.get("log_file")
    if not isinstance(log_rel, str) or not log_rel.strip():
        raise SystemExit("[evaluate_trial_model] queue record missing log_file")
    log_path = root / log_rel

    gpu_id = rec.get("gpu_id")
    if gpu_id is None:
        raise SystemExit("[evaluate_trial_model] queue record missing gpu_id")

    ensure_protocol_subsets_10pct(root, args.dry_run)

    ckpt = extract_averaged_ckpt(log_path)
    print(f"[evaluate_trial_model] CKPT={ckpt}")

    env = os.environ.copy()
    env["MIG_ID"] = str(gpu_id)
    env["CKPT"] = ckpt
    env["TRIAL_ID"] = tid

    bench_script = root / "plans/01042026/run_benchmark_eval_10pct.sh"
    if not bench_script.is_file():
        raise SystemExit(f"[evaluate_trial_model] missing {bench_script}")

    cmd = ["bash", str(bench_script)]
    print(f"[evaluate_trial_model] running: TRIAL_ID={tid} MIG_ID={gpu_id} {' '.join(cmd)}")
    if args.dry_run:
        print("[evaluate_trial_model] dry-run: skipping benchmark and event append")
        return 0

    subprocess.run(cmd, cwd=root, env=env, check=True)

    append_event(
        events_path,
        {
            "trial_id": tid,
            "event": "EVALUATE_MODEL_DONE",
            "reason": "run_benchmark_eval_10pct.sh completed",
            "ckpt": ckpt,
            "gpu_id": gpu_id,
            "timestamp": now_iso(),
        },
    )
    print(f"[evaluate_trial_model] appended EVALUATE_MODEL_DONE to {events_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
