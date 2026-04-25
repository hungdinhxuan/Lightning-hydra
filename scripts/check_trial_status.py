#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-file", default="plans/01042026/queue.jsonl")
    parser.add_argument("--events-file", default="plans/01042026/events.jsonl")
    args = parser.parse_args()

    queue_path = Path(args.queue_file)
    events_path = Path(args.events_file)

    if not queue_path.exists():
        print("queue file not found")
        return 1

    rows = [json.loads(x) for x in queue_path.read_text().splitlines() if x.strip()]
    print("=== Queue Status ===")
    for r in rows:
        pid = r.get("pid")
        alive = pid_exists(int(pid)) if pid is not None else False
        print(
            f"{r.get('trial_id')} | status={r.get('status')} | gpu_id={r.get('gpu_id')} | pid={pid} | alive={alive}"
        )

    print("\n=== Recent Events ===")
    if not events_path.exists():
        print("events file not found")
        return 0
    for line in events_path.read_text().splitlines()[-10:]:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

