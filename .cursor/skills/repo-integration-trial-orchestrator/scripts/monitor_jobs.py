#!/usr/bin/env python3
"""
Monitor queue, schedule GPU, detect failures, emit events.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EVENT_RESULT_READY = "RESULT_READY"
EVENT_FAILED = "FAILED"
EVENT_NEED_DECISION = "NEED_DECISION"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-file", default="queue.jsonl")
    parser.add_argument("--events-file", default="events.jsonl")
    args = parser.parse_args()

    queue_path = Path(args.queue_file)
    events_path = Path(args.events_file)
    if not queue_path.exists():
        print(f"[monitor_jobs] queue not found: {queue_path}")
        return 0

    lines = queue_path.read_text().splitlines()
    records = [json.loads(line) for line in lines if line.strip()]

    # Placeholder scheduler: pick first pending entry.
    pending = next((r for r in records if r.get("status") == "pending"), None)
    if pending is None:
        print("[monitor_jobs] no pending tasks")
        return 0

    # Placeholder event emission until integrated with real runner status.
    event = {
        "trial_id": pending["trial_id"],
        "event": EVENT_NEED_DECISION,
        "reason": "placeholder scheduler needs project-specific GPU/job wiring",
    }
    with events_path.open("a") as f:
        f.write(json.dumps(event) + "\n")
    print(f"[monitor_jobs] emitted {event['event']} for {event['trial_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
