#!/usr/bin/env python3
"""
Finalize trial:
- compute metrics
- write report
- update memory
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--memory-file", default="memory-db.md")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{args.trial_id}.md"
    report_path.write_text(
        f"# Trial Report: {args.trial_id}\n\n"
        "## Metrics\n"
        "- TODO: compute and write project-specific metrics\n"
    )

    timestamp = datetime.now(timezone.utc).isoformat()
    memory_entry = (
        f"## {timestamp} | {args.trial_id} | finalize\n"
        "Decision:\n"
        "- finalize placeholder completed\n"
        "Evidence:\n"
        f"- report: {report_path}\n"
        "Artifacts:\n"
        f"- {report_path}\n"
        "Blockers:\n"
        "- none\n"
        "Next:\n"
        "- review report and decide follow-up trial\n"
    )

    memory_path = Path(args.memory_file)
    with memory_path.open("a") as f:
        f.write(memory_entry + "\n")

    print(f"[finalize_trial] wrote {report_path} and updated {memory_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
