#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--report-dir", default="plans/01042026/reports")
    parser.add_argument("--memory-file", default="plans/01042026/memory-db.md")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{args.trial_id}.md"
    report_path.write_text(
        f"# Trial Report: {args.trial_id}\n\n"
        "## Metrics\n"
        "- TODO: populate EER/CM metrics\n",
        encoding="utf-8",
    )

    ts = datetime.now(timezone.utc).isoformat()
    entry = (
        f"## {ts} | {args.trial_id} | finalize\n"
        "Decision:\n"
        "- finalize executed\n"
        "Evidence:\n"
        f"- report: {report_path}\n"
        "Artifacts:\n"
        f"- {report_path}\n"
        "Blockers:\n"
        "- none\n"
        "Next:\n"
        "- review report and queue follow-up if needed\n"
    )
    with Path(args.memory_file).open("a", encoding="utf-8") as f:
        f.write(entry + "\n")
    print(f"[finalize_trial] wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

