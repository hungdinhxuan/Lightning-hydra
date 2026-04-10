#!/usr/bin/env python3
"""
Pipeline:
1) small test
2) full training
3) CM eval
4) privacy eval (optional)
5) finalize
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def run_step(name: str, cmd: list[str]) -> None:
    print(f"[run_trial] {name}: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with code {result.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--skip-privacy-eval", action="store_true")
    args = parser.parse_args()

    # Replace placeholder commands with project-specific commands.
    run_step("small-test", ["python", "-c", "print('small test placeholder')"])
    run_step("full-training", ["python", "-c", "print('full training placeholder')"])
    run_step("cm-eval", ["python", "-c", "print('cm eval placeholder')"])

    if not args.skip_privacy_eval:
        run_step("privacy-eval", ["python", "-c", "print('privacy eval placeholder')"])

    run_step("finalize", ["python", "scripts/finalize_trial.py", "--trial-id", args.trial_id])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"[run_trial] FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
