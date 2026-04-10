#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys


def run_step(name: str, command: list[str]) -> None:
    print(f"[run_trial] {name}: {' '.join(command)}")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed ({result.returncode})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--skip-privacy-eval", action="store_true")
    args = parser.parse_args()

    run_step(
        "small-test",
        [
            "python",
            "src/train.py",
            f"experiment={args.experiment}",
            "trainer.max_epochs=2",
            "data.batch_size=16",
        ],
    )
    run_step(
        "full-training",
        [
            "python",
            "src/train.py",
            f"experiment={args.experiment}",
        ],
    )
    run_step("cm-eval", ["python", "src/eval.py", "ckpt_path=last"])
    if not args.skip_privacy_eval:
        run_step("privacy-eval", ["python", "-c", "print('privacy eval placeholder')"])
    run_step("finalize", ["python", "scripts/finalize_trial.py", "--trial-id", args.trial_id])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[run_trial] FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)

