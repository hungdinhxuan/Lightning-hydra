"""Tests for scripts/evaluate_trial_model.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_trial_model  # noqa: E402


def test_extract_averaged_ckpt_strips_ansi(tmp_path: Path) -> None:
    log = tmp_path / "t.log"
    log.write_text(
        "[\x1b[32mINFO\x1b[0m] Created averaged checkpoint: "
        "/tmp/run/checkpoints/averaged_top5.ckpt\x1b[0m\n",
        encoding="utf-8",
    )
    assert (
        evaluate_trial_model.extract_averaged_ckpt(log)
        == "/tmp/run/checkpoints/averaged_top5.ckpt"
    )


def test_extract_averaged_ckpt_last_wins(tmp_path: Path) -> None:
    log = tmp_path / "t.log"
    log.write_text(
        "Created averaged checkpoint: /first/a.ckpt\n"
        "noise\n"
        "Created averaged checkpoint: /second/b.ckpt\n",
        encoding="utf-8",
    )
    assert evaluate_trial_model.extract_averaged_ckpt(log) == "/second/b.ckpt"


def test_find_trial(tmp_path: Path) -> None:
    q = tmp_path / "q.jsonl"
    q.write_text(
        json.dumps({"trial_id": "a", "x": 1}) + "\n"
        + json.dumps({"trial_id": "b", "log_file": "l.log", "gpu_id": "MIG-1"}) + "\n",
        encoding="utf-8",
    )
    rows = evaluate_trial_model.load_queue(q)
    assert evaluate_trial_model.find_trial(rows, "b")["gpu_id"] == "MIG-1"
