#!/usr/bin/env python3
"""Submit benchmark jobs to persistent benchmark worker queue."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark_service.queue import FileJobQueue
from src.benchmark_service.schemas import BenchmarkJob


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit persistent benchmark service job")
    parser.add_argument("--queue-dir", default=".benchmark_service_queue")
    parser.add_argument("-g", "--gpu", default="0")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-b", "--benchmark-folder", required=True, type=Path)
    parser.add_argument("-m", "--model-path", required=True)
    parser.add_argument("-r", "--results-folder", required=True, type=Path)
    parser.add_argument("-n", "--comment", required=True)
    parser.add_argument("-a", "--adapter-paths", default=None)
    parser.add_argument("-l", "--is-ln", default=True, type=lambda x: str(x).lower() == "true")
    parser.add_argument("-s", "--random-start", default=True, type=lambda x: str(x).lower() == "true")
    parser.add_argument("-t", "--trim-length", default=64000, type=int)
    parser.add_argument("-z", "--batch-size", default=128, type=int)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--eval-config", default=None, type=Path)
    parser.add_argument(
        "--missing-protocol-label",
        default="skip",
        choices=["ask", "skip", "auto", "spoof", "bonafide"],
    )
    args, extra_overrides = parser.parse_known_args()
    args.extra_overrides = extra_overrides
    return args


def main() -> int:
    args = parse_args()
    job = BenchmarkJob(
        dataset_path=str(args.benchmark_folder),
        result_dir=str(args.results_folder),
        run_name=args.comment,
        config_path=args.config,
        model_path=args.model_path,
        gpu_id=args.gpu,
        adapter_path=args.adapter_paths,
        batch_size=args.batch_size,
        precision=args.precision,
        is_ln=args.is_ln,
        random_start=args.random_start,
        trim_length=args.trim_length,
        eval_config=str(args.eval_config) if args.eval_config else None,
        missing_protocol_label=args.missing_protocol_label,
        extra_overrides=args.extra_overrides,
    )
    path = FileJobQueue(args.queue_dir).submit(job)
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
