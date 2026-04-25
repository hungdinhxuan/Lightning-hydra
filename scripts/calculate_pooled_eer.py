#!/usr/bin/env python3
"""Multi-dataset pooled evaluation for benchmark outputs."""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from benchmark_py.binary_eval import (
    build_eval_frame,
    compute_legacy_compatibility_metrics,
    dumps_json,
    evaluate_binary_classification,
    load_eval_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate pooled raw/balanced metrics from benchmark score files."
    )
    parser.add_argument("results_folder", help="Folder containing per-dataset score files")
    parser.add_argument("normalized_yaml", help="Normalized yaml name used in score filenames")
    parser.add_argument("comment", help="Benchmark comment used in score filenames")
    parser.add_argument("benchmark_folders", nargs="+", help="Dataset folders containing protocol.txt")
    parser.add_argument(
        "--output-format",
        choices=["legacy", "json", "table"],
        default="legacy",
        help="legacy preserves the original five-value stdout format.",
    )
    parser.add_argument(
        "--protocol-subset",
        default="eval",
        help="Preferred protocol subset to load when present.",
    )
    parser.add_argument(
        "--include-best-f1-threshold",
        action="store_true",
        help="Also select and report the best-F1 threshold using pooled test data as a fallback source.",
    )
    parser.add_argument(
        "--eval-config",
        default=None,
        help="Optional eval yaml with per-dataset fill_policy.",
    )
    return parser.parse_args()


def collect_frames(
    results_folder: Path,
    normalized_yaml: str,
    comment: str,
    benchmark_folders: List[str],
    protocol_subset: str,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    frames: List[pd.DataFrame] = []
    dataset_stats: List[Dict[str, Any]] = []

    for benchmark_folder in benchmark_folders:
        benchmark_path = Path(benchmark_folder)
        dataset_name = benchmark_path.name
        score_file = results_folder / f"{dataset_name}_{normalized_yaml}_{comment}.txt"
        protocol_file = benchmark_path / "protocol.txt"

        if not score_file.exists() or not protocol_file.exists():
            print(
                f"Skipping {dataset_name}: missing score or protocol file "
                f"(score={score_file.exists()}, protocol={protocol_file.exists()})",
                file=sys.stderr,
            )
            continue

        frame = build_eval_frame(
            score_file=score_file,
            protocol_file=protocol_file,
            dataset_name=dataset_name,
            preferred_subset=protocol_subset,
            fallback_to_all=True,
        )

        dataset_stats.append(
            {
                "dataset_name": dataset_name,
                "n_samples": int(len(frame)),
                "n_bonafide": int((frame["label"] == 1).sum()) if not frame.empty else 0,
                "n_spoof": int((frame["label"] == 0).sum()) if not frame.empty else 0,
                "score_file": str(score_file),
                "protocol_file": str(protocol_file),
            }
        )

        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(
            columns=["filename", "subset", "dataset_name", "label_name", "label", "spoof_score", "score"]
        ), dataset_stats

    return pd.concat(frames, ignore_index=True), dataset_stats


def main() -> None:
    args = parse_args()
    results_folder = Path(args.results_folder)

    frame, dataset_stats = collect_frames(
        results_folder=results_folder,
        normalized_yaml=args.normalized_yaml,
        comment=args.comment,
        benchmark_folders=args.benchmark_folders,
        protocol_subset=args.protocol_subset,
    )

    if frame.empty:
        raise SystemExit("No valid datasets found for pooled evaluation.")

    legacy = compute_legacy_compatibility_metrics(frame)
    results = evaluate_binary_classification(
        test_frame=frame,
        validation_frame=None,
        include_best_f1_threshold=args.include_best_f1_threshold,
        allow_test_threshold_fallback=True,
        fill_policy=load_eval_config(Path(args.eval_config)).get("fill_policy") if args.eval_config else None,
    )
    payload = {
        "dataset_stats": dataset_stats,
        "legacy_compat": legacy,
        "results": results,
    }

    if args.output_format == "table":
        print(results["summary_text"])
        return

    if args.output_format == "json":
        print(dumps_json(payload))
        return

    print(
        f"{legacy['min_score']:.6f} "
        f"{legacy['max_score']:.6f} "
        f"{legacy['threshold']:.6f} "
        f"{legacy['eer']:.6f} "
        f"{legacy['accuracy']:.6f}"
    )


if __name__ == "__main__":
    main()
