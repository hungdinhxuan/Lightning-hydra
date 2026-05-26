#!/usr/bin/env python3
"""Rank May 2026 benchmark runs by selected Threshold-Free dataset metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_RESULTS_ROOT = Path("logs/results/May_2026_benchmark")
DEFAULT_PHONE_DATASET = "1-phone_large-corpus"
DEFAULT_VIDEO_DATASET = "May_08_2026_seonghak_spoof_video_converted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank benchmark result folders by Threshold-Free "
            "1-phone bonafide_accuracy and video_converted accuracy."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Benchmark results root. Default: {DEFAULT_RESULTS_ROOT}",
    )
    parser.add_argument(
        "--phone-dataset",
        default=DEFAULT_PHONE_DATASET,
        help=f"Dataset whose bonafide_accuracy is used. Default: {DEFAULT_PHONE_DATASET}",
    )
    parser.add_argument(
        "--video-dataset",
        default=DEFAULT_VIDEO_DATASET,
        help=f"Dataset whose accuracy is used. Default: {DEFAULT_VIDEO_DATASET}",
    )
    parser.add_argument(
        "--sort",
        choices=["average", "min", "phone_bonafide", "video_accuracy"],
        default="video_accuracy",
        help="Main ranking key. Default: video_accuracy.",
    )
    parser.add_argument(
        "--min-phone-bonafide-pct",
        type=float,
        default=None,
        help="Strict lower bound for phone bonafide_accuracy, in percent. Example: 95 means >95%%.",
    )
    parser.add_argument("--top", type=int, default=20, help="Number of ranked rows to print.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path.")
    return parser.parse_args()


def find_result_dirs(results_root: Path) -> Iterable[Path]:
    for child in sorted(results_root.iterdir()):
        if child.is_dir() and (child / "summary_results_details.jsonl").exists():
            yield child


def iter_threshold_free_dataset_items(summary_path: Path) -> Iterable[Dict[str, Any]]:
    with summary_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            record = json.loads(raw_line)
            results = record.get("payload", {}).get("results", {})
            threshold_free = results.get("threshold_free", {})
            for item in threshold_free.get("per_dataset", []) or []:
                if isinstance(item, dict) and item.get("dataset_name"):
                    yield item


def load_model_metrics(model_dir: Path, phone_dataset: str, video_dataset: str) -> Optional[Dict[str, Any]]:
    summary_path = model_dir / "summary_results_details.jsonl"
    phone_item: Optional[Dict[str, Any]] = None
    video_item: Optional[Dict[str, Any]] = None

    for item in iter_threshold_free_dataset_items(summary_path):
        if item.get("dataset_name") == phone_dataset:
            phone_item = item
        elif item.get("dataset_name") == video_dataset:
            video_item = item

    if phone_item is None or video_item is None:
        return None

    phone_metrics = phone_item.get("threshold_free", {}) or {}
    video_metrics = video_item.get("threshold_free", {}) or {}
    phone_bonafide = safe_float(phone_metrics.get("bonafide_accuracy"))
    video_accuracy = safe_float(video_metrics.get("accuracy"))
    if math.isnan(phone_bonafide) or math.isnan(video_accuracy):
        return None

    return {
        "model": model_dir.name,
        "phone_dataset": phone_dataset,
        "phone_n_bonafide": phone_item.get("n_bonafide"),
        "phone_n_spoof": phone_item.get("n_spoof"),
        "phone_accuracy": safe_float(phone_metrics.get("accuracy")),
        "phone_bonafide_accuracy": phone_bonafide,
        "phone_spoof_accuracy": safe_float(phone_metrics.get("spoof_accuracy")),
        "video_dataset": video_dataset,
        "video_n_bonafide": video_item.get("n_bonafide"),
        "video_n_spoof": video_item.get("n_spoof"),
        "video_accuracy": video_accuracy,
        "video_spoof_accuracy": safe_float(video_metrics.get("spoof_accuracy")),
        "average_requested_metrics": (phone_bonafide + video_accuracy) / 2.0,
        "min_requested_metric": min(phone_bonafide, video_accuracy),
        "summary_file": str(summary_path),
    }


def safe_float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def ranked_rows(rows: List[Dict[str, Any]], sort_key: str) -> List[Dict[str, Any]]:
    key_map = {
        "average": "average_requested_metrics",
        "min": "min_requested_metric",
        "phone_bonafide": "phone_bonafide_accuracy",
        "video_accuracy": "video_accuracy",
    }
    primary = key_map[sort_key]
    return sorted(
        rows,
        key=lambda row: (
            row[primary],
            row["min_requested_metric"],
            row["phone_bonafide_accuracy"],
            row["video_accuracy"],
            row["model"],
        ),
        reverse=True,
    )


def filter_rows(rows: List[Dict[str, Any]], min_phone_bonafide_pct: Optional[float]) -> List[Dict[str, Any]]:
    if min_phone_bonafide_pct is None:
        return rows
    min_phone_bonafide = min_phone_bonafide_pct / 100.0
    return [row for row in rows if row["phone_bonafide_accuracy"] > min_phone_bonafide]


def pareto_front(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    front: List[Dict[str, Any]] = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            phone_ge = other["phone_bonafide_accuracy"] >= row["phone_bonafide_accuracy"]
            video_ge = other["video_accuracy"] >= row["video_accuracy"]
            strictly_better = (
                other["phone_bonafide_accuracy"] > row["phone_bonafide_accuracy"]
                or other["video_accuracy"] > row["video_accuracy"]
            )
            if phone_ge and video_ge and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(row)
    return ranked_rows(front, "phone_bonafide")


def pct(value: Any) -> str:
    value = safe_float(value)
    if math.isnan(value):
        return "NaN"
    return f"{value * 100.0:.2f}%"


def print_table(title: str, rows: List[Dict[str, Any]], limit: Optional[int] = None) -> None:
    selected = rows[:limit] if limit is not None else rows
    print(f"\n{title}")
    print("rank | phone_bonafide | video_accuracy | avg | min | model")
    print("-----+-----------------+----------------+-----+-----+------")
    for index, row in enumerate(selected, start=1):
        print(
            f"{index:>4} | {pct(row['phone_bonafide_accuracy']):>15} | "
            f"{pct(row['video_accuracy']):>14} | {pct(row['average_requested_metrics']):>5} | "
            f"{pct(row['min_requested_metric']):>5} | {row['model']}"
        )


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "model",
        "phone_dataset",
        "phone_n_bonafide",
        "phone_n_spoof",
        "phone_accuracy",
        "phone_bonafide_accuracy",
        "phone_spoof_accuracy",
        "video_dataset",
        "video_n_bonafide",
        "video_n_spoof",
        "video_accuracy",
        "video_spoof_accuracy",
        "average_requested_metrics",
        "min_requested_metric",
        "summary_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    results_root = args.results_root.resolve()
    rows = [
        metrics
        for model_dir in find_result_dirs(results_root)
        if (metrics := load_model_metrics(model_dir, args.phone_dataset, args.video_dataset)) is not None
    ]
    if not rows:
        raise RuntimeError(f"No comparable result folders found under {results_root}")

    filtered_rows = filter_rows(rows, args.min_phone_bonafide_pct)
    if not filtered_rows:
        raise RuntimeError(
            f"No result folders matched phone bonafide_accuracy > {args.min_phone_bonafide_pct:.2f}%"
        )

    ranked = ranked_rows(filtered_rows, args.sort)
    print(f"Results root: {results_root}")
    print(f"Phone criterion: {args.phone_dataset} Threshold-Free bonafide_accuracy")
    print(f"Video criterion: {args.video_dataset} Threshold-Free accuracy")
    print(f"Comparable models: {len(rows)}")
    if args.min_phone_bonafide_pct is not None:
        print(f"Phone bonafide filter: > {args.min_phone_bonafide_pct:.2f}%")
        print(f"Models after filter: {len(filtered_rows)}")
    print_table(f"Top by {args.sort}", ranked, args.top)
    print_table("Top by phone bonafide_accuracy", ranked_rows(filtered_rows, "phone_bonafide"), min(args.top, 10))
    print_table("Top by video accuracy", ranked_rows(filtered_rows, "video_accuracy"), min(args.top, 10))
    print_table("Pareto front", pareto_front(filtered_rows), None)

    if args.csv:
        write_csv(args.csv, ranked)
        print(f"\nWrote CSV: {args.csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
