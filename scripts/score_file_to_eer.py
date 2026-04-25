#!/usr/bin/env python
"""Binary spoof/bonafide evaluator."""

import argparse
import os.path
from pathlib import Path

from benchmark_py.binary_eval import (
    build_eval_frame,
    compute_legacy_compatibility_metrics,
    dumps_json,
    evaluate_binary_classification,
)


def _safe_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate binary bonafide/spoof scores with optional validation thresholds."
    )
    parser.add_argument("score_file", help="Test score file")
    parser.add_argument("protocol_file", help="Test protocol file")
    parser.add_argument(
        "--validation-score-file",
        help="Validation score file used to select thresholds. If omitted, test data is used as a legacy fallback.",
    )
    parser.add_argument(
        "--validation-protocol-file",
        help="Validation protocol file. Required when --validation-score-file is set.",
    )
    parser.add_argument(
        "--output-format",
        choices=["legacy", "json", "table"],
        default="legacy",
        help="legacy keeps the original five-value stdout format.",
    )
    parser.add_argument(
        "--protocol-subset",
        default=None,
        help="Optional preferred subset name (for example: eval or dev). Falls back to all rows if the subset is absent.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional dataset name override for summary output.",
    )
    parser.add_argument(
        "--include-best-f1-threshold",
        action="store_true",
        help="Also select and report the best-F1 threshold on validation.",
    )
    parser.add_argument(
        "--no-test-threshold-fallback",
        action="store_true",
        help="Do not select thresholds from test data when validation data is missing.",
    )
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    required_paths = [args.score_file, args.protocol_file]
    if args.validation_score_file or args.validation_protocol_file:
        if not (args.validation_score_file and args.validation_protocol_file):
            raise SystemExit("--validation-score-file and --validation-protocol-file must be provided together.")
        required_paths.extend([args.validation_score_file, args.validation_protocol_file])

    for path_text in required_paths:
        if not os.path.isfile(path_text):
            raise SystemExit(f"{path_text} doesn't exist")


def main() -> None:
    args = parse_args()
    validate_paths(args)

    test_frame = build_eval_frame(
        score_file=Path(args.score_file),
        protocol_file=Path(args.protocol_file),
        dataset_name=args.dataset_name,
        preferred_subset=args.protocol_subset,
        fallback_to_all=True,
    )

    validation_frame = None
    if args.validation_score_file and args.validation_protocol_file:
        validation_frame = build_eval_frame(
            score_file=Path(args.validation_score_file),
            protocol_file=Path(args.validation_protocol_file),
            dataset_name=args.dataset_name or "validation",
            preferred_subset=args.protocol_subset,
            fallback_to_all=True,
        )

    legacy = compute_legacy_compatibility_metrics(test_frame)
    if args.output_format == "legacy" and validation_frame is None and not args.include_best_f1_threshold:
        print(
            f"{_safe_float(legacy.get('min_score')):.6f} "
            f"{_safe_float(legacy.get('max_score')):.6f} "
            f"{_safe_float(legacy.get('threshold')):.6f} "
            f"{_safe_float(legacy.get('eer')):.6f} "
            f"{_safe_float(legacy.get('accuracy')):.6f}"
        )
        return

    results = evaluate_binary_classification(
        test_frame=test_frame,
        validation_frame=validation_frame,
        include_best_f1_threshold=args.include_best_f1_threshold,
        allow_test_threshold_fallback=not args.no_test_threshold_fallback,
    )
    payload = {
        "legacy_compat": legacy,
        "results": results,
    }

    if args.output_format == "table":
        print(results["summary_text"])
        return

    if args.output_format == "legacy":
        eer_threshold = results["validation"]["thresholds"].get("eer_threshold", {})
        accuracy = results["threshold_based"].get("eer_threshold", {}).get("raw_pooled", {}).get("accuracy")
        accuracy_percent = _safe_float(accuracy) * 100.0
        pooled_eer_percent = _safe_float(results['threshold_free']['raw_pooled']['threshold_free'].get('eer')) * 100.0
        print(
            f"{_safe_float(legacy.get('min_score')):.6f} "
            f"{_safe_float(legacy.get('max_score')):.6f} "
            f"{_safe_float(eer_threshold.get('threshold')):.6f} "
            f"{pooled_eer_percent:.6f} "
            f"{accuracy_percent:.6f}"
        )
        return

    print(dumps_json(payload))


if __name__ == "__main__":
    main()
