import math
import sys
from pathlib import Path

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


from benchmark_py.binary_eval import evaluate_binary_classification


def make_frame(rows):
    return pd.DataFrame(
        rows,
        columns=["filename", "subset", "dataset_name", "label_name", "label", "spoof_score", "score"],
    )


def test_validation_thresholds_and_one_class_nan():
    validation_frame = make_frame(
        [
            ("va_bf_1", "dev", "validation", "bonafide", 1, 0.1, 0.90),
            ("va_bf_2", "dev", "validation", "bonafide", 1, 0.2, 0.80),
            ("va_sp_1", "dev", "validation", "spoof", 0, 0.8, 0.40),
            ("va_sp_2", "dev", "validation", "spoof", 0, 0.7, 0.30),
        ]
    )

    test_frame = make_frame(
        [
            ("a_bf_1", "eval", "dataset_a", "bonafide", 1, 0.1, 0.95),
            ("a_bf_2", "eval", "dataset_a", "bonafide", 1, 0.2, 0.85),
            ("a_sp_1", "eval", "dataset_a", "spoof", 0, 0.9, 0.35),
            ("a_sp_2", "eval", "dataset_a", "spoof", 0, 0.8, 0.25),
            ("b_sp_1", "eval", "spoof_only", "spoof", 0, 0.7, 0.10),
            ("b_sp_2", "eval", "spoof_only", "spoof", 0, 0.8, 0.15),
        ]
    )

    results = evaluate_binary_classification(
        test_frame=test_frame,
        validation_frame=validation_frame,
        include_best_f1_threshold=True,
        allow_test_threshold_fallback=False,
    )

    eer_threshold = results["validation"]["thresholds"]["eer_threshold"]["threshold"]
    assert results["validation"]["threshold_source"] == "validation"
    assert math.isfinite(eer_threshold)

    dataset_a = next(record for record in results["threshold_free"]["per_dataset"] if record["dataset_name"] == "dataset_a")
    spoof_only = next(record for record in results["threshold_free"]["per_dataset"] if record["dataset_name"] == "spoof_only")

    assert math.isfinite(dataset_a["threshold_free"]["eer"])
    assert math.isfinite(dataset_a["threshold_free"]["roc_auc"])
    assert dataset_a["threshold_metrics"]["eer_threshold"]["threshold"] == eer_threshold

    assert math.isnan(spoof_only["threshold_free"]["eer"])
    assert math.isnan(spoof_only["threshold_free"]["roc_auc"])
    assert "only one class is present" in spoof_only["threshold_free"]["reason"]

    expected_macro = dataset_a["threshold_free"]["eer"]
    assert math.isclose(results["threshold_free"]["macro_average_eer"], expected_macro, rel_tol=1e-9)

    far_1pct = results["threshold_based"]["far_1pct_threshold"]["raw_pooled"]
    assert math.isfinite(far_1pct["mdr"])


def test_threshold_metrics_include_class_accuracy_in_detail_text():
    validation_frame = make_frame(
        [
            ("va_bf_1", "dev", "validation", "bonafide", 1, 0.1, 0.80),
            ("va_sp_1", "dev", "validation", "spoof", 0, 0.8, 0.20),
        ]
    )
    test_frame = make_frame(
        [
            ("bf_ok", "eval", "dataset_a", "bonafide", 1, 0.1, 0.90),
            ("bf_miss", "eval", "dataset_a", "bonafide", 1, 0.6, 0.10),
            ("sp_ok", "eval", "dataset_a", "spoof", 0, 0.9, 0.20),
            ("sp_false_accept", "eval", "dataset_a", "spoof", 0, 0.2, 0.85),
        ]
    )

    results = evaluate_binary_classification(
        test_frame=test_frame,
        validation_frame=validation_frame,
        allow_test_threshold_fallback=False,
    )
    metrics = results["threshold_based"]["eer_threshold"]["per_dataset"][0]
    threshold_free = results["threshold_free"]["per_dataset"][0]["threshold_free"]

    assert math.isclose(metrics["bonafide_accuracy"], 0.5)
    assert math.isclose(metrics["spoof_accuracy"], 0.5)
    assert math.isclose(threshold_free["bonafide_accuracy"], 0.5)
    assert math.isclose(threshold_free["spoof_accuracy"], 0.5)
    assert "bonafide_accuracy" in results["summary_text"]
    assert "spoof_accuracy" in results["summary_text"]


def test_balanced_pooled_differs_from_raw_pooled():
    large_easy_rows = []
    for index in range(40):
        large_easy_rows.append((f"easy_bf_{index}", "eval", "large_easy", "bonafide", 1, 0.1, 0.95))
        large_easy_rows.append((f"easy_sp_{index}", "eval", "large_easy", "spoof", 0, 0.9, 0.05))

    small_hard_rows = [
        ("hard_bf_1", "eval", "small_hard", "bonafide", 1, 0.1, 0.45),
        ("hard_bf_2", "eval", "small_hard", "bonafide", 1, 0.2, 0.40),
        ("hard_sp_1", "eval", "small_hard", "spoof", 0, 0.8, 0.60),
        ("hard_sp_2", "eval", "small_hard", "spoof", 0, 0.7, 0.55),
    ]

    test_frame = make_frame(large_easy_rows + small_hard_rows)
    results = evaluate_binary_classification(test_frame=test_frame, validation_frame=None)

    raw_eer = results["threshold_free"]["raw_pooled"]["threshold_free"]["eer"]
    balanced_eer = results["threshold_free"]["balanced_pooled"]["threshold_free"]["eer"]

    assert math.isfinite(raw_eer)
    assert math.isfinite(balanced_eer)
    assert balanced_eer > raw_eer


def test_fill_policy_completes_missing_class_for_per_dataset_eer():
    test_frame = make_frame(
        [
            ("src_bf_1", "eval", "source_bona", "bonafide", 1, 0.1, 0.90),
            ("src_bf_2", "eval", "source_bona", "bonafide", 1, 0.1, 0.85),
            ("tgt_sp_1", "eval", "target_spoof_only", "spoof", 0, 0.8, 0.30),
            ("tgt_sp_2", "eval", "target_spoof_only", "spoof", 0, 0.7, 0.25),
        ]
    )

    results = evaluate_binary_classification(
        test_frame=test_frame,
        validation_frame=None,
        fill_policy={
            "target_spoof_only": {
                "bonafide_source": "source_bona",
                "spoof_source": None,
            }
        },
    )

    target_record = next(
        record for record in results["threshold_free"]["per_dataset"]
        if record["dataset_name"] == "target_spoof_only"
    )

    assert math.isfinite(target_record["threshold_free"]["eer"])
    assert math.isfinite(target_record["threshold_free"]["roc_auc"])
    assert target_record["completion"]["used_fill"] is True
    assert target_record["completion"]["filled_counts"]["bonafide"] == 2
    assert "filled bonafide from source_bona" in target_record["threshold_free"]["note"]
    assert "min_score" not in results["summary_text"]
