import json
import math

from scripts.benchmark_py.rank_may2026_threshold_free_models import (
    DEFAULT_PHONE_DATASET,
    DEFAULT_VIDEO_DATASET,
    filter_rows,
    load_model_metrics,
    pareto_front,
    ranked_rows,
)


def write_summary(path, phone_bonafide, video_accuracy):
    payload = {
        "type": "pooled",
        "payload": {
            "results": {
                "threshold_free": {
                    "per_dataset": [
                        {
                            "dataset_name": DEFAULT_PHONE_DATASET,
                            "n_bonafide": 10,
                            "n_spoof": 20,
                            "threshold_free": {
                                "accuracy": 0.95,
                                "bonafide_accuracy": phone_bonafide,
                                "spoof_accuracy": 0.9,
                            },
                        },
                        {
                            "dataset_name": DEFAULT_VIDEO_DATASET,
                            "n_bonafide": 0,
                            "n_spoof": 5,
                            "threshold_free": {
                                "accuracy": video_accuracy,
                                "bonafide_accuracy": math.nan,
                                "spoof_accuracy": video_accuracy,
                            },
                        },
                    ]
                }
            }
        },
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_load_model_metrics_uses_phone_bonafide_accuracy_and_video_accuracy(tmp_path):
    model_dir = tmp_path / "model-a"
    model_dir.mkdir()
    write_summary(model_dir / "summary_results_details.jsonl", phone_bonafide=0.81, video_accuracy=0.74)

    row = load_model_metrics(model_dir, DEFAULT_PHONE_DATASET, DEFAULT_VIDEO_DATASET)

    assert row is not None
    assert row["phone_bonafide_accuracy"] == 0.81
    assert row["video_accuracy"] == 0.74
    assert row["average_requested_metrics"] == 0.775
    assert row["min_requested_metric"] == 0.74


def test_ranking_and_pareto_front_show_tradeoffs(tmp_path):
    rows = []
    for name, phone_bonafide, video_accuracy in [
        ("phone-leader", 0.99, 0.70),
        ("balanced", 0.86, 0.85),
        ("dominated", 0.80, 0.75),
    ]:
        model_dir = tmp_path / name
        model_dir.mkdir()
        write_summary(model_dir / "summary_results_details.jsonl", phone_bonafide, video_accuracy)
        rows.append(load_model_metrics(model_dir, DEFAULT_PHONE_DATASET, DEFAULT_VIDEO_DATASET))

    assert [row["model"] for row in ranked_rows(rows, "average")] == [
        "balanced",
        "phone-leader",
        "dominated",
    ]
    assert [row["model"] for row in pareto_front(rows)] == ["phone-leader", "balanced"]


def test_filter_rows_keeps_strictly_greater_phone_bonafide(tmp_path):
    rows = []
    for name, phone_bonafide, video_accuracy in [
        ("exact-threshold", 0.95, 0.99),
        ("over-threshold", 0.951, 0.80),
        ("under-threshold", 0.949, 0.90),
    ]:
        model_dir = tmp_path / name
        model_dir.mkdir()
        write_summary(model_dir / "summary_results_details.jsonl", phone_bonafide, video_accuracy)
        rows.append(load_model_metrics(model_dir, DEFAULT_PHONE_DATASET, DEFAULT_VIDEO_DATASET))

    filtered = filter_rows(rows, min_phone_bonafide_pct=95.0)

    assert [row["model"] for row in filtered] == ["over-threshold"]
