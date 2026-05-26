#!/usr/bin/env python3
"""Migrate Weights & Biases runs to a local or remote MLflow tracking server.

Remote setup (in .env or shell):
  MLFLOW_TRACKING_URI=https://mlflow.example.com          # required
  MLFLOW_TRACKING_TOKEN=...                               # bearer token (if used)
  MLFLOW_TRACKING_USERNAME=...                            # basic auth (if used)
  MLFLOW_TRACKING_PASSWORD=...
  MLFLOW_TRACKING_INSECURE_TLS=true                       # self-signed HTTPS only
  MLFLOW_EXPERIMENT_NAME=CNSL2026

Artifacts are uploaded through the tracking server (S3/GCS/Azure must be
configured on the server). Use --no-artifacts for metrics/params-only migration.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from typing import Any, Iterator

import dotenv
import mlflow
import pandas as pd
import wandb
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient

dotenv.load_dotenv()

from src.utils.mlflow_auth import register_dex_auth_for_mlflow

LOGGER = logging.getLogger(__name__)

METRIC_SKIP_COLUMNS = {"_step", "_timestamp", "_runtime"}
MLFLOW_PARAM_MAX_LEN = 500
METRIC_BATCH_SIZE = 1000
WANDB_INTERNAL_PREFIXES = ("_", "wandb_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate W&B runs to MLflow.")
    parser.add_argument(
        "--entity",
        default=os.getenv("WANDB_ENTITY"),
        help="W&B entity (team/user). Defaults to WANDB_ENTITY or your W&B default.",
    )
    parser.add_argument(
        "--project",
        default=os.getenv("WANDB_PROJECT"),
        help="W&B project. Defaults to WANDB_PROJECT.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI"),
        help=(
            "Remote MLflow server, e.g. https://mlflow.example.com. "
            "Defaults to MLFLOW_TRACKING_URI (not a local file path)."
        ),
    )
    parser.add_argument(
        "--experiment",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "Migrated_WandB"),
        help="MLflow experiment name. Defaults to MLFLOW_EXPERIMENT_NAME or Migrated_WandB.",
    )
    parser.add_argument(
        "--state",
        default="finished",
        help="W&B run state filter (e.g. finished, running, crashed). Use 'any' for all states.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Migrate at most this many W&B runs (newest first).",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip runs whose wandb_id tag already exists in the target experiment.",
    )
    parser.add_argument(
        "--artifacts",
        action="store_true",
        help="Download W&B run files and log them as MLflow artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List runs that would be migrated without writing to MLflow.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Test MLflow connectivity and exit (no W&B migration).",
    )
    return parser.parse_args()


def require(value: str | None, name: str) -> str:
    if not value:
        LOGGER.error("Missing required value: %s", name)
        sys.exit(1)
    return value


def is_remote_tracking_uri(tracking_uri: str) -> bool:
    lowered = tracking_uri.strip().lower()
    return lowered.startswith(("http://", "https://")) or lowered == "databricks"


def configure_remote_mlflow(tracking_uri: str) -> None:
    """Apply auth/TLS env vars that MLflow reads for remote tracking servers."""
    if not is_remote_tracking_uri(tracking_uri):
        LOGGER.warning(
            "Tracking URI %r looks local (file://). For a remote server use "
            "https://host:port and set MLFLOW_TRACKING_TOKEN or USERNAME/PASSWORD.",
            tracking_uri,
        )
        return

    auth_bits: list[str] = []
    if os.getenv("MLFLOW_TRACKING_TOKEN"):
        auth_bits.append("token")
    if os.getenv("MLFLOW_TRACKING_USERNAME"):
        auth_bits.append("basic-auth")
    if os.getenv("MLFLOW_TRACKING_INSECURE_TLS", "").lower() in {"1", "true", "yes"}:
        auth_bits.append("insecure-tls")

    if auth_bits:
        LOGGER.info("Remote MLflow auth: %s", ", ".join(auth_bits))
    else:
        LOGGER.warning(
            "No MLFLOW_TRACKING_TOKEN or MLFLOW_TRACKING_USERNAME set; "
            "the server must allow unauthenticated access."
        )

    register_dex_auth_for_mlflow(tracking_uri)


def verify_mlflow_connection(
    client: MlflowClient,
    experiment_name: str,
) -> str:
    """Create/resolve experiment and perform a lightweight API call."""
    experiment = mlflow.set_experiment(experiment_name)
    client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1,
    )
    return experiment.experiment_id


def flatten_config(
    config: dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    items: dict[str, Any] = {}
    for key, value in config.items():
        if any(str(key).startswith(prefix) for prefix in WANDB_INTERNAL_PREFIXES):
            continue
        full_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, dict):
            items.update(flatten_config(value, full_key, sep=sep))
        else:
            items[full_key] = value
    return items


def param_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, default=str)
    else:
        text = str(value)
    if len(text) > MLFLOW_PARAM_MAX_LEN:
        return text[: MLFLOW_PARAM_MAX_LEN - 3] + "..."
    return text


def metric_rows(history: pd.DataFrame) -> Iterator[tuple[int, str, float]]:
    for _, row in history.iterrows():
        step = int(row.get("_step", 0))
        for metric_name, value in row.items():
            if metric_name in METRIC_SKIP_COLUMNS or pd.isna(value):
                continue
            try:
                yield step, str(metric_name), float(value)
            except (TypeError, ValueError):
                LOGGER.debug("Skipping non-numeric metric %s at step %s", metric_name, step)


def log_metrics_batch(client: MlflowClient, run_id: str, history: pd.DataFrame) -> int:
    timestamp_ms = int(time.time() * 1000)
    metrics = [
        Metric(key=name, value=value, step=step, timestamp=timestamp_ms)
        for step, name, value in metric_rows(history)
    ]
    for offset in range(0, len(metrics), METRIC_BATCH_SIZE):
        client.log_batch(run_id, metrics=metrics[offset : offset + METRIC_BATCH_SIZE])
    return len(metrics)


def log_summary_metrics(
    client: MlflowClient,
    run_id: str,
    summary: dict[str, Any],
    final_step: int,
) -> int:
    timestamp_ms = int(time.time() * 1000)
    metrics: list[Metric] = []
    for name, value in summary.items():
        if any(str(name).startswith(prefix) for prefix in WANDB_INTERNAL_PREFIXES):
            continue
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        try:
            metrics.append(
                Metric(
                    key=str(name),
                    value=float(value),
                    step=final_step,
                    timestamp=timestamp_ms,
                )
            )
        except (TypeError, ValueError):
            continue
    if metrics:
        client.log_batch(run_id, metrics=metrics)
    return len(metrics)


def migrate_tags(wandb_run: Any) -> None:
    mlflow.set_tag("wandb_id", wandb_run.id)
    mlflow.set_tag("wandb_name", wandb_run.name)
    mlflow.set_tag("wandb_url", wandb_run.url)
    mlflow.set_tag("wandb_state", wandb_run.state)
    if wandb_run.notes:
        mlflow.set_tag("mlflow.note.content", wandb_run.notes)

    raw_tags = wandb_run.tags or []

    if isinstance(raw_tags, dict):
        for key, value in raw_tags.items():
            if value is None:
                continue
            mlflow.set_tag(f"wandb.{key}", str(value))
    else:
        tag_list = [str(t) for t in raw_tags if t]
        if tag_list:
            mlflow.set_tag("wandb.tags", ",".join(tag_list))


def migrate_artifacts(wandb_run: Any) -> int:
    files = list(wandb_run.files())
    if not files:
        return 0
    with tempfile.TemporaryDirectory(prefix="wandb_migrate_") as tmpdir:
        for wandb_file in files:
            wandb_file.download(root=tmpdir, replace=True)
        mlflow.log_artifacts(tmpdir)
    return len(files)


def wandb_run_path(entity: str | None, project: str) -> str:
    if entity:
        return f"{entity}/{project}"
    return project


def existing_wandb_ids(client: MlflowClient, experiment_id: str) -> set[str]:
    existing: set[str] = set()
    page_token: str | None = None
    while True:
        result = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="tags.wandb_id != ''",
            max_results=500,
            page_token=page_token,
        )
        for run in result:
            wandb_id = run.data.tags.get("wandb_id")
            if wandb_id:
                existing.add(wandb_id)
        page_token = result.token
        if page_token is None:
            break
    return existing


def migrate_run(
    client: MlflowClient,
    wandb_run: Any,
    *,
    migrate_artifacts_flag: bool,
) -> None:
    flat_config = flatten_config(dict(wandb_run.config or {}))
    safe_params = {key: param_value(value) for key, value in flat_config.items()}

    with mlflow.start_run(run_name=wandb_run.name):
        mlflow_run_id = mlflow.active_run().info.run_id

        if safe_params:
            mlflow.log_params(safe_params)

        migrate_tags(wandb_run)

        history = wandb_run.history(samples=10_000)
        metric_count = 0
        final_step = 0
        if not history.empty:
            final_step = int(history["_step"].max()) if "_step" in history.columns else 0
            metric_count = log_metrics_batch(client, mlflow_run_id, history)

        summary = dict(wandb_run.summary or {})
        summary_count = log_summary_metrics(client, mlflow_run_id, summary, final_step)

        artifact_count = 0
        if migrate_artifacts_flag:
            artifact_count = migrate_artifacts(wandb_run)

    LOGGER.info(
        "Migrated %s (%s): %d params, %d history metrics, %d summary metrics, %d files",
        wandb_run.name,
        wandb_run.id,
        len(safe_params),
        metric_count,
        summary_count,
        artifact_count,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    tracking_uri = require(args.tracking_uri, "MLFLOW_TRACKING_URI or --tracking-uri")
    experiment_name = require(args.experiment, "MLFLOW_EXPERIMENT_NAME or --experiment")

    mlflow.set_tracking_uri(tracking_uri)
    configure_remote_mlflow(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    try:
        experiment_id = verify_mlflow_connection(client, experiment_name)
    except Exception:
        LOGGER.exception(
            "Cannot reach MLflow at %s — check URI, VPN, and credentials.",
            tracking_uri,
        )
        sys.exit(1)

    if args.verify:
        LOGGER.info(
            "MLflow OK: uri=%s experiment=%s id=%s",
            tracking_uri,
            experiment_name,
            experiment_id,
        )
        return

    project = require(args.project, "WANDB_PROJECT or --project")

    api = wandb.Api()
    entity = args.entity or api.default_entity
    path = wandb_run_path(entity, project)

    filters: dict[str, Any] = {}
    if args.state and args.state.lower() != "any":
        filters["state"] = args.state

    runs = api.runs(path, filters=filters, order="-created_at")
    if args.limit is not None:
        runs = runs[: args.limit]

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        LOGGER.error("Experiment %r not found after verify step", experiment_name)
        sys.exit(1)

    migrated_ids: set[str] = set()
    if args.skip_existing and not args.dry_run:
        migrated_ids = existing_wandb_ids(client, experiment.experiment_id)

    LOGGER.info(
        "Source: %s | Target: %s (%s) | Runs: %d | skip_existing=%s",
        path,
        experiment_name,
        tracking_uri,
        len(runs),
        args.skip_existing,
    )

    migrated = 0
    skipped = 0
    failed = 0

    for wandb_run in runs:
        if args.skip_existing and wandb_run.id in migrated_ids:
            LOGGER.info("Skipping %s (%s): already in MLflow", wandb_run.name, wandb_run.id)
            skipped += 1
            continue

        if args.dry_run:
            LOGGER.info("[dry-run] Would migrate %s (%s) state=%s", wandb_run.name, wandb_run.id, wandb_run.state)
            migrated += 1
            continue

        try:
            migrate_run(
                client,
                wandb_run,
                migrate_artifacts_flag=args.artifacts,
            )
            migrated += 1
        except Exception:
            LOGGER.exception("Failed to migrate %s (%s)", wandb_run.name, wandb_run.id)
            failed += 1

    LOGGER.info("Done. migrated=%d skipped=%d failed=%d", migrated, skipped, failed)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
