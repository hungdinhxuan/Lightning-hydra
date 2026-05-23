#!/usr/bin/env python3
"""Upload April_2026_benchmark scores + per-dataset metadata to CM Performance Dashboard API.

Auth: POST ``/v1/auth/login`` when credentials present. Defaults read from repo-root
``.env``: ``CM_PERFORMANCE_DASHBOARD_USERNAME``, ``CM_PERFORMANCE_DASHBOARD_PASSWORD``
(override with ``--username`` / ``--password``). Bearer token attached to all requests.

Two modes:

* ``--mode single`` (default): use a single existing session id; PUT the merged
  metadata of every dataset and POST every score file there.
* ``--mode per-dataset``: create one fresh session per dataset, then PUT only
  that dataset's metadata and POST only its score file. Useful when the server
  rejects a large merged-metadata PUT.

Skip rules
----------
* Score files for datasets listed in ``--skip-dataset`` are not uploaded.
* Pooled / merged aggregate files (``pooled_merged*``, ``merged_protocol*``,
  ``merged_scores*``) are never uploaded.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, TypeVar

import requests

T = TypeVar("T")


def with_retries(
    label: str,
    func: Callable[[], requests.Response],
    *,
    attempts: int = 5,
    backoff: float = 2.0,
    accept_status: tuple[int, ...] = (200, 201, 204),
) -> requests.Response:
    """Retry ``func`` with exponential backoff for transient 5xx / network errors."""
    last_err: str = ""
    for i in range(1, attempts + 1):
        try:
            r = func()
            if r.status_code in accept_status:
                return r
            if 500 <= r.status_code < 600 and i < attempts:
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                print(f"  retry {i}/{attempts} {label}: {last_err}", flush=True)
                time.sleep(backoff ** i)
                continue
            return r
        except requests.RequestException as e:
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
            if i < attempts:
                print(f"  retry {i}/{attempts} {label}: {last_err}", flush=True)
                time.sleep(backoff ** i)
                continue
            raise
    raise RuntimeError(f"{label}: exhausted retries; last={last_err}")

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_USER_KEY = "CM_PERFORMANCE_DASHBOARD_USERNAME"
ENV_PASS_KEY = "CM_PERFORMANCE_DASHBOARD_PASSWORD"

DEFAULT_API = "http://192.168.0.17:8000"
DEFAULT_DATA_DIR = "/nvme2/hungdx/Lightning-hydra/data/April_2026_benchmark"
DEFAULT_SCORES_DIR = (
    "/nvme2/hungdx/Lightning-hydra/logs/results/April_2026_benchmark/"
    "12May2026_lora_from_29April26_xlsr_conformertcm_mdt"
)
DEFAULT_MODEL_ID = "12May26_xlsr_conformertcm_mdt"
DATASET_SPLIT_TOKEN = "_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_lora_infer_"
IGNORED_PREFIXES = ("pooled_merged", "merged_protocol", "merged_scores")


def _load_repo_dotenv() -> None:
    """Populate os.environ from repo-root .env if present (no extra deps)."""
    path = REPO_ROOT / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        os.environ[key] = val


def dashboard_session(
    api: str, username: str | None, password: str | None
) -> requests.Session:
    """Bearer token from POST /v1/auth/login when credentials given; else bare Session."""
    sess = requests.Session()
    base = api.rstrip("/")
    if not username and not password:
        return sess
    if not (username and password):
        sys.exit(
            f"Need both username and password (CLI or env {ENV_USER_KEY} + {ENV_PASS_KEY})."
        )
    url = f"{base}/v1/auth/login"
    r = with_retries(
        "login",
        lambda: sess.post(
            url,
            json={"username": username, "password": password},
            timeout=60,
        ),
        accept_status=(200,),
    )
    if r.status_code != 200:
        sys.exit(f"Login failed: {r.status_code} {r.text[:500]}")
    body = r.json()
    token = body.get("token")
    if not token:
        sys.exit("Login response missing token")
    sess.headers["Authorization"] = f"Bearer {token}"
    # Login may Set-Cookie; stale cookie + Bearer confuses some stacks on later POSTs.
    sess.cookies.clear()
    print(f"Logged in as {body.get('username', username)!r}", flush=True)
    return sess


def refresh_dashboard_auth(
    sess: requests.Session,
    api: str,
    username: str | None,
    password: str | None,
) -> None:
    """Re-POST /v1/auth/login on existing Session (server drops Bearer after large transfers)."""
    if not username or not password:
        return
    base = api.rstrip("/")
    url = f"{base}/v1/auth/login"
    r = with_retries(
        "login_refresh",
        lambda: sess.post(
            url,
            json={"username": username, "password": password},
            timeout=60,
        ),
        accept_status=(200,),
    )
    if r.status_code != 200:
        sys.exit(f"Re-login failed: {r.status_code} {r.text[:500]}")
    body = r.json()
    token = body.get("token")
    if not token:
        sys.exit("Re-login response missing token")
    sess.headers["Authorization"] = f"Bearer {token}"
    sess.cookies.clear()


def merge_metadata(data_dir: Path) -> tuple[str, int, list[str]]:
    """Concatenate every ``metadata.csv`` under ``data_dir`` into one CSV string."""
    csv_paths = sorted(data_dir.glob("*/metadata.csv"))
    if not csv_paths:
        raise SystemExit(f"No metadata.csv files found under {data_dir}")

    header: list[str] | None = None
    body_rows: list[str] = []
    datasets: list[str] = []
    total = 0
    for csv_path in csv_paths:
        dataset = csv_path.parent.name
        datasets.append(dataset)
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            first = f.readline().rstrip("\n")
            if header is None:
                header = first
            elif first != header:
                raise SystemExit(
                    f"Header mismatch in {csv_path}:\n  expected: {header}\n  got: {first}"
                )
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                body_rows.append(line)
                total += 1
    assert header is not None
    merged = "\n".join([header, *body_rows]) + "\n"
    return merged, total, datasets


def iter_score_files(scores_dir: Path, skip_datasets: set[str]) -> Iterable[tuple[Path, str]]:
    for path in sorted(scores_dir.glob("*.txt")):
        name = path.name
        if any(name.startswith(p) for p in IGNORED_PREFIXES):
            continue
        if DATASET_SPLIT_TOKEN not in name:
            continue
        dataset_id = name.split(DATASET_SPLIT_TOKEN, 1)[0]
        if dataset_id in skip_datasets:
            continue
        yield path, dataset_id


def read_dataset_metadata(data_dir: Path, dataset_id: str) -> str:
    csv_path = data_dir / dataset_id / "metadata.csv"
    if not csv_path.exists():
        raise SystemExit(f"Metadata not found: {csv_path}")
    return csv_path.read_text(encoding="utf-8")


def create_session(
    api: str, sess: requests.Session, name: str, description: str | None = None
) -> str:
    url = f"{api}/v1/sessions"
    payload: dict = {"name": name[:120]}
    if description is not None:
        payload["description"] = description
    r = with_retries(
        f"create_session({name!r})",
        lambda: sess.post(url, json=payload, timeout=60),
    )
    if r.status_code >= 300:
        sys.exit(f"  FAILED to create session {name!r}: {r.status_code} {r.text[:500]}")
    return r.json()["id"]


def put_metadata(
    api: str, sess: requests.Session, session_id: str, csv_text: str, file_name: str
) -> None:
    url = f"{api}/v1/sessions/{session_id}/metadata"
    payload = {"file_name": file_name, "text": csv_text}
    print(f"[metadata] PUT session={session_id} ({len(csv_text):,} bytes)...", flush=True)
    r = with_retries(
        f"put_metadata(session={session_id})",
        lambda: sess.put(url, json=payload, timeout=600),
    )
    if r.status_code >= 300:
        sys.exit(f"  FAILED {r.status_code}: {r.text[:1000]}")
    body = r.json()
    print(
        f"  OK rows_loaded={body.get('rows_loaded')} "
        f"protocol_rows_loaded={body.get('protocol_rows_loaded')} "
        f"errors={len(body.get('errors', []))}"
    )
    for err in body.get("errors", [])[:5]:
        print("    error:", err)


def post_score_file(
    api: str,
    sess: requests.Session,
    session_id: str,
    score_path: Path,
    model_id: str,
    dataset_id: str,
    on_conflict: str,
) -> None:
    url = f"{api}/v1/sessions/{session_id}/score-files/upload"
    params = {"on_conflict": on_conflict}
    data = {"model_id": model_id, "dataset_id": dataset_id}
    print(
        f"[score] POST dataset={dataset_id:<35} file={score_path.name}",
        flush=True,
    )

    def _send() -> requests.Response:
        with score_path.open("rb") as fh:
            files = {"file": (score_path.name, fh, "text/plain")}
            return sess.post(url, params=params, data=data, files=files, timeout=600)

    r = with_retries(f"post_score({dataset_id})", _send)
    if r.status_code >= 300:
        print(f"  FAILED {r.status_code}: {r.text[:500]}")
        return
    body = r.json()
    sf = body.get("score_file", {})
    print(
        f"  OK id={sf.get('id')} parsed_rows={sf.get('parsed_rows')} "
        f"error_count={sf.get('error_count')} replaced_id={body.get('replaced_score_file_id')}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--api", default=DEFAULT_API)
    p.add_argument(
        "--username",
        default=os.environ.get(ENV_USER_KEY),
        help=f"Dashboard login (default: env {ENV_USER_KEY} from repo .env).",
    )
    p.add_argument(
        "--password",
        default=os.environ.get(ENV_PASS_KEY),
        help=f"Dashboard password (default: env {ENV_PASS_KEY} from repo .env).",
    )
    p.add_argument(
        "--mode",
        choices=("single", "per-dataset"),
        default="single",
        help="single: one shared session (requires --session-id). "
        "per-dataset: create one fresh session per dataset.",
    )
    p.add_argument(
        "--session-id",
        default=None,
        help="Existing session id (required for --mode single).",
    )
    p.add_argument(
        "--session-name-prefix",
        default="",
        help="Optional prefix prepended to per-dataset session names "
        "(e.g. '29April26-'). Default: empty (session name = dataset_id).",
    )
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR, type=Path)
    p.add_argument("--scores-dir", default=DEFAULT_SCORES_DIR, type=Path)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument(
        "--skip-dataset",
        action="append",
        default=["MLAAD_v6"],
        help="Score files for these datasets are not uploaded (repeatable). "
        "Default: MLAAD_v6 (already uploaded).",
    )
    p.add_argument(
        "--on-conflict",
        choices=("error", "replace"),
        default="replace",
        help="What to do if (model_id, dataset_id) score file already exists.",
    )
    p.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Do not PUT metadata (assume already uploaded).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded, but make no API calls.",
    )
    p.add_argument(
        "--mapping-out",
        type=Path,
        default=None,
        help="(per-dataset mode) Write a JSON file mapping dataset_id -> session_id.",
    )
    return p.parse_args()


def run_single(
    args: argparse.Namespace, sess: requests.Session, score_jobs: list[tuple[Path, str]]
) -> None:
    if not args.session_id:
        sys.exit("--session-id is required when --mode single")

    if not args.skip_metadata:
        refresh_dashboard_auth(sess, args.api, args.username, args.password)
        merged_csv, total, datasets = merge_metadata(args.data_dir)
        print(
            f"Merged metadata: datasets={len(datasets)} rows={total:,} "
            f"bytes={len(merged_csv):,}"
        )
        print(f"  datasets: {datasets}")
        if args.dry_run:
            print("\n[dry-run] No requests sent.")
            return
        put_metadata(
            args.api, sess, args.session_id, merged_csv, "April_2026_benchmark_metadata.csv"
        )
    elif args.dry_run:
        print("\n[dry-run] No requests sent.")
        return

    for path, dataset_id in score_jobs:
        refresh_dashboard_auth(sess, args.api, args.username, args.password)
        post_score_file(
            args.api,
            sess,
            args.session_id,
            path,
            args.model_id,
            dataset_id,
            args.on_conflict,
        )


def run_per_dataset(
    args: argparse.Namespace, sess: requests.Session, score_jobs: list[tuple[Path, str]]
) -> None:
    mapping: dict[str, str] = {}
    failures: list[str] = []
    for path, dataset_id in score_jobs:
        refresh_dashboard_auth(sess, args.api, args.username, args.password)
        session_name = f"{args.session_name_prefix}{dataset_id}"
        description = f"{args.model_id} on {dataset_id}"
        print(f"\n=== dataset={dataset_id} ===")
        if args.dry_run:
            print(f"  [dry-run] would create session name={session_name!r}")
            print(f"  [dry-run] would PUT metadata from {args.data_dir / dataset_id / 'metadata.csv'}")
            print(f"  [dry-run] would POST score file {path.name}")
            continue
        try:
            sid = create_session(args.api, sess, session_name, description=description)
            mapping[dataset_id] = sid
            print(f"  created session {sid} (name={session_name!r})")
            if not args.skip_metadata:
                csv_text = read_dataset_metadata(args.data_dir, dataset_id)
                put_metadata(args.api, sess, sid, csv_text, f"{dataset_id}_metadata.csv")
                refresh_dashboard_auth(sess, args.api, args.username, args.password)
            post_score_file(
                args.api,
                sess,
                sid,
                path,
                args.model_id,
                dataset_id,
                args.on_conflict,
            )
        except SystemExit as e:
            print(f"  SKIPPED {dataset_id}: {e}")
            failures.append(dataset_id)
        except Exception as e:
            print(f"  SKIPPED {dataset_id}: {type(e).__name__}: {e}")
            failures.append(dataset_id)
        # Incremental save so progress is preserved on crash.
        if args.mapping_out and mapping:
            args.mapping_out.write_text(json.dumps(mapping, indent=2) + "\n")

    if mapping:
        print("\nSession map (dataset_id -> session_id):")
        for ds, sid in mapping.items():
            print(f"  {ds:<35} {sid}")
        if args.mapping_out:
            print(f"Wrote {args.mapping_out}")
    if failures:
        print(f"\nFailures ({len(failures)}): {failures}")


def main() -> None:
    _load_repo_dotenv()
    args = parse_args()
    sess = dashboard_session(args.api, args.username, args.password)

    skip_datasets = set(args.skip_dataset)

    score_jobs = list(iter_score_files(args.scores_dir, skip_datasets))
    print(f"Mode: {args.mode}")
    print(f"Planned score uploads ({len(score_jobs)} files):")
    for path, ds in score_jobs:
        print(f"  - {ds:<35} <- {path.name}")
    print(f"Skip datasets: {sorted(skip_datasets)}")

    if args.mode == "single":
        run_single(args, sess, score_jobs)
    else:
        run_per_dataset(args, sess, score_jobs)

    print("\nDone.")


if __name__ == "__main__":
    main()
