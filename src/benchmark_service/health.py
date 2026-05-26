"""Health snapshot helpers for persistent benchmark worker."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.benchmark_service.schemas import RuntimeConfig, utc_now_iso


def build_health_payload(
    worker_started_at: str,
    runtime_config: Optional[RuntimeConfig],
    status: str,
    current_job_id: Optional[str] = None,
    load_count: int = 0,
) -> Dict[str, Any]:
    return {
        "status": status,
        "worker_pid": os.getpid(),
        "worker_started_at": worker_started_at,
        "checked_at": utc_now_iso(),
        "current_job_id": current_job_id,
        "load_count": load_count,
        "runtime_signature": runtime_config.signature() if runtime_config else None,
    }


def write_health(path: Union[str, Path], payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
