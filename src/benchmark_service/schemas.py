"""Schemas for persistent benchmark service jobs and runtime state."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stringify_path(value: Union[str, Path]) -> str:
    return str(Path(value))


def _normalize_extra_overrides(value: Optional[Union[Dict[str, Any], List[str]]]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [f"++{key}={val}" for key, val in sorted(value.items())]
    return list(value)


@dataclass(frozen=True)
class RuntimeConfig:
    """Model/runtime identity for one resident benchmark worker."""

    gpu_id: str
    config_path: str
    model_path: str
    adapter_path: Optional[str] = None
    is_ln: bool = True
    precision: Optional[str] = None
    extra_overrides: List[str] = field(default_factory=list)

    @classmethod
    def from_job(cls, job: "BenchmarkJob") -> "RuntimeConfig":
        extra_overrides = list(job.extra_overrides)
        if job.precision:
            precision_override = f"++trainer.precision={job.precision}"
            if not any(
                item.startswith("++trainer.precision=") or item.startswith("+trainer.precision=")
                for item in extra_overrides
            ):
                extra_overrides.append(precision_override)
        return cls(
            gpu_id=str(job.gpu_id),
            config_path=job.config_path,
            model_path=job.model_path,
            adapter_path=job.adapter_path,
            is_ln=job.is_ln,
            precision=job.precision,
            extra_overrides=extra_overrides,
        )

    def signature(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkJob:
    """Dataset benchmark request compatible with existing benchmark.py arguments."""

    dataset_path: str
    result_dir: str
    run_name: str
    config_path: str
    model_path: str
    gpu_id: str = "0"
    adapter_path: Optional[str] = None
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    batch_size: int = 128
    precision: Optional[str] = None
    is_ln: bool = True
    random_start: bool = True
    trim_length: int = 64000
    eval_config: Optional[str] = None
    missing_protocol_label: str = "skip"
    extra_overrides: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        self.dataset_path = _stringify_path(self.dataset_path)
        self.result_dir = _stringify_path(self.result_dir)
        if self.eval_config is not None:
            self.eval_config = _stringify_path(self.eval_config)
        self.gpu_id = str(self.gpu_id)
        self.extra_overrides = _normalize_extra_overrides(self.extra_overrides)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BenchmarkJob":
        payload = dict(payload)
        if "adapter_paths" in payload and "adapter_path" not in payload:
            payload["adapter_path"] = payload.pop("adapter_paths")
        if "benchmark_folder" in payload and "dataset_path" not in payload:
            payload["dataset_path"] = payload.pop("benchmark_folder")
        if "results_folder" in payload and "result_dir" not in payload:
            payload["result_dir"] = payload.pop("results_folder")
        if "comment" in payload and "run_name" not in payload:
            payload["run_name"] = payload.pop("comment")
        if "config" in payload and "config_path" not in payload:
            payload["config_path"] = payload.pop("config")
        return cls(**payload)

    @classmethod
    def from_json_file(cls, path: Union[str, Path]) -> "BenchmarkJob":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def write_json(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")

    def validate(self) -> None:
        dataset_path = Path(self.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"dataset_path does not exist: {dataset_path}")
        if not dataset_path.is_dir():
            raise NotADirectoryError(f"dataset_path is not a directory: {dataset_path}")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.missing_protocol_label not in {"ask", "skip", "auto", "spoof", "bonafide"}:
            raise ValueError("missing_protocol_label must be ask|skip|auto|spoof|bonafide")
        invalid_overrides = [
            override
            for override in self.extra_overrides
            if not (override.startswith("+") or override.startswith("++"))
        ]
        if invalid_overrides:
            raise ValueError(
                "extra_overrides must be Hydra overrides starting with + or ++: "
                + ", ".join(invalid_overrides)
            )

    def command_equivalent(self) -> List[str]:
        cmd = [
            "python",
            "scripts/benchmark_py/benchmark.py",
            "-g",
            self.gpu_id,
            "-c",
            self.config_path,
            "-b",
            self.dataset_path,
            "-m",
            self.model_path,
            "-r",
            self.result_dir,
            "-n",
            self.run_name,
            "-l",
            str(self.is_ln).lower(),
            "-s",
            str(self.random_start).lower(),
            "-t",
            str(self.trim_length),
            "-z",
            str(self.batch_size),
            "--missing-protocol-label",
            self.missing_protocol_label,
        ]
        if self.adapter_path:
            cmd.extend(["-a", self.adapter_path])
        if self.eval_config:
            cmd.extend(["--eval-config", self.eval_config])
        cmd.extend(self.extra_overrides)
        return cmd
