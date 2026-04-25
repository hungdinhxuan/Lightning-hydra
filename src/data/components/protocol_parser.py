from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class ProtocolRow:
    file_path: str
    subset: str
    label: str


def parse_protocol_line(line: str) -> ProtocolRow:
    parts = line.strip().split()
    if len(parts) != 3:
        raise ValueError(
            "Protocol line must be '<file_path> <subset> <label>' separated by spaces"
        )
    return ProtocolRow(file_path=parts[0], subset=parts[1], label=parts[2])


def load_space_protocol(path: str | Path) -> List[ProtocolRow]:
    rows: List[ProtocolRow] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(parse_protocol_line(line))
    return rows


def parse_legacy_csv_line(line: str) -> ProtocolRow:
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) != 2:
        raise ValueError("Legacy CSV line must be 'file_name,label'")
    label = parts[1]
    subset = "unknown"
    return ProtocolRow(file_path=parts[0], subset=subset, label=label)

