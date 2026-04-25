import pytest

from src.data.components.protocol_parser import (
    load_space_protocol,
    parse_legacy_csv_line,
    parse_protocol_line,
)


def test_parse_protocol_line_space_format() -> None:
    row = parse_protocol_line("audio/a.wav train bonafide")
    assert row.file_path == "audio/a.wav"
    assert row.subset == "train"
    assert row.label == "bonafide"


def test_parse_protocol_line_invalid_format() -> None:
    with pytest.raises(ValueError):
        parse_protocol_line("audio/a.wav,bonafide")


def test_parse_legacy_csv_line() -> None:
    row = parse_legacy_csv_line("audio/a.wav,spoof")
    assert row.file_path == "audio/a.wav"
    assert row.subset == "unknown"
    assert row.label == "spoof"


def test_load_space_protocol(tmp_path) -> None:
    protocol_file = tmp_path / "protocol.txt"
    protocol_file.write_text(
        "# comment\naudio/a.wav train bonafide\naudio/b.wav dev spoof\n",
        encoding="utf-8",
    )
    rows = load_space_protocol(protocol_file)
    assert len(rows) == 2
    assert rows[0].subset == "train"
    assert rows[1].label == "spoof"

