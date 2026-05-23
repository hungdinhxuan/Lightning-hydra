import csv
import wave
from pathlib import Path

from scripts.dataset_metadata.create_april2026_metadata import (
    audio_info,
    infer_attack_family,
    parse_protocol_line,
    write_dataset_metadata,
)


def _write_wav(path: Path, seconds: float = 0.25, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\0\0" * frames)


def test_parse_protocol_line_handles_quoted_paths_with_spaces():
    row = parse_protocol_line('"fake/de/Chatterbox Multilingual/a.wav" eval spoof')

    assert row is not None
    assert row.rel_path == "fake/de/Chatterbox Multilingual/a.wav"
    assert row.split == "eval"
    assert row.attack_type == "spoof"


def test_audio_info_reads_wav_header(tmp_path):
    wav_path = tmp_path / "x.wav"
    _write_wav(wav_path, seconds=0.5)

    info = audio_info(wav_path)

    assert info.codec == "pcm_s16le"
    assert info.channel == "1"
    assert info.duration == "0.5"


def test_infer_attack_family_uses_path_hints():
    assert infer_attack_family("fake/en/RVC/example.wav") == "vc"
    assert infer_attack_family("fake/en/Some-TTS/example.wav") == "tts"
    assert infer_attack_family("fake/en/Chatterbox/example.wav") == "unk"


def test_write_dataset_metadata_infers_mlaad_fields(tmp_path):
    dataset = tmp_path / "MLAAD_v8"
    _write_wav(dataset / "fake/de/Chatterbox Multilingual/a.wav", seconds=0.25)
    (dataset / "protocol.txt").write_text(
        '"fake/de/Chatterbox Multilingual/a.wav" eval spoof\n',
        encoding="utf-8",
    )

    output_path, count = write_dataset_metadata(
        dataset_dir=dataset,
        output_name="metadata.csv",
        workers=1,
        skip_duration=False,
        overwrite=False,
    )

    assert count == 1
    with output_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    assert rows == [
        {
            "path": "fake/de/Chatterbox Multilingual/a.wav",
            "split": "eval",
            "attack_type": "Chatterbox Multilingual",
            "attack_family": "unk",
            "codec": "pcm_s16le",
            "channel": "1",
            "language": "de",
            "speaker_id": "",
            "duration": "0.25",
            "source": "MLAAD",
            "label": "spoof",
        }
    ]


def test_write_dataset_metadata_infers_m_ailabs_fields(tmp_path):
    dataset = tmp_path / "M-AILABS"
    rel_path = "de_DE/by_book/female/angela_merkel/book/wavs/a.wav"
    _write_wav(dataset / rel_path)
    (dataset / "protocol.txt").write_text(f"{rel_path} eval bonafide\n", encoding="utf-8")

    output_path, count = write_dataset_metadata(
        dataset_dir=dataset,
        output_name="metadata.csv",
        workers=1,
        skip_duration=False,
        overwrite=False,
    )

    assert count == 1
    with output_path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle, delimiter=";"))
    assert row["attack_type"] == "bonafide"
    assert row["attack_family"] == "unk"
    assert row["language"] == "de_DE"
    assert row["speaker_id"] == "angela_merkel"
    assert row["source"] == "M-AILABS"
    assert row["label"] == "bonafide"
