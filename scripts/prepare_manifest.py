"""Create a reproducible balanced manifest and pitch configuration.

The script indexes local AISHELL-1 WAV files and LibriSpeech FLAC files,
selects a deterministic balanced subset, and writes the two CSV files
required by ``src/extract_features.py``. Raw recordings are never copied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path


def stable_key(path: Path) -> str:
    """Return a stable sort key independent of filesystem order."""
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()


def english_speaker(path: Path) -> str:
    match = re.match(r"(\d+)-", path.stem)
    if not match:
        raise ValueError(f"Cannot parse LibriSpeech speaker ID: {path.name}")
    return f"EN_{match.group(1)}"


def mandarin_speaker(path: Path) -> str:
    match = re.search(r"(S\d{4})W", path.stem)
    if not match:
        raise ValueError(f"Cannot parse AISHELL speaker ID: {path.name}")
    return f"ZH_{match.group(1)}"


def build_rows(zh_root: Path, en_root: Path, n_per_language: int) -> list[dict[str, str]]:
    zh_files = sorted(zh_root.rglob("*.wav"), key=stable_key)
    en_files = sorted(en_root.rglob("*.flac"), key=stable_key)
    if len(zh_files) < n_per_language or len(en_files) < n_per_language:
        raise ValueError(
            f"Not enough files: AISHELL={len(zh_files)}, LibriSpeech={len(en_files)}, "
            f"requested={n_per_language} per language"
        )

    rows: list[dict[str, str]] = []
    for language, corpus, files, speaker_fn, partition in [
        ("ZH", "AISHELL-1", zh_files[:n_per_language], mandarin_speaker, "test"),
        ("EN", "LibriSpeech", en_files[:n_per_language], english_speaker, "dev-clean"),
    ]:
        for index, path in enumerate(files, start=1):
            rows.append(
                {
                    "language": language,
                    "corpus": corpus,
                    "speaker_id": speaker_fn(path),
                    "utterance_id": path.stem,
                    "audio_file": str(path.resolve()),
                    "partition": partition,
                    "selection_index": str(index),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_pitch_config(path: Path, rows: list[dict[str, str]], floor: float, ceiling: float) -> None:
    speakers = sorted({row["speaker_id"] for row in rows})
    write_csv(
        path,
        [
            {"speaker_id": speaker, "pitch_floor_hz": str(floor), "pitch_ceiling_hz": str(ceiling)}
            for speaker in speakers
        ],
        ["speaker_id", "pitch_floor_hz", "pitch_ceiling_hz"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mandarin-root", required=True, type=Path)
    parser.add_argument("--english-root", required=True, type=Path)
    parser.add_argument("--manifest", default="data/sample_manifest.csv", type=Path)
    parser.add_argument("--pitch-config", default="data/speaker_pitch_config.csv", type=Path)
    parser.add_argument("--n-per-language", default=500, type=int)
    parser.add_argument("--pitch-floor-hz", default=60.0, type=float)
    parser.add_argument("--pitch-ceiling-hz", default=500.0, type=float)
    args = parser.parse_args()

    rows = build_rows(args.mandarin_root, args.english_root, args.n_per_language)
    write_csv(args.manifest, rows, [
        "language", "corpus", "speaker_id", "utterance_id", "audio_file", "partition", "selection_index"
    ])
    write_pitch_config(args.pitch_config, rows, args.pitch_floor_hz, args.pitch_ceiling_hz)
    print(f"Wrote {len(rows)} manifest rows to {args.manifest}")
    print(f"Wrote {len({row['speaker_id'] for row in rows})} speaker pitch settings to {args.pitch_config}")


if __name__ == "__main__":
    main()
