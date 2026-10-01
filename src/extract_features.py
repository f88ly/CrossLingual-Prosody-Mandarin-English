"""
Acoustic feature extraction for cross-linguistic prosody analysis.

This script extracts seven utterance-level acoustic measures used in the study:

1. F0 mean
2. F0 standard deviation
3. F0 range
4. RMS energy
5. Duration
6. Pause ratio
7. Voicing ratio

The script is designed for balanced Mandarin (AISHELL-1) and English
(LibriSpeech) read-speech samples.

IMPORTANT
---------
Pitch limits and silence-detection parameters should reflect the actual
experimental settings used in the study. They are intentionally supplied
through configuration/command-line arguments rather than silently fixed
inside the analysis code.

Original corpus audio is not redistributed with this repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import librosa
import numpy as np
import pandas as pd
import parselmouth
import soundfile as sf


FEATURE_COLUMNS = [
    "f0_mean_hz",
    "f0_std_hz",
    "f0_range_hz",
    "rms_energy",
    "duration_s",
    "pause_ratio",
    "voicing_ratio",
]


def load_audio(
    audio_path: str | Path,
    target_sr: int = 16000,
) -> tuple[np.ndarray, int]:
    """
    Load an audio file as mono and resample it to the target sampling rate.

    Parameters
    ----------
    audio_path : str or Path
        Path to an audio file.
    target_sr : int
        Target sampling rate in Hz.

    Returns
    -------
    y : np.ndarray
        Mono waveform.
    sr : int
        Sampling rate.
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    y, sr = sf.read(audio_path, always_2d=False)

    # Convert stereo/multichannel recordings to mono.
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    y = np.asarray(y, dtype=np.float64)

    if sr != target_sr:
        y = librosa.resample(
            y,
            orig_sr=sr,
            target_sr=target_sr,
        )
        sr = target_sr

    # Remove NaN/Inf values if present.
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    return y, sr


def compute_duration(
    y: np.ndarray,
    sr: int,
) -> float:
    """Return utterance duration in seconds."""
    if sr <= 0:
        raise ValueError("Sampling rate must be positive.")

    return float(len(y) / sr)


def compute_rms_energy(y: np.ndarray) -> float:
    """
    Calculate root mean square (RMS) energy over the entire utterance.
    """
    if len(y) == 0:
        return np.nan

    return float(np.sqrt(np.mean(np.square(y))))


def extract_pitch_features(
    y: np.ndarray,
    sr: int,
    pitch_floor_hz: float,
    pitch_ceiling_hz: float,
    time_step: float = 0.0,
) -> Dict[str, float]:
    """
    Extract F0 statistics using Praat's autocorrelation-based pitch analysis.

    Parameters
    ----------
    y : np.ndarray
        Audio waveform.
    sr : int
        Sampling rate.
    pitch_floor_hz : float
        Minimum F0 allowed in pitch tracking.
    pitch_ceiling_hz : float
        Maximum F0 allowed in pitch tracking.
    time_step : float
        Praat pitch-analysis time step. A value of 0 lets Praat select
        an appropriate value automatically.

    Returns
    -------
    dict
        F0 mean, standard deviation and range in Hz.
    """
    if pitch_floor_hz <= 0:
        raise ValueError("pitch_floor_hz must be greater than zero.")

    if pitch_ceiling_hz <= pitch_floor_hz:
        raise ValueError(
            "pitch_ceiling_hz must be greater than pitch_floor_hz."
        )

    sound = parselmouth.Sound(y, sampling_frequency=sr)

    pitch = sound.to_pitch_ac(
        time_step=time_step,
        pitch_floor=pitch_floor_hz,
        pitch_ceiling=pitch_ceiling_hz,
    )

    # Praat represents unvoiced frames as 0 Hz.
    f0 = pitch.selected_array["frequency"]
    voiced_f0 = f0[f0 > 0]

    if voiced_f0.size == 0:
        return {
            "f0_mean_hz": np.nan,
            "f0_std_hz": np.nan,
            "f0_range_hz": np.nan,
        }

    return {
        "f0_mean_hz": float(np.mean(voiced_f0)),
        "f0_std_hz": float(np.std(voiced_f0, ddof=1))
        if voiced_f0.size > 1
        else 0.0,
        "f0_range_hz": float(np.max(voiced_f0) - np.min(voiced_f0)),
    }


def compute_voicing_ratio(
    y: np.ndarray,
    sr: int,
    pitch_floor_hz: float,
    pitch_ceiling_hz: float,
    time_step: float = 0.0,
) -> float:
    """
    Estimate voicing ratio from the proportion of Praat pitch frames
    containing a valid F0 estimate.
    """
    sound = parselmouth.Sound(y, sampling_frequency=sr)

    pitch = sound.to_pitch_ac(
        time_step=time_step,
        pitch_floor=pitch_floor_hz,
        pitch_ceiling=pitch_ceiling_hz,
    )

    f0 = pitch.selected_array["frequency"]

    if f0.size == 0:
        return np.nan

    voiced_frames = np.sum(f0 > 0)

    return float(voiced_frames / f0.size)


def compute_pause_ratio(
    y: np.ndarray,
    sr: int,
    silence_threshold_db: float,
    frame_length: int = 2048,
    hop_length: int = 512,
    minimum_pause_ms: float = 0.0,
) -> float:
    """
    Estimate the proportion of an utterance occupied by silence/pause frames.

    Silence is detected relative to the maximum frame RMS level.

    Parameters
    ----------
    y : np.ndarray
        Audio waveform.
    sr : int
        Sampling rate.
    silence_threshold_db : float
        Relative energy threshold in dB. This value MUST be set according
        to the experimental segmentation/silence-detection procedure.
        For example, a frame is considered a silence candidate when its
        RMS level falls more than |threshold| dB below the utterance maximum.
    frame_length : int
        Analysis-frame length.
    hop_length : int
        Hop length between frames.
    minimum_pause_ms : float
        Minimum consecutive silence duration required for frames to count
        as a pause. Set this to the value used in the actual experiment.

    Returns
    -------
    float
        Pause duration divided by total analyzed frame duration.
    """
    if len(y) == 0:
        return np.nan

    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]

    if rms.size == 0:
        return np.nan

    eps = np.finfo(float).eps

    rms_db = 20.0 * np.log10(
        np.maximum(rms, eps) / np.maximum(np.max(rms), eps)
    )

    silence_candidates = rms_db <= silence_threshold_db

    if minimum_pause_ms <= 0:
        return float(np.mean(silence_candidates))

    minimum_frames = int(
        np.ceil(
            (minimum_pause_ms / 1000.0)
            * sr
            / hop_length
        )
    )

    minimum_frames = max(minimum_frames, 1)

    valid_pause_frames = np.zeros_like(
        silence_candidates,
        dtype=bool,
    )

    start: Optional[int] = None

    for i, is_silent in enumerate(silence_candidates):
        if is_silent and start is None:
            start = i

        at_last_frame = i == len(silence_candidates) - 1

        if start is not None and (
            (not is_silent) or at_last_frame
        ):
            end = i if not is_silent else i + 1

            if end - start >= minimum_frames:
                valid_pause_frames[start:end] = True

            start = None

    return float(np.mean(valid_pause_frames))


def extract_features(
    audio_path: str | Path,
    pitch_floor_hz: float,
    pitch_ceiling_hz: float,
    silence_threshold_db: float,
    minimum_pause_ms: float,
    target_sr: int = 16000,
    pitch_time_step: float = 0.0,
) -> Dict[str, float]:
    """
    Extract all seven acoustic features from a single utterance.
    """
    y, sr = load_audio(
        audio_path,
        target_sr=target_sr,
    )

    features: Dict[str, float] = {}

    features.update(
        extract_pitch_features(
            y,
            sr,
            pitch_floor_hz=pitch_floor_hz,
            pitch_ceiling_hz=pitch_ceiling_hz,
            time_step=pitch_time_step,
        )
    )

    features["rms_energy"] = compute_rms_energy(y)

    features["duration_s"] = compute_duration(
        y,
        sr,
    )

    features["pause_ratio"] = compute_pause_ratio(
        y,
        sr,
        silence_threshold_db=silence_threshold_db,
        minimum_pause_ms=minimum_pause_ms,
    )

    features["voicing_ratio"] = compute_voicing_ratio(
        y,
        sr,
        pitch_floor_hz=pitch_floor_hz,
        pitch_ceiling_hz=pitch_ceiling_hz,
        time_step=pitch_time_step,
    )

    return features


def load_speaker_pitch_config(
    config_path: str | Path,
) -> pd.DataFrame:
    """
    Load speaker-specific pitch limits.

    Required columns:
        speaker_id
        pitch_floor_hz
        pitch_ceiling_hz
    """
    config = pd.read_csv(config_path)

    required = {
        "speaker_id",
        "pitch_floor_hz",
        "pitch_ceiling_hz",
    }

    missing = required - set(config.columns)

    if missing:
        raise ValueError(
            f"Pitch configuration is missing columns: "
            f"{sorted(missing)}"
        )

    if config["speaker_id"].duplicated().any():
        raise ValueError(
            "speaker_id must be unique in the pitch configuration file."
        )

    return config


def process_manifest(
    manifest_path: str | Path,
    output_path: str | Path,
    pitch_config_path: str | Path,
    silence_threshold_db: float,
    minimum_pause_ms: float,
    target_sr: int = 16000,
    pitch_time_step: float = 0.0,
) -> pd.DataFrame:
    """
    Process all utterances listed in sample_manifest.csv.

    Required manifest columns:
        language
        corpus
        speaker_id
        utterance_id
        audio_file
    """
    manifest = pd.read_csv(manifest_path)

    required_columns = {
        "language",
        "corpus",
        "speaker_id",
        "utterance_id",
        "audio_file",
    }

    missing = required_columns - set(manifest.columns)

    if missing:
        raise ValueError(
            f"Manifest is missing required columns: "
            f"{sorted(missing)}"
        )

    pitch_config = load_speaker_pitch_config(
        pitch_config_path
    )

    pitch_lookup = pitch_config.set_index(
        "speaker_id"
    )[
        ["pitch_floor_hz", "pitch_ceiling_hz"]
    ].to_dict("index")

    rows = []

    total = len(manifest)

    for index, row in manifest.iterrows():
        speaker_id = row["speaker_id"]

        if speaker_id not in pitch_lookup:
            raise ValueError(
                f"No pitch configuration found for speaker: "
                f"{speaker_id}"
            )

        pitch_limits = pitch_lookup[speaker_id]

        print(
            f"[{index + 1}/{total}] "
            f"{row['language']} | "
            f"{row['utterance_id']}"
        )

        try:
            acoustic = extract_features(
                audio_path=row["audio_file"],
                pitch_floor_hz=float(
                    pitch_limits["pitch_floor_hz"]
                ),
                pitch_ceiling_hz=float(
                    pitch_limits["pitch_ceiling_hz"]
                ),
                silence_threshold_db=silence_threshold_db,
                minimum_pause_ms=minimum_pause_ms,
                target_sr=target_sr,
                pitch_time_step=pitch_time_step,
            )

            result = row.to_dict()
            result.update(acoustic)
            result["extraction_status"] = "success"
            result["extraction_error"] = ""

        except Exception as exc:
            result = row.to_dict()

            for feature in FEATURE_COLUMNS:
                result[feature] = np.nan

            result["extraction_status"] = "failed"
            result["extraction_error"] = str(exc)

        rows.append(result)

    output_df = pd.DataFrame(rows)

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_csv(
        output_path,
        index=False,
    )

    return output_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract acoustic-prosodic features from "
            "Mandarin and English read-speech samples."
        )
    )

    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to data/sample_manifest.csv",
    )

    parser.add_argument(
        "--pitch-config",
        required=True,
        help=(
            "CSV containing speaker_id, pitch_floor_hz, "
            "and pitch_ceiling_hz."
        ),
    )

    parser.add_argument(
        "--output",
        default="results/tables/acoustic_features.csv",
        help="Output CSV file.",
    )

    parser.add_argument(
        "--silence-threshold-db",
        required=True,
        type=float,
        help=(
            "Relative RMS threshold in dB used for "
            "silence/pause detection. Use the value from "
            "the actual experiment."
        ),
    )

    parser.add_argument(
        "--minimum-pause-ms",
        required=True,
        type=float,
        help=(
            "Minimum silence duration in milliseconds "
            "counted as a pause. Use the value from "
            "the actual experiment."
        ),
    )

    parser.add_argument(
        "--target-sr",
        type=int,
        default=16000,
        help="Target sampling rate. Default: 16000 Hz.",
    )

    parser.add_argument(
        "--pitch-time-step",
        type=float,
        default=0.0,
        help=(
            "Praat pitch-analysis time step in seconds. "
            "0 lets Praat determine the time step."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output = process_manifest(
        manifest_path=args.manifest,
        output_path=args.output,
        pitch_config_path=args.pitch_config,
        silence_threshold_db=args.silence_threshold_db,
        minimum_pause_ms=args.minimum_pause_ms,
        target_sr=args.target_sr,
        pitch_time_step=args.pitch_time_step,
    )

    n_success = (
        output["extraction_status"] == "success"
    ).sum()

    n_failed = (
        output["extraction_status"] == "failed"
    ).sum()

    print("\nFeature extraction completed.")
    print(f"Successful utterances: {n_success}")
    print(f"Failed utterances:     {n_failed}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
