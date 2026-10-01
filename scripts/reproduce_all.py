"""
Run the complete reproducibility pipeline for the
Mandarin-English cross-linguistic prosody study.

Pipeline:
1. (Optional) Manifest preparation
2. Acoustic feature extraction
3. Statistical analysis
4. PCA
5. Linear mixed-effects modeling
6. Figure generation

Important
---------
The silence threshold, minimum pause duration, and multiple-comparison
correction method must match the settings actually used in the study.
They are therefore required as explicit command-line arguments.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str]) -> None:
    """
    Execute one pipeline command and stop immediately if it fails.
    """
    print("\n" + "=" * 72)
    print("Running:")
    print(" ".join(command))
    print("=" * 72)

    subprocess.run(
        command,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the full Mandarin-English "
            "prosody analysis pipeline."
        )
    )

    parser.add_argument(
        "--manifest",
        default="data/sample_manifest.csv",
        help="Path to sample manifest.",
    )

    parser.add_argument(
        "--manifest-note",
        default="",
        help=(
            "Optional note identifying the manifest-generation run. "
            "This is recorded in the console output for provenance."
        ),
    )

    parser.add_argument(
        "--pitch-config",
        default="data/speaker_pitch_config.csv",
        help="Path to speaker-specific pitch configuration.",
    )

    parser.add_argument(
        "--silence-threshold-db",
        required=True,
        type=float,
        help=(
            "Relative RMS threshold in dB used for "
            "pause detection. Must match the study."
        ),
    )

    parser.add_argument(
        "--minimum-pause-ms",
        required=True,
        type=float,
        help=(
            "Minimum silence duration in milliseconds "
            "counted as a pause. Must match the study."
        ),
    )

    parser.add_argument(
        "--correction",
        required=True,
        choices=[
            "bonferroni",
            "sidak",
            "holm-sidak",
            "holm",
            "simes-hochberg",
            "hommel",
            "fdr_bh",
            "fdr_by",
            "fdr_tsbh",
            "fdr_tsbky",
        ],
        help=(
            "Multiple-comparison correction method "
            "used in the study."
        ),
    )

    parser.add_argument(
        "--reference-language",
        default="EN",
        choices=["EN", "ZH"],
        help=(
            "Reference language for the mixed-effects model. "
            "Default: EN."
        ),
    )

    parser.add_argument(
        "--target-sr",
        default=16000,
        type=int,
        help="Target sampling rate. Default: 16000 Hz.",
    )

    parser.add_argument(
        "--results-dir",
        default="results/tables",
        help="Directory for numerical outputs.",
    )

    parser.add_argument(
        "--figure-dir",
        default="figures",
        help="Directory for manuscript figures.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.manifest_note:
        print(f"Manifest provenance: {args.manifest_note}")

    python = sys.executable

    results_dir = Path(args.results_dir)
    figure_dir = Path(args.figure_dir)

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    acoustic_features = (
        results_dir / "acoustic_features.csv"
    )

    # ------------------------------------------------------------
    # 1. Acoustic feature extraction
    # ------------------------------------------------------------
    run_command(
        [
            python,
            "src/extract_features.py",
            "--manifest",
            args.manifest,
            "--pitch-config",
            args.pitch_config,
            "--output",
            str(acoustic_features),
            "--silence-threshold-db",
            str(args.silence_threshold_db),
            "--minimum-pause-ms",
            str(args.minimum_pause_ms),
            "--target-sr",
            str(args.target_sr),
        ]
    )

    # ------------------------------------------------------------
    # 2. Statistical analysis
    # ------------------------------------------------------------
    run_command(
        [
            python,
            "src/statistical_analysis.py",
            "--input",
            str(acoustic_features),
            "--output-dir",
            str(results_dir),
            "--correction",
            args.correction,
        ]
    )

    # ------------------------------------------------------------
    # 3. PCA
    # ------------------------------------------------------------
    standardized_features = (
        results_dir / "standardized_features.csv"
    )

    run_command(
        [
            python,
            "src/pca_analysis.py",
            "--input",
            str(standardized_features),
            "--output-dir",
            str(results_dir),
            "--components",
            "2",
        ]
    )

    # ------------------------------------------------------------
    # 4. Linear mixed-effects model
    # ------------------------------------------------------------
    run_command(
        [
            python,
            "src/mixed_effects_model.py",
            "--input",
            str(acoustic_features),
            "--output-dir",
            str(results_dir),
            "--reference-language",
            args.reference_language,
        ]
    )

    # ------------------------------------------------------------
    # 5. Generate Figures 5-9
    # ------------------------------------------------------------
    run_command(
        [
            python,
            "src/generate_figures.py",
            "--table-dir",
            str(results_dir),
            "--figure-dir",
            str(figure_dir),
        ]
    )

    print("\n" + "=" * 72)
    print("Full reproducibility pipeline completed successfully.")
    print("=" * 72)

    print(f"\nNumerical outputs: {results_dir}")
    print(f"Figures:           {figure_dir}")


if __name__ == "__main__":
    main()
