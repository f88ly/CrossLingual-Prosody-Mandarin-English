"""
Principal component analysis for the Mandarin-English prosody study.

This script:
1. Loads standardized acoustic features
2. Performs PCA
3. Exports PC scores
4. Exports feature loadings
5. Exports explained-variance statistics

Expected input:
    results/tables/standardized_features.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


FEATURES = [
    "f0_mean_hz_z",
    "f0_std_hz_z",
    "f0_range_hz_z",
    "rms_energy_z",
    "duration_s_z",
    "pause_ratio_z",
    "voicing_ratio_z",
]

FEATURE_LABELS = {
    "f0_mean_hz_z": "F0 mean",
    "f0_std_hz_z": "F0 std",
    "f0_range_hz_z": "F0 range",
    "rms_energy_z": "RMS energy",
    "duration_s_z": "Duration",
    "pause_ratio_z": "Pause ratio",
    "voicing_ratio_z": "Voicing ratio",
}


def load_standardized_data(
    input_path: str | Path,
) -> pd.DataFrame:
    """
    Load standardized acoustic features.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    required_columns = {
        "language",
        *FEATURES,
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    return df


def prepare_feature_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retain rows with complete standardized features.

    Returns
    -------
    metadata : pd.DataFrame
        Non-feature columns retained for sample identification.

    matrix : pd.DataFrame
        Complete standardized feature matrix.
    """
    valid_mask = df[FEATURES].notna().all(axis=1)

    clean = df.loc[valid_mask].copy()

    metadata_columns = [
        column
        for column in [
            "language",
            "corpus",
            "speaker_id",
            "utterance_id",
        ]
        if column in clean.columns
    ]

    metadata = clean[metadata_columns].copy()

    matrix = clean[FEATURES].astype(float)

    return metadata, matrix


def run_pca(
    matrix: pd.DataFrame,
    n_components: int = 2,
) -> tuple[PCA, np.ndarray]:
    """
    Fit PCA and transform the standardized feature matrix.
    """
    if n_components < 1:
        raise ValueError(
            "n_components must be at least 1."
        )

    if n_components > matrix.shape[1]:
        raise ValueError(
            "n_components cannot exceed number of features."
        )

    pca = PCA(
        n_components=n_components
    )

    scores = pca.fit_transform(matrix)

    return pca, scores


def build_score_table(
    metadata: pd.DataFrame,
    scores: np.ndarray,
) -> pd.DataFrame:
    """
    Combine sample metadata with PCA scores.
    """
    output = metadata.reset_index(drop=True).copy()

    for i in range(scores.shape[1]):
        output[f"PC{i + 1}"] = scores[:, i]

    return output


def build_loading_table(
    pca: PCA,
) -> pd.DataFrame:
    """
    Build a feature-loading table for all retained components.

    sklearn PCA components are eigenvectors with shape:
        n_components x n_features
    """
    rows = []

    for feature_index, feature in enumerate(FEATURES):
        row = {
            "feature": feature.replace("_z", ""),
            "feature_label": FEATURE_LABELS[feature],
        }

        for component_index in range(
            pca.components_.shape[0]
        ):
            row[
                f"PC{component_index + 1}_loading"
            ] = pca.components_[
                component_index,
                feature_index,
            ]

        rows.append(row)

    return pd.DataFrame(rows)


def build_variance_table(
    pca: PCA,
) -> pd.DataFrame:
    """
    Build explained-variance statistics.
    """
    rows = []

    cumulative = 0.0

    for i, ratio in enumerate(
        pca.explained_variance_ratio_
    ):
        cumulative += ratio

        rows.append(
            {
                "component": f"PC{i + 1}",
                "explained_variance": (
                    pca.explained_variance_[i]
                ),
                "explained_variance_ratio": ratio,
                "explained_variance_percent": (
                    ratio * 100.0
                ),
                "cumulative_variance_ratio": cumulative,
                "cumulative_variance_percent": (
                    cumulative * 100.0
                ),
            }
        )

    return pd.DataFrame(rows)


def build_centroid_table(
    score_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate language-group centroids in PCA space.
    """
    pc_columns = [
        column
        for column in score_table.columns
        if column.startswith("PC")
    ]

    centroids = (
        score_table
        .groupby(
            "language",
            as_index=False,
        )[pc_columns]
        .mean()
    )

    return centroids


def save_outputs(
    score_table: pd.DataFrame,
    loading_table: pd.DataFrame,
    variance_table: pd.DataFrame,
    centroid_table: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """
    Save PCA outputs.
    """
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    score_table.to_csv(
        output_dir / "pca_scores.csv",
        index=False,
    )

    loading_table.to_csv(
        output_dir / "pca_loadings.csv",
        index=False,
    )

    variance_table.to_csv(
        output_dir / "pca_explained_variance.csv",
        index=False,
    )

    centroid_table.to_csv(
        output_dir / "pca_group_centroids.csv",
        index=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PCA for the Mandarin-English "
            "prosody study."
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "results/tables/"
            "standardized_features.csv"
        ),
        help="Input standardized feature table.",
    )

    parser.add_argument(
        "--output-dir",
        default="results/tables",
        help="Directory for PCA output tables.",
    )

    parser.add_argument(
        "--components",
        type=int,
        default=2,
        help=(
            "Number of principal components "
            "to retain. Default: 2."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_standardized_data(
        args.input
    )

    metadata, matrix = prepare_feature_matrix(
        df
    )

    pca, scores = run_pca(
        matrix,
        n_components=args.components,
    )

    score_table = build_score_table(
        metadata,
        scores,
    )

    loading_table = build_loading_table(
        pca
    )

    variance_table = build_variance_table(
        pca
    )

    centroid_table = build_centroid_table(
        score_table
    )

    save_outputs(
        score_table=score_table,
        loading_table=loading_table,
        variance_table=variance_table,
        centroid_table=centroid_table,
        output_dir=args.output_dir,
    )

    print("\nPCA completed.")

    print(
        f"Samples included: {len(score_table)}"
    )

    for _, row in variance_table.iterrows():
        print(
            f"{row['component']}: "
            f"{row['explained_variance_percent']:.2f}%"
        )

    print(
        "\nCumulative variance explained: "
        f"{variance_table.iloc[-1]['cumulative_variance_percent']:.2f}%"
    )

    print("\nFeature loadings:")
    print(
        loading_table.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
