"""
Statistical analysis for cross-linguistic prosody comparison.

This script performs:
1. Descriptive statistics
2. Two-sided Mann-Whitney U tests
3. Multiple-comparison correction
4. Cohen's d effect-size estimation
5. Z-score standardization
6. Export of datasets used for Figures 5, 6, and 9

Expected input:
    results/tables/acoustic_features.csv

Core acoustic features:
    - F0 mean
    - F0 standard deviation
    - F0 range
    - RMS energy
    - Duration
    - Pause ratio
    - Voicing ratio

Effect-size direction:
    Positive Cohen's d = higher values in Mandarin (ZH)
    Negative Cohen's d = higher values in English (EN)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler


FEATURES: Dict[str, str] = {
    "f0_mean_hz": "F0 mean",
    "f0_std_hz": "F0 std",
    "f0_range_hz": "F0 range",
    "rms_energy": "RMS energy",
    "duration_s": "Duration",
    "pause_ratio": "Pause ratio",
    "voicing_ratio": "Voicing ratio",
}


def load_features(input_path: str | Path) -> pd.DataFrame:
    """
    Load the acoustic feature table and validate required columns.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Feature table not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    required_columns = {
        "language",
        *FEATURES.keys(),
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Input table is missing required columns: "
            f"{sorted(missing)}"
        )

    return df


def normalize_language_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common language labels to ZH and EN.

    Accepted Mandarin labels include:
        ZH, Mandarin, Mandarin Chinese, Chinese

    Accepted English labels include:
        EN, English
    """
    df = df.copy()

    mapping = {
        "ZH": "ZH",
        "zh": "ZH",
        "Mandarin": "ZH",
        "mandarin": "ZH",
        "Mandarin Chinese": "ZH",
        "Chinese": "ZH",
        "chinese": "ZH",
        "EN": "EN",
        "en": "EN",
        "English": "EN",
        "english": "EN",
    }

    df["language"] = df["language"].map(
        lambda x: mapping.get(str(x).strip(), str(x).strip())
    )

    unexpected = set(df["language"].dropna().unique()) - {"ZH", "EN"}

    if unexpected:
        raise ValueError(
            "Unexpected language labels found: "
            f"{sorted(unexpected)}"
        )

    return df


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate descriptive statistics separately for ZH and EN.
    """
    rows: List[dict] = []

    for feature, feature_label in FEATURES.items():
        for language in ["ZH", "EN"]:
            values = (
                df.loc[df["language"] == language, feature]
                .dropna()
                .astype(float)
            )

            row = {
                "feature": feature,
                "feature_label": feature_label,
                "language": language,
                "n": len(values),
                "mean": values.mean() if len(values) else np.nan,
                "std": values.std(ddof=1) if len(values) > 1 else np.nan,
                "median": values.median() if len(values) else np.nan,
                "q1": values.quantile(0.25) if len(values) else np.nan,
                "q3": values.quantile(0.75) if len(values) else np.nan,
                "min": values.min() if len(values) else np.nan,
                "max": values.max() if len(values) else np.nan,
            }

            rows.append(row)

    return pd.DataFrame(rows)


def pooled_standard_deviation(
    group_zh: np.ndarray,
    group_en: np.ndarray,
) -> float:
    """
    Compute pooled sample standard deviation used in Cohen's d.
    """
    n_zh = len(group_zh)
    n_en = len(group_en)

    if n_zh < 2 or n_en < 2:
        return np.nan

    var_zh = np.var(group_zh, ddof=1)
    var_en = np.var(group_en, ddof=1)

    denominator = n_zh + n_en - 2

    if denominator <= 0:
        return np.nan

    pooled_variance = (
        ((n_zh - 1) * var_zh)
        + ((n_en - 1) * var_en)
    ) / denominator

    if pooled_variance < 0:
        return np.nan

    return float(np.sqrt(pooled_variance))


def cohens_d(
    group_zh: np.ndarray,
    group_en: np.ndarray,
) -> float:
    """
    Calculate Cohen's d using:

        d = (mean_ZH - mean_EN) / pooled_SD

    Therefore:
        d > 0  -> higher values in Mandarin (ZH)
        d < 0  -> higher values in English (EN)
    """
    pooled_sd = pooled_standard_deviation(
        group_zh,
        group_en,
    )

    if np.isnan(pooled_sd) or pooled_sd == 0:
        return np.nan

    return float(
        (np.mean(group_zh) - np.mean(group_en))
        / pooled_sd
    )


def significance_label(p_value: float) -> str:
    """
    Convert p-value into conventional significance symbols.
    """
    if np.isnan(p_value):
        return ""

    if p_value < 0.001:
        return "***"

    if p_value < 0.01:
        return "**"

    if p_value < 0.05:
        return "*"

    return "ns"


def run_group_comparisons(
    df: pd.DataFrame,
    correction_method: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Perform Mann-Whitney U tests and Cohen's d for all features.

    Multiple-testing correction is applied across the seven acoustic
    features.
    """
    rows: List[dict] = []

    for feature, feature_label in FEATURES.items():
        zh = (
            df.loc[df["language"] == "ZH", feature]
            .dropna()
            .astype(float)
            .to_numpy()
        )

        en = (
            df.loc[df["language"] == "EN", feature]
            .dropna()
            .astype(float)
            .to_numpy()
        )

        if len(zh) == 0 or len(en) == 0:
            statistic = np.nan
            raw_p = np.nan
            effect = np.nan

        else:
            test = mannwhitneyu(
                zh,
                en,
                alternative="two-sided",
            )

            statistic = float(test.statistic)
            raw_p = float(test.pvalue)

            effect = cohens_d(
                zh,
                en,
            )

        rows.append(
            {
                "feature": feature,
                "feature_label": feature_label,
                "n_ZH": len(zh),
                "n_EN": len(en),
                "mean_ZH": np.mean(zh) if len(zh) else np.nan,
                "mean_EN": np.mean(en) if len(en) else np.nan,
                "median_ZH": np.median(zh) if len(zh) else np.nan,
                "median_EN": np.median(en) if len(en) else np.nan,
                "mannwhitney_U": statistic,
                "p_raw": raw_p,
                "cohens_d": effect,
            }
        )

    results = pd.DataFrame(rows)

    valid_mask = results["p_raw"].notna()

    results["p_adjusted"] = np.nan
    results["reject_null"] = False

    if valid_mask.sum() > 0:
        reject, adjusted_p, _, _ = multipletests(
            results.loc[valid_mask, "p_raw"],
            alpha=alpha,
            method=correction_method,
        )

        results.loc[
            valid_mask,
            "p_adjusted",
        ] = adjusted_p

        results.loc[
            valid_mask,
            "reject_null",
        ] = reject

    results["significance"] = results[
        "p_adjusted"
    ].apply(significance_label)

    results["abs_cohens_d"] = results[
        "cohens_d"
    ].abs()

    results["effect_direction"] = np.where(
        results["cohens_d"] > 0,
        "ZH > EN",
        np.where(
            results["cohens_d"] < 0,
            "EN > ZH",
            "No directional difference",
        ),
    )

    return results


def create_standardized_feature_table(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Z-score standardize the seven acoustic features across the complete
    analysis dataset.

    The standardized output is intended for multivariate analyses and
    Figure 6.
    """
    output = df.copy()

    scaler = StandardScaler()

    feature_matrix = output[
        list(FEATURES.keys())
    ].astype(float)

    valid_rows = feature_matrix.notna().all(axis=1)

    standardized_columns = [
        f"{feature}_z"
        for feature in FEATURES
    ]

    for column in standardized_columns:
        output[column] = np.nan

    if valid_rows.sum() > 0:
        standardized = scaler.fit_transform(
            feature_matrix.loc[valid_rows]
        )

        output.loc[
            valid_rows,
            standardized_columns,
        ] = standardized

    return output


def create_figure5_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Export raw values for Figure 5.

    Current manuscript Figure 5 focuses on:
        F0 mean
        F0 range
        RMS energy
        Duration
    """
    figure5_features = [
        "f0_mean_hz",
        "f0_range_hz",
        "rms_energy",
        "duration_s",
    ]

    columns = [
        column
        for column in [
            "language",
            "corpus",
            "speaker_id",
            "utterance_id",
            *figure5_features,
        ]
        if column in df.columns
    ]

    return df[columns].copy()


def create_figure6_data(
    standardized_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Export standardized seven-feature data for Figure 6.
    """
    columns = [
        column
        for column in [
            "language",
            "corpus",
            "speaker_id",
            "utterance_id",
            *[
                f"{feature}_z"
                for feature in FEATURES
            ],
        ]
        if column in standardized_df.columns
    ]

    return standardized_df[columns].copy()


def create_figure9_data(
    comparison_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Export effect sizes used for Figure 9.
    """
    columns = [
        "feature",
        "feature_label",
        "cohens_d",
        "abs_cohens_d",
        "effect_direction",
        "p_raw",
        "p_adjusted",
        "significance",
    ]

    return comparison_results[
        columns
    ].sort_values(
        by="cohens_d",
        ascending=True,
    )


def save_outputs(
    descriptive: pd.DataFrame,
    comparisons: pd.DataFrame,
    standardized: pd.DataFrame,
    figure5: pd.DataFrame,
    figure6: pd.DataFrame,
    figure9: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """
    Save all statistical-analysis outputs.
    """
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptive.to_csv(
        output_dir / "descriptive_statistics.csv",
        index=False,
    )

    comparisons.to_csv(
        output_dir / "group_comparisons.csv",
        index=False,
    )

    standardized.to_csv(
        output_dir / "standardized_features.csv",
        index=False,
    )

    figure5.to_csv(
        output_dir / "figure5_distribution_data.csv",
        index=False,
    )

    figure6.to_csv(
        output_dir / "figure6_standardized_data.csv",
        index=False,
    )

    figure9.to_csv(
        output_dir / "figure9_effect_sizes.csv",
        index=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run statistical analyses for the "
            "Mandarin-English prosody study."
        )
    )

    parser.add_argument(
        "--input",
        default="results/tables/acoustic_features.csv",
        help="Input acoustic feature CSV.",
    )

    parser.add_argument(
        "--output-dir",
        default="results/tables",
        help="Directory for statistical output tables.",
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
            "Multiple-comparison correction method. "
            "Use the method actually employed in the study."
        ),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Statistical significance level. Default: 0.05.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_features(args.input)

    df = normalize_language_labels(df)

    descriptive = descriptive_statistics(df)

    comparisons = run_group_comparisons(
        df,
        correction_method=args.correction,
        alpha=args.alpha,
    )

    standardized = create_standardized_feature_table(
        df
    )

    figure5 = create_figure5_data(df)

    figure6 = create_figure6_data(
        standardized
    )

    figure9 = create_figure9_data(
        comparisons
    )

    save_outputs(
        descriptive=descriptive,
        comparisons=comparisons,
        standardized=standardized,
        figure5=figure5,
        figure6=figure6,
        figure9=figure9,
        output_dir=args.output_dir,
    )

    print("\nStatistical analysis completed.")
    print(
        f"Input samples: {len(df)}"
    )
    print(
        f"Multiple-comparison method: "
        f"{args.correction}"
    )
    print(
        f"Output directory: "
        f"{args.output_dir}"
    )

    print("\nEffect sizes:")
    print(
        comparisons[
            [
                "feature_label",
                "cohens_d",
                "p_adjusted",
                "significance",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
