"""
Generate manuscript figures for the Mandarin-English prosody study.

Figures:
    Figure 5  Raw acoustic feature distributions
    Figure 6  Standardized seven-feature distributions
    Figure 7  PCA scores and feature loadings
    Figure 8  Linear mixed-effects model results
    Figure 9  Cohen's d effect sizes

Expected analysis outputs:
    results/tables/figure5_distribution_data.csv
    results/tables/figure6_standardized_data.csv
    results/tables/pca_scores.csv
    results/tables/pca_loadings.csv
    results/tables/pca_explained_variance.csv
    results/tables/figure8_fixed_effect_data.csv
    results/tables/figure8_random_effect_data.csv
    results/tables/figure9_effect_sizes.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURE_LABELS = {
    "f0_mean_hz": "F0 mean (Hz)",
    "f0_range_hz": "F0 range (Hz)",
    "rms_energy": "RMS energy",
    "duration_s": "Duration (s)",
    "f0_mean_hz_z": "F0 mean",
    "f0_std_hz_z": "F0 std",
    "f0_range_hz_z": "F0 range",
    "rms_energy_z": "RMS energy",
    "duration_s_z": "Duration",
    "pause_ratio_z": "Pause ratio",
    "voicing_ratio_z": "Voicing ratio",
}


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
) -> None:
    """
    Save manuscript figures in PNG, PDF, and SVG formats.
    """
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir / f"{stem}.svg",
        bbox_inches="tight",
    )

    plt.close(fig)


def language_values(
    df: pd.DataFrame,
    feature: str,
    language: str,
) -> np.ndarray:
    return (
        df.loc[
            df["language"] == language,
            feature,
        ]
        .dropna()
        .astype(float)
        .to_numpy()
    )


def figure5(
    table_dir: Path,
    figure_dir: Path,
) -> None:
    """
    Figure 5:
    Distributional comparison of four major acoustic features.
    """
    path = table_dir / "figure5_distribution_data.csv"
    df = pd.read_csv(path)

    features = [
        "f0_mean_hz",
        "f0_range_hz",
        "rms_energy",
        "duration_s",
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 8),
    )

    axes = axes.flatten()

    for ax, feature in zip(axes, features):
        zh = language_values(df, feature, "ZH")
        en = language_values(df, feature, "EN")

        data = [zh, en]

        violin = ax.violinplot(
            data,
            positions=[1, 2],
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        for body in violin["bodies"]:
            body.set_alpha(0.45)

        ax.boxplot(
            data,
            positions=[1, 2],
            widths=0.20,
            showfliers=False,
        )

        rng = np.random.default_rng(42)

        for position, values in zip(
            [1, 2],
            data,
        ):
            if len(values) == 0:
                continue

            jitter = rng.normal(
                loc=position,
                scale=0.035,
                size=len(values),
            )

            ax.scatter(
                jitter,
                values,
                s=6,
                alpha=0.22,
            )

        ax.set_xticks([1, 2])
        ax.set_xticklabels(
            ["Mandarin", "English"]
        )

        ax.set_ylabel(
            FEATURE_LABELS[feature]
        )

        ax.set_title(
            FEATURE_LABELS[feature]
        )

    fig.tight_layout()

    save_figure(
        fig,
        figure_dir,
        "Figure5_distribution",
    )


def figure6(
    table_dir: Path,
    figure_dir: Path,
) -> None:
    """
    Figure 6:
    Standardized distributions of seven acoustic features.
    """
    path = table_dir / "figure6_standardized_data.csv"
    df = pd.read_csv(path)

    features = [
        "f0_mean_hz_z",
        "f0_std_hz_z",
        "f0_range_hz_z",
        "rms_energy_z",
        "duration_s_z",
        "pause_ratio_z",
        "voicing_ratio_z",
    ]

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(14, 7),
    )

    axes = axes.flatten()

    for ax, feature in zip(
        axes,
        features,
    ):
        zh = language_values(
            df,
            feature,
            "ZH",
        )

        en = language_values(
            df,
            feature,
            "EN",
        )

        data = [zh, en]

        violin = ax.violinplot(
            data,
            positions=[1, 2],
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        for body in violin["bodies"]:
            body.set_alpha(0.45)

        ax.boxplot(
            data,
            positions=[1, 2],
            widths=0.20,
            showfliers=False,
        )

        ax.axhline(
            0,
            linewidth=0.8,
        )

        ax.set_xticks([1, 2])
        ax.set_xticklabels(
            ["ZH", "EN"]
        )

        ax.set_title(
            FEATURE_LABELS[feature]
        )

        ax.set_ylabel("z-score")

    if len(axes) > len(features):
        for ax in axes[len(features):]:
            ax.axis("off")

    fig.tight_layout()

    save_figure(
        fig,
        figure_dir,
        "Figure6_standardized",
    )


def figure7(
    table_dir: Path,
    figure_dir: Path,
) -> None:
    """
    Figure 7:
    PCA sample projection and PC1/PC2 feature loadings.
    """
    scores = pd.read_csv(
        table_dir / "pca_scores.csv"
    )

    loadings = pd.read_csv(
        table_dir / "pca_loadings.csv"
    )

    variance = pd.read_csv(
        table_dir / "pca_explained_variance.csv"
    )

    pc1_percent = float(
        variance.loc[
            variance["component"] == "PC1",
            "explained_variance_percent",
        ].iloc[0]
    )

    pc2_percent = float(
        variance.loc[
            variance["component"] == "PC2",
            "explained_variance_percent",
        ].iloc[0]
    )

    fig = plt.figure(
        figsize=(10, 10)
    )

    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.4, 1],
    )

    ax_scatter = fig.add_subplot(
        grid[0, :]
    )

    for language, marker in [
        ("ZH", "o"),
        ("EN", "^"),
    ]:
        group = scores[
            scores["language"] == language
        ]

        ax_scatter.scatter(
            group["PC1"],
            group["PC2"],
            label=language,
            marker=marker,
            s=22,
            alpha=0.45,
        )

        if len(group):
            ax_scatter.scatter(
                group["PC1"].mean(),
                group["PC2"].mean(),
                marker="X",
                s=150,
                edgecolors="black",
                label=f"{language} centroid",
            )

    ax_scatter.axhline(
        0,
        linewidth=0.7,
    )

    ax_scatter.axvline(
        0,
        linewidth=0.7,
    )

    ax_scatter.set_xlabel(
        f"PC1 ({pc1_percent:.2f}%)"
    )

    ax_scatter.set_ylabel(
        f"PC2 ({pc2_percent:.2f}%)"
    )

    ax_scatter.set_title(
        "PCA projection of acoustic features"
    )

    ax_scatter.legend(
        frameon=False
    )

    ax_pc1 = fig.add_subplot(
        grid[1, 0]
    )

    ax_pc2 = fig.add_subplot(
        grid[1, 1]
    )

    labels = (
        loadings["feature_label"]
        .astype(str)
        .tolist()
    )

    positions = np.arange(
        len(labels)
    )

    ax_pc1.barh(
        positions,
        loadings["PC1_loading"],
    )

    ax_pc1.set_yticks(
        positions
    )

    ax_pc1.set_yticklabels(
        labels
    )

    ax_pc1.axvline(
        0,
        linewidth=0.8,
    )

    ax_pc1.set_title(
        "PC1 feature loadings"
    )

    ax_pc1.set_xlabel(
        "Loading"
    )

    ax_pc2.barh(
        positions,
        loadings["PC2_loading"],
    )

    ax_pc2.set_yticks(
        positions
    )

    ax_pc2.set_yticklabels(
        labels
    )

    ax_pc2.axvline(
        0,
        linewidth=0.8,
    )

    ax_pc2.set_title(
        "PC2 feature loadings"
    )

    ax_pc2.set_xlabel(
        "Loading"
    )

    fig.tight_layout()

    save_figure(
        fig,
        figure_dir,
        "Figure7_PCA",
    )


def figure8(
    table_dir: Path,
    figure_dir: Path,
) -> None:
    """
    Figure 8:
    F0 language summary and speaker-level random intercepts.
    """
    fixed = pd.read_csv(
        table_dir / "figure8_fixed_effect_data.csv"
    )

    random = pd.read_csv(
        table_dir / "figure8_random_effect_data.csv"
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 5),
    )

    fixed = fixed.set_index(
        "language"
    )

    languages = [
        language
        for language in ["ZH", "EN"]
        if language in fixed.index
    ]

    means = [
        fixed.loc[
            language,
            "mean_f0_hz",
        ]
        for language in languages
    ]

    errors = [
        fixed.loc[
            language,
            "se_f0_hz",
        ]
        for language in languages
    ]

    axes[0].bar(
        languages,
        means,
        yerr=errors,
        capsize=4,
    )

    axes[0].set_ylabel(
        "F0 mean (Hz)"
    )

    axes[0].set_xlabel(
        "Language"
    )

    axes[0].set_title(
        "(A) Language-level F0 mean"
    )

    random = random.sort_values(
        "random_intercept"
    ).reset_index(drop=True)

    axes[1].barh(
        np.arange(len(random)),
        random["random_intercept"],
    )

    axes[1].axvline(
        0,
        linewidth=0.8,
    )

    axes[1].set_xlabel(
        "Random intercept"
    )

    axes[1].set_ylabel(
        "Speaker"
    )

    axes[1].set_yticks([])

    axes[1].set_title(
        "(B) Speaker-level random intercepts"
    )

    fig.tight_layout()

    save_figure(
        fig,
        figure_dir,
        "Figure8_LMM",
    )


def figure9(
    table_dir: Path,
    figure_dir: Path,
) -> None:
    """
    Figure 9:
    Cohen's d effect sizes across acoustic features.
    """
    df = pd.read_csv(
        table_dir / "figure9_effect_sizes.csv"
    )

    df = df.sort_values(
        "cohens_d"
    ).reset_index(drop=True)

    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    positions = np.arange(
        len(df)
    )

    ax.barh(
        positions,
        df["cohens_d"],
    )

    ax.axvline(
        0,
        linewidth=0.9,
    )

    ax.set_yticks(
        positions
    )

    ax.set_yticklabels(
        df["feature_label"]
    )

    ax.set_xlabel(
        "Cohen's d"
    )

    ax.set_title(
        "Effect sizes across acoustic features"
    )

    for i, value in enumerate(
        df["cohens_d"]
    ):
        if pd.isna(value):
            continue

        horizontal_alignment = (
            "left"
            if value >= 0
            else "right"
        )

        offset = (
            0.04
            if value >= 0
            else -0.04
        )

        ax.text(
            value + offset,
            i,
            f"{value:.2f}",
            va="center",
            ha=horizontal_alignment,
        )

    fig.tight_layout()

    save_figure(
        fig,
        figure_dir,
        "Figure9_effect_sizes",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Figures 5-9 for the "
            "Mandarin-English prosody study."
        )
    )

    parser.add_argument(
        "--table-dir",
        default="results/tables",
        help="Directory containing analysis CSV files.",
    )

    parser.add_argument(
        "--figure-dir",
        default="figures",
        help="Directory for generated figures.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    table_dir = Path(
        args.table_dir
    )

    figure_dir = ensure_output_dir(
        args.figure_dir
    )

    print("Generating Figure 5...")
    figure5(
        table_dir,
        figure_dir,
    )

    print("Generating Figure 6...")
    figure6(
        table_dir,
        figure_dir,
    )

    print("Generating Figure 7...")
    figure7(
        table_dir,
        figure_dir,
    )

    print("Generating Figure 8...")
    figure8(
        table_dir,
        figure_dir,
    )

    print("Generating Figure 9...")
    figure9(
        table_dir,
        figure_dir,
    )

    print("\nFigures completed.")
    print(
        f"Output directory: {figure_dir}"
    )


if __name__ == "__main__":
    main()
