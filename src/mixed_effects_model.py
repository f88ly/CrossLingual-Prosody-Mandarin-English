"""
Linear mixed-effects modeling for the Mandarin-English prosody study.

Primary model:
    F0 mean ~ Language + (1 | Speaker)

This script:
1. Loads utterance-level acoustic features
2. Fits a random-intercept linear mixed-effects model
3. Exports fixed-effect estimates
4. Exports model summary statistics
5. Exports speaker-level random intercepts
6. Exports plotting data for Figure 8

Expected input:
    results/tables/acoustic_features.csv

Required columns:
    language
    speaker_id
    f0_mean_hz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def load_data(
    input_path: str | Path,
) -> pd.DataFrame:
    """
    Load acoustic feature data and validate required columns.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    required_columns = {
        "language",
        "speaker_id",
        "f0_mean_hz",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Input table is missing required columns: "
            f"{sorted(missing)}"
        )

    return df


def normalize_language_labels(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize language labels to ZH and EN.
    """
    df = df.copy()

    mapping: Dict[str, str] = {
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
        lambda x: mapping.get(
            str(x).strip(),
            str(x).strip(),
        )
    )

    unexpected = (
        set(df["language"].dropna().unique())
        - {"ZH", "EN"}
    )

    if unexpected:
        raise ValueError(
            "Unexpected language labels found: "
            f"{sorted(unexpected)}"
        )

    return df


def prepare_model_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare complete cases for the mixed-effects model.
    """
    model_df = (
        df[
            [
                "language",
                "speaker_id",
                "f0_mean_hz",
            ]
        ]
        .dropna()
        .copy()
    )

    model_df["speaker_id"] = (
        model_df["speaker_id"].astype(str)
    )

    model_df["f0_mean_hz"] = pd.to_numeric(
        model_df["f0_mean_hz"],
        errors="coerce",
    )

    model_df = (
        model_df
        .dropna(
            subset=["f0_mean_hz"]
        )
        .reset_index(drop=True)
    )

    if model_df.empty:
        raise ValueError(
            "No complete observations are available "
            "for mixed-effects modeling."
        )

    if model_df["language"].nunique() != 2:
        raise ValueError(
            "The model requires both ZH and EN samples."
        )

    if model_df["speaker_id"].nunique() < 2:
        raise ValueError(
            "At least two speakers are required "
            "for random-intercept modeling."
        )

    return model_df


def fit_mixed_model(
    model_df: pd.DataFrame,
    reference_language: str = "EN",
):
    """
    Fit:

        F0 mean ~ Language + (1 | Speaker)

    Language is treated as a categorical fixed effect.
    Speaker is modeled as a random intercept.

    The default reference category is English (EN), so the ZH
    coefficient represents:

        mean_F0_ZH - mean_F0_EN

    after accounting for speaker-level random intercepts.
    """
    if reference_language not in {"ZH", "EN"}:
        raise ValueError(
            "reference_language must be 'ZH' or 'EN'."
        )

    formula = (
        "f0_mean_hz ~ "
        f"C(language, Treatment(reference='{reference_language}'))"
    )

    model = smf.mixedlm(
        formula=formula,
        data=model_df,
        groups=model_df["speaker_id"],
        re_formula="1",
    )

    result = model.fit(
        reml=True,
        method="lbfgs",
    )

    return result


def build_fixed_effect_table(
    result,
) -> pd.DataFrame:
    """
    Export fixed-effect estimates and inferential statistics.
    """
    params = result.fe_params
    standard_errors = result.bse_fe
    pvalues = result.pvalues.loc[
        params.index
    ]

    conf_int = result.conf_int().loc[
        params.index
    ]

    rows = []

    for parameter in params.index:
        rows.append(
            {
                "parameter": parameter,
                "estimate": float(
                    params[parameter]
                ),
                "std_error": float(
                    standard_errors[parameter]
                ),
                "z_value": float(
                    params[parameter]
                    / standard_errors[parameter]
                )
                if standard_errors[parameter] != 0
                else np.nan,
                "p_value": float(
                    pvalues[parameter]
                ),
                "ci_lower_95": float(
                    conf_int.loc[
                        parameter
                    ].iloc[0]
                ),
                "ci_upper_95": float(
                    conf_int.loc[
                        parameter
                    ].iloc[1]
                ),
            }
        )

    return pd.DataFrame(rows)


def build_random_effect_table(
    result,
) -> pd.DataFrame:
    """
    Export speaker-level random intercepts.
    """
    rows = []

    for speaker_id, random_effect in (
        result.random_effects.items()
    ):
        values = np.asarray(
            random_effect
        ).reshape(-1)

        intercept = (
            float(values[0])
            if values.size > 0
            else np.nan
        )

        rows.append(
            {
                "speaker_id": str(
                    speaker_id
                ),
                "random_intercept": intercept,
            }
        )

    output = pd.DataFrame(rows)

    if not output.empty:
        output = output.sort_values(
            by="random_intercept"
        ).reset_index(
            drop=True
        )

    return output


def build_model_statistics(
    result,
    model_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Export general model statistics.
    """
    random_variance = np.nan

    try:
        random_variance = float(
            result.cov_re.iloc[0, 0]
        )
    except Exception:
        pass

    residual_variance = float(
        result.scale
    )

    return pd.DataFrame(
        [
            {
                "n_observations": int(
                    result.nobs
                ),
                "n_speakers": int(
                    model_df[
                        "speaker_id"
                    ].nunique()
                ),
                "log_likelihood": float(
                    result.llf
                ),
                "aic": float(
                    result.aic
                )
                if np.isfinite(result.aic)
                else np.nan,
                "bic": float(
                    result.bic
                )
                if np.isfinite(result.bic)
                else np.nan,
                "random_intercept_variance": (
                    random_variance
                ),
                "residual_variance": (
                    residual_variance
                ),
                "converged": bool(
                    result.converged
                ),
            }
        ]
    )


def build_language_summary(
    model_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate descriptive F0 summary by language for Figure 8A.
    """
    summary = (
        model_df
        .groupby(
            "language",
            as_index=False,
        )
        .agg(
            n=("f0_mean_hz", "size"),
            mean_f0_hz=(
                "f0_mean_hz",
                "mean",
            ),
            std_f0_hz=(
                "f0_mean_hz",
                "std",
            ),
            median_f0_hz=(
                "f0_mean_hz",
                "median",
            ),
        )
    )

    summary["se_f0_hz"] = (
        summary["std_f0_hz"]
        / np.sqrt(
            summary["n"]
        )
    )

    return summary


def save_model_summary_text(
    result,
    output_path: str | Path,
) -> None:
    """
    Save the full statsmodels textual model summary.
    """
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            result.summary().as_text()
        )


def save_outputs(
    fixed_effects: pd.DataFrame,
    random_effects: pd.DataFrame,
    model_statistics: pd.DataFrame,
    language_summary: pd.DataFrame,
    model_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """
    Save all mixed-effects outputs.
    """
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fixed_effects.to_csv(
        output_dir
        / "lmm_fixed_effects.csv",
        index=False,
    )

    random_effects.to_csv(
        output_dir
        / "lmm_random_intercepts.csv",
        index=False,
    )

    model_statistics.to_csv(
        output_dir
        / "lmm_model_statistics.csv",
        index=False,
    )

    language_summary.to_csv(
        output_dir
        / "figure8_fixed_effect_data.csv",
        index=False,
    )

    random_effects.to_csv(
        output_dir
        / "figure8_random_effect_data.csv",
        index=False,
    )

    model_df.to_csv(
        output_dir
        / "lmm_analysis_dataset.csv",
        index=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a linear mixed-effects model "
            "for F0 mean."
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "results/tables/"
            "acoustic_features.csv"
        ),
        help=(
            "Input acoustic feature CSV."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="results/tables",
        help=(
            "Directory for mixed-effects outputs."
        ),
    )

    parser.add_argument(
        "--reference-language",
        choices=[
            "EN",
            "ZH",
        ],
        default="EN",
        help=(
            "Reference category for the language "
            "fixed effect. Default: EN."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_data(
        args.input
    )

    df = normalize_language_labels(
        df
    )

    model_df = prepare_model_data(
        df
    )

    result = fit_mixed_model(
        model_df,
        reference_language=(
            args.reference_language
        ),
    )

    fixed_effects = (
        build_fixed_effect_table(
            result
        )
    )

    random_effects = (
        build_random_effect_table(
            result
        )
    )

    model_statistics = (
        build_model_statistics(
            result,
            model_df,
        )
    )

    language_summary = (
        build_language_summary(
            model_df
        )
    )

    save_outputs(
        fixed_effects=(
            fixed_effects
        ),
        random_effects=(
            random_effects
        ),
        model_statistics=(
            model_statistics
        ),
        language_summary=(
            language_summary
        ),
        model_df=model_df,
        output_dir=(
            args.output_dir
        ),
    )

    save_model_summary_text(
        result,
        Path(
            args.output_dir
        )
        / "lmm_model_summary.txt",
    )

    print(
        "\nLinear mixed-effects "
        "model completed."
    )

    print(
        f"Observations: "
        f"{len(model_df)}"
    )

    print(
        f"Speakers: "
        f"{model_df['speaker_id'].nunique()}"
    )

    print(
        f"Reference language: "
        f"{args.reference_language}"
    )

    print(
        "\nFixed effects:"
    )

    print(
        fixed_effects.to_string(
            index=False
        )
    )

    print(
        "\nModel statistics:"
    )

    print(
        model_statistics.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
