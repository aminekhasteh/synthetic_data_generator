"""Statistical fidelity metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wasserstein_distance

from ..data import DEFAULT_TARGET_COLUMN

if TYPE_CHECKING:
    from ..config import DatasetConfig


def _numeric_columns(
    df: pd.DataFrame,
    target_column: str | None,
) -> list[str]:
    cols = []
    for c in df.columns:
        if c == target_column:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def compute_fidelity_metrics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    config: DatasetConfig | None = None,
) -> dict:
    """Compute univariate and bivariate statistical fidelity metrics."""
    target = config.target_column if config else (
        DEFAULT_TARGET_COLUMN if DEFAULT_TARGET_COLUMN in real.columns else None
    )
    numeric_cols = _numeric_columns(real, target)
    ks_stats = []
    wasserstein_dists = []
    mean_pct_errors = []
    std_pct_errors = []

    for col in numeric_cols:
        r = real[col].dropna().values
        s = synthetic[col].dropna().values
        if len(r) == 0 or len(s) == 0:
            continue
        ks_stats.append(stats.ks_2samp(r, s).statistic)
        wasserstein_dists.append(wasserstein_distance(r, s))

        r_mean, s_mean = np.mean(r), np.mean(s)
        r_std, s_std = np.std(r), np.std(s)
        if abs(r_mean) > 1e-9:
            mean_pct_errors.append(abs(s_mean - r_mean) / abs(r_mean))
        if abs(r_std) > 1e-9:
            std_pct_errors.append(abs(s_std - r_std) / abs(r_std))

    real_corr = real[numeric_cols].corr().values
    synth_corr = synthetic[numeric_cols].corr().values
    corr_diff = np.linalg.norm(real_corr - synth_corr, ord="fro")

    metrics = {
        "fidelity_ks_mean": float(np.mean(ks_stats)) if ks_stats else np.nan,
        "fidelity_ks_max": float(np.max(ks_stats)) if ks_stats else np.nan,
        "fidelity_wasserstein_mean": float(np.mean(wasserstein_dists))
        if wasserstein_dists
        else np.nan,
        "fidelity_mean_pct_error": float(np.mean(mean_pct_errors))
        if mean_pct_errors
        else np.nan,
        "fidelity_std_pct_error": float(np.mean(std_pct_errors))
        if std_pct_errors
        else np.nan,
        "fidelity_corr_l2_diff": float(corr_diff),
    }

    if target and target in real.columns and target in synthetic.columns:
        prevalence_error = float(abs(synthetic[target].mean() - real[target].mean()))
        metrics["fidelity_target_prevalence_error"] = prevalence_error
        metrics["fidelity_class_prevalence_error"] = prevalence_error
    else:
        metrics["fidelity_target_prevalence_error"] = np.nan
        metrics["fidelity_class_prevalence_error"] = np.nan

    return metrics
