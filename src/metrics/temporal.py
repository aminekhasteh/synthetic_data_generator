"""Temporal structure metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from ..config import DatasetConfig


def _inter_arrival_gaps(df: pd.DataFrame, time_col: str) -> np.ndarray:
    times = df.sort_values(time_col)[time_col].values
    if len(times) < 2:
        return np.array([])
    gaps = np.diff(times)
    return gaps[gaps >= 0]


def _binned_counts(df: pd.DataFrame, time_col: str, n_bins: int = 20) -> np.ndarray:
    times = df[time_col].values
    if len(times) == 0:
        return np.zeros(n_bins)
    edges = np.linspace(times.min(), times.max(), n_bins + 1)
    counts, _ = np.histogram(times, bins=edges)
    return counts.astype(float)


def _target_rate_by_quartile(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
) -> np.ndarray:
    if len(df) == 0:
        return np.zeros(4)
    work = df.copy()
    work["time_quartile"] = pd.qcut(
        work[time_col], q=4, labels=False, duplicates="drop"
    )
    rates = work.groupby("time_quartile", observed=True)[target_col].mean()
    return rates.reindex(range(4), fill_value=0.0).values


def compute_temporal_metrics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    config: DatasetConfig | None = None,
    n_bins: int = 20,
) -> dict:
    """Compare temporal patterns between real and synthetic data."""
    time_col = config.temporal_column if config else None
    if not time_col:
        for candidate in ("Time", "time", "timestamp", "datetime", "date"):
            if candidate in real.columns:
                time_col = candidate
                break

    if not time_col or time_col not in real.columns:
        return {
            "temporal_time_ks": np.nan,
            "temporal_gap_ks": np.nan,
            "temporal_bin_rate_error": np.nan,
            "temporal_quartile_target_error": np.nan,
            "temporal_skipped": 1.0,
        }

    real_time = real[time_col].dropna().values
    synth_time = synthetic[time_col].dropna().values
    time_ks = (
        stats.ks_2samp(real_time, synth_time).statistic
        if len(real_time) and len(synth_time)
        else np.nan
    )

    real_gaps = _inter_arrival_gaps(real, time_col)
    synth_gaps = _inter_arrival_gaps(synthetic, time_col)
    gap_ks = (
        stats.ks_2samp(real_gaps, synth_gaps).statistic
        if len(real_gaps) and len(synth_gaps)
        else np.nan
    )

    real_bins = _binned_counts(real, time_col, n_bins)
    synth_bins = _binned_counts(synthetic, time_col, n_bins)
    total_real = real_bins.sum() or 1.0
    total_synth = synth_bins.sum() or 1.0
    bin_rate_error = float(
        np.mean(np.abs(real_bins / total_real - synth_bins / total_synth))
    )

    target = config.target_column if config else None
    quartile_error = np.nan
    if target and target in real.columns:
        real_q = _target_rate_by_quartile(real, time_col, target)
        synth_q = _target_rate_by_quartile(synthetic, time_col, target)
        quartile_error = float(np.mean(np.abs(real_q - synth_q)))

    return {
        "temporal_time_ks": float(time_ks),
        "temporal_gap_ks": float(gap_ks),
        "temporal_bin_rate_error": bin_rate_error,
        "temporal_quartile_target_error": quartile_error,
        "temporal_quartile_fraud_error": quartile_error,
        "temporal_skipped": 0.0,
    }
