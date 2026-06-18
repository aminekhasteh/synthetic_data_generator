"""Privacy risk metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

if TYPE_CHECKING:
    from ..config import DatasetConfig


def _feature_columns(df: pd.DataFrame, target_column: str | None) -> list[str]:
    if target_column and target_column in df.columns:
        return [c for c in df.columns if c != target_column]
    return list(df.columns)


def _encode_mixed_frame(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Encode mixed numeric/categorical columns to numeric arrays for distance metrics."""
    real_block = real[cols].copy()
    synth_block = synthetic[cols].copy()
    encoded_real = []
    encoded_synth = []

    for col in cols:
        r = real_block[col]
        s = synth_block[col]
        if pd.api.types.is_numeric_dtype(r):
            encoded_real.append(pd.to_numeric(r, errors="coerce").fillna(0).values)
            encoded_synth.append(pd.to_numeric(s, errors="coerce").fillna(0).values)
        else:
            combined = pd.concat([r.astype(str), s.astype(str)], ignore_index=True)
            encoder = LabelEncoder()
            encoder.fit(combined)
            encoded_real.append(encoder.transform(r.astype(str)))
            encoded_synth.append(encoder.transform(s.astype(str)))

    return np.column_stack(encoded_real), np.column_stack(encoded_synth)


def _gower_distance_matrix(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    target_column: str | None,
) -> np.ndarray:
    """Compute Gower-like distance from each synthetic row to nearest real row."""
    cols = _feature_columns(real, target_column)
    r, s = _encode_mixed_frame(real, synthetic, cols)

    ranges = np.maximum(r.max(axis=0) - r.min(axis=0), 1e-9)
    n_synth = len(s)
    min_dists = np.empty(n_synth)

    for i in range(n_synth):
        diff = np.abs(r - s[i])
        dists = np.mean(diff / ranges, axis=1)
        min_dists[i] = dists.min()
    return min_dists


def _nn_overlap_rate(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    target_column: str | None,
) -> float:
    """Fraction of synthetic rows sharing the same nearest real neighbor."""
    cols = _feature_columns(real, target_column)
    r, s = _encode_mixed_frame(real, synthetic, cols)

    scaler = StandardScaler()
    r_scaled = scaler.fit_transform(r)
    s_scaled = scaler.transform(s)

    nn = NearestNeighbors(n_neighbors=1).fit(r_scaled)
    _, indices = nn.kneighbors(s_scaled)
    flat = indices.flatten()
    unique, counts = np.unique(flat, return_counts=True)
    shared = counts[counts > 1]
    if len(shared) == 0:
        return 0.0
    return float(shared.sum() / len(flat))


def _exact_match_count(real: pd.DataFrame, synthetic: pd.DataFrame) -> int:
    merged = synthetic.merge(real, how="inner")
    return len(merged)


def _encode_for_classifier(df: pd.DataFrame, cols: list[str], encoders: dict) -> np.ndarray:
    parts = []
    for col in cols:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            parts.append(pd.to_numeric(series, errors="coerce").fillna(0).values.reshape(-1, 1))
        else:
            if col not in encoders:
                encoders[col] = LabelEncoder()
                encoders[col].fit(series.astype(str))
            parts.append(encoders[col].transform(series.astype(str)).reshape(-1, 1))
    return np.hstack(parts)


def _membership_inference_auc(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    real_holdout: pd.DataFrame,
    target_column: str | None,
) -> float:
    """Train shadow model to distinguish members vs non-members; test on synthetic."""
    feature_cols = _feature_columns(real_train, target_column)
    encoders: dict = {}

    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(real_train[col]):
            combined = pd.concat(
                [
                    real_train[col].astype(str),
                    real_holdout[col].astype(str),
                    synthetic[col].astype(str),
                ],
                ignore_index=True,
            )
            encoders[col] = LabelEncoder()
            encoders[col].fit(combined)

    member_x = _encode_for_classifier(real_train, feature_cols, encoders)
    non_member_x = _encode_for_classifier(real_holdout, feature_cols, encoders)
    test_x = _encode_for_classifier(synthetic, feature_cols, encoders)

    X_shadow = np.vstack([member_x, non_member_x])
    y_shadow = np.array([1] * len(member_x) + [0] * len(non_member_x))

    if len(np.unique(y_shadow)) < 2:
        return np.nan

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_shadow, y_shadow, test_size=0.3, random_state=42, stratify=y_shadow
    )
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_tr, y_tr)

    try:
        scores = clf.predict_proba(test_x)[:, 1]
        return float(roc_auc_score(np.ones(len(scores)), scores))
    except ValueError:
        return np.nan


def compute_privacy_metrics(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    real_holdout: pd.DataFrame,
    privacy_budget: dict | None = None,
    config: DatasetConfig | None = None,
) -> dict:
    """Compute empirical privacy risk metrics."""
    target = config.target_column if config else None
    dcr = _gower_distance_matrix(real_train, synthetic, target)
    metrics = {
        "privacy_exact_match_count": _exact_match_count(real_train, synthetic),
        "privacy_dcr_mean": float(np.mean(dcr)) if len(dcr) else np.nan,
        "privacy_dcr_min": float(np.min(dcr)) if len(dcr) else np.nan,
        "privacy_nn_overlap_rate": _nn_overlap_rate(real_train, synthetic, target),
        "privacy_mia_auc": _membership_inference_auc(
            real_train, synthetic, real_holdout, target
        ),
    }
    if privacy_budget:
        metrics.update(
            {
                "privacy_epsilon_spent": privacy_budget.get("privacy_epsilon_spent"),
                "privacy_delta_spent": privacy_budget.get("privacy_delta_spent"),
            }
        )
    return metrics
