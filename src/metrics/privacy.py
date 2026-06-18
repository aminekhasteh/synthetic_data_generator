"""Privacy risk metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ..data import TARGET_COLUMN


def _gower_distance_matrix(real: pd.DataFrame, synthetic: pd.DataFrame) -> np.ndarray:
    """Compute Gower-like distance from each synthetic row to nearest real row."""
    cols = [c for c in real.columns if c != TARGET_COLUMN]
    r = real[cols].values.astype(float)
    s = synthetic[cols].values.astype(float)

    ranges = np.maximum(r.max(axis=0) - r.min(axis=0), 1e-9)
    n_synth = len(s)
    min_dists = np.empty(n_synth)

    for i in range(n_synth):
        diff = np.abs(r - s[i])
        dists = np.mean(diff / ranges, axis=1)
        min_dists[i] = dists.min()
    return min_dists


def _nn_overlap_rate(real: pd.DataFrame, synthetic: pd.DataFrame) -> float:
    """Fraction of synthetic rows sharing the same nearest real neighbor."""
    cols = [c for c in real.columns if c != TARGET_COLUMN]
    scaler = StandardScaler()
    r = scaler.fit_transform(real[cols].values)
    s = scaler.transform(synthetic[cols].values)

    nn = NearestNeighbors(n_neighbors=1).fit(r)
    _, indices = nn.kneighbors(s)
    flat = indices.flatten()
    unique, counts = np.unique(flat, return_counts=True)
    shared = counts[counts > 1]
    if len(shared) == 0:
        return 0.0
    return float(shared.sum() / len(flat))


def _exact_match_count(real: pd.DataFrame, synthetic: pd.DataFrame) -> int:
    merged = synthetic.merge(real, how="inner")
    return len(merged)


def _membership_inference_auc(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    real_holdout: pd.DataFrame,
) -> float:
    """Train shadow model to distinguish members vs non-members; test on synthetic."""
    feature_cols = [c for c in real_train.columns if c != TARGET_COLUMN]

    member = real_train[feature_cols].copy()
    member["is_member"] = 1
    non_member = real_holdout[feature_cols].copy()
    non_member["is_member"] = 0
    shadow = pd.concat([member, non_member], ignore_index=True)

    X_shadow = shadow[feature_cols].values
    y_shadow = shadow["is_member"].values
    X_test = synthetic[feature_cols].values

    if len(np.unique(y_shadow)) < 2:
        return np.nan

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_shadow, y_shadow, test_size=0.3, random_state=42, stratify=y_shadow
    )
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_tr, y_tr)

    try:
        scores = clf.predict_proba(X_test)[:, 1]
        return float(roc_auc_score(np.ones(len(scores)), scores))
    except ValueError:
        return np.nan


def compute_privacy_metrics(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    real_holdout: pd.DataFrame,
    privacy_budget: dict | None = None,
) -> dict:
    """Compute empirical privacy risk metrics."""
    dcr = _gower_distance_matrix(real_train, synthetic)
    metrics = {
        "privacy_exact_match_count": _exact_match_count(real_train, synthetic),
        "privacy_dcr_mean": float(np.mean(dcr)) if len(dcr) else np.nan,
        "privacy_dcr_min": float(np.min(dcr)) if len(dcr) else np.nan,
        "privacy_nn_overlap_rate": _nn_overlap_rate(real_train, synthetic),
        "privacy_mia_auc": _membership_inference_auc(
            real_train, synthetic, real_holdout
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
