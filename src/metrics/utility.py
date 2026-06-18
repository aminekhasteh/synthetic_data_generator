"""Downstream utility metrics (TSTR / TRTR)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from ..data import DEFAULT_TARGET_COLUMN
from .encoding import encode_features

if TYPE_CHECKING:
    from ..config import DatasetConfig


def _train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
) -> dict:
    feature_cols = [c for c in train_df.columns if c != target_column]
    y_train = train_df[target_column].values
    y_test = test_df[target_column].values

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {"auroc": np.nan, "pr_auc": np.nan, "f1": np.nan}

    X_train, X_test, _ = encode_features(train_df, test_df, feature_cols)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)

    return {
        "auroc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
    }


def compute_utility_metrics(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    real_holdout: pd.DataFrame,
    config: DatasetConfig | None = None,
) -> dict:
    """Compute TRTR and TSTR downstream utility metrics."""
    target = config.target_column if config else (
        DEFAULT_TARGET_COLUMN if DEFAULT_TARGET_COLUMN in real_train.columns else None
    )
    if not target or target not in real_train.columns:
        return {
            "utility_trtr_auroc": np.nan,
            "utility_trtr_pr_auc": np.nan,
            "utility_trtr_f1": np.nan,
            "utility_tstr_auroc": np.nan,
            "utility_tstr_pr_auc": np.nan,
            "utility_tstr_f1": np.nan,
            "utility_relative_auroc": np.nan,
            "utility_skipped": 1.0,
        }

    trtr = _train_and_evaluate(real_train, real_holdout, target)
    tstr = _train_and_evaluate(synthetic, real_holdout, target)

    relative = (
        tstr["auroc"] / trtr["auroc"]
        if trtr["auroc"] and not np.isnan(trtr["auroc"]) and trtr["auroc"] > 0
        else np.nan
    )

    return {
        "utility_trtr_auroc": trtr["auroc"],
        "utility_trtr_pr_auc": trtr["pr_auc"],
        "utility_trtr_f1": trtr["f1"],
        "utility_tstr_auroc": tstr["auroc"],
        "utility_tstr_pr_auc": tstr["pr_auc"],
        "utility_tstr_f1": tstr["f1"],
        "utility_relative_auroc": float(relative) if not np.isnan(relative) else np.nan,
        "utility_skipped": 0.0,
    }
