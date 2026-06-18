"""Shared encoding helpers for mixed-type tabular data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Label-encode categoricals; return train/test feature matrices and encoders."""
    encoders: dict[str, LabelEncoder] = {}
    train_parts = []
    test_parts = []

    for col in feature_cols:
        train_series = train_df[col]
        test_series = test_df[col]
        if pd.api.types.is_numeric_dtype(train_series):
            train_parts.append(pd.to_numeric(train_series, errors="coerce").fillna(0).values.reshape(-1, 1))
            test_parts.append(pd.to_numeric(test_series, errors="coerce").fillna(0).values.reshape(-1, 1))
        else:
            combined = pd.concat(
                [train_series.astype(str), test_series.astype(str)],
                ignore_index=True,
            )
            encoders[col] = LabelEncoder()
            encoders[col].fit(combined)
            train_parts.append(encoders[col].transform(train_series.astype(str)).reshape(-1, 1))
            test_parts.append(encoders[col].transform(test_series.astype(str)).reshape(-1, 1))

    return np.hstack(train_parts), np.hstack(test_parts), encoders
