"""Generic data loading, investigation, transformation, and sampling utilities."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from .config import DatasetConfig

logger = logging.getLogger(__name__)

# Backward-compatible default used only when no config target is available
DEFAULT_TARGET_COLUMN = "Class"

_ID_COLUMN_PATTERN = re.compile(r"^(id|index|row_?id|unnamed: 0)$", re.I)
_TARGET_NAME_CANDIDATES = ("class", "target", "label", "y", "is_fraud", "fraud")
_TEMPORAL_NAME_CANDIDATES = ("time", "timestamp", "datetime", "date", "event_time")


def read_raw_dataframe(path: str | Path) -> pd.DataFrame:
    """Read a dataset file into a DataFrame without column filtering."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Seed dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError(
        f"Unsupported seed dataset format '{suffix}'. Use CSV, Parquet, JSON, or TSV."
    )


def _column_summary(series: pd.Series) -> dict:
    summary: dict = {
        "dtype": str(series.dtype),
        "null_count": int(series.isna().sum()),
        "null_pct": float(series.isna().mean()),
        "nunique": int(series.nunique(dropna=True)),
    }
    if pd.api.types.is_numeric_dtype(series):
        summary.update(
            {
                "min": float(series.min()) if series.notna().any() else None,
                "max": float(series.max()) if series.notna().any() else None,
                "mean": float(series.mean()) if series.notna().any() else None,
                "std": float(series.std()) if series.notna().any() else None,
            }
        )
    else:
        top = series.value_counts(dropna=True).head(3)
        summary["top_values"] = {str(k): int(v) for k, v in top.items()}
    return summary


def investigate_dataset(df: pd.DataFrame) -> dict:
    """Profile a dataset: shape, dtypes, nulls, and per-column summaries."""
    profile = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "null_counts": {c: int(df[c].isna().sum()) for c in df.columns},
        "column_summaries": {c: _column_summary(df[c]) for c in df.columns},
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1e6),
    }
    logger.info(
        "Dataset profile: %s rows, %s columns, %.2f MB, %s duplicate rows",
        profile["n_rows"],
        profile["n_columns"],
        profile["memory_mb"],
        profile["duplicate_rows"],
    )
    for col, summary in profile["column_summaries"].items():
        if summary["null_count"]:
            logger.info("  %s: %s nulls (%.1f%%)", col, summary["null_count"], summary["null_pct"] * 100)
    return profile


def _is_id_column(name: str) -> bool:
    return bool(_ID_COLUMN_PATTERN.match(name.strip()))


def _find_by_candidates(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    for col in columns:
        if any(candidate in col.lower() for candidate in candidates):
            return col
    return None


def _infer_binary_target(df: pd.DataFrame, columns: list[str]) -> str | None:
    named = _find_by_candidates(columns, _TARGET_NAME_CANDIDATES)
    if named:
        return named
    for col in columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            values = set(series.dropna().unique())
            if values.issubset({0, 1}) and len(values) == 2:
                return col
    return None


def infer_fallback_config(
    df: pd.DataFrame,
    name: str,
    columns: list[str] | None = None,
) -> DatasetConfig:
    """Infer columns, target, temporal, and types when no data dictionary is provided."""
    from .config import DatasetConfig

    if columns:
        selected = [c for c in columns if c in df.columns]
        missing = set(columns) - set(df.columns)
        if missing:
            raise ValueError(f"Requested columns not found in dataset: {sorted(missing)}")
    else:
        selected = [
            c for c in df.columns
            if not _is_id_column(str(c))
        ]

    if not selected:
        raise ValueError("No usable columns found after excluding ID-like columns.")

    target = _infer_binary_target(df, selected)
    temporal = _find_by_candidates(selected, _TEMPORAL_NAME_CANDIDATES)

    categorical: list[str] = []
    continuous: list[str] = []
    column_specs: dict[str, dict] = {}

    for col in selected:
        series = df[col]
        if col == target:
            categorical.append(col)
            column_specs[col] = {"type": "categorical"}
            continue
        if col == temporal:
            column_specs[col] = {"type": "temporal"}
            if not pd.api.types.is_numeric_dtype(series):
                column_specs[col]["parse_datetime"] = True
            continuous.append(col)
            continue
        if pd.api.types.is_numeric_dtype(series):
            nunique = series.nunique(dropna=True)
            if nunique <= 20 and nunique / max(len(series), 1) < 0.05:
                categorical.append(col)
                column_specs[col] = {"type": "categorical"}
            else:
                continuous.append(col)
                column_specs[col] = {"type": "continuous"}
        else:
            categorical.append(col)
            column_specs[col] = {"type": "categorical"}

    logger.info(
        "Inferred config (no dictionary): target=%s temporal=%s categorical=%s continuous=%s",
        target,
        temporal,
        len(categorical),
        len(continuous),
    )

    return DatasetConfig(
        name=name,
        columns=selected,
        target_column=target,
        temporal_column=temporal,
        categorical_columns=categorical,
        continuous_columns=continuous,
        column_specs=column_specs,
        table_name=name,
        inferred=True,
    )


def apply_column_transforms(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """Apply per-column transforms defined in the data dictionary."""
    out = df[config.columns].copy()

    for col in config.columns:
        spec = config.column_specs.get(col, {})
        if not spec:
            continue

        series = out[col]

        if spec.get("drop"):
            logger.warning("Column %s marked drop=true but is still in selected columns", col)
            continue

        if spec.get("parse_datetime") and not pd.api.types.is_datetime64_any_dtype(series):
            out[col] = pd.to_datetime(series, errors="coerce")
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = out[col].astype("int64") // 10**9

        fillna = spec.get("fillna")
        if fillna is not None:
            out[col] = series.fillna(fillna)

        cast = spec.get("cast")
        if cast:
            out[col] = _cast_series(out[col], cast)

        clip = spec.get("clip")
        if clip and len(clip) == 2 and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].clip(float(clip[0]), float(clip[1]))

        log_transform = spec.get("log_transform")
        if log_transform and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = np.log1p(out[col].clip(lower=0))

        rename_to = spec.get("rename_to")
        if rename_to and rename_to != col:
            out = out.rename(columns={col: rename_to})

    if config.target_column and config.target_column in out.columns:
        out[config.target_column] = _normalize_binary_target(out[config.target_column])

    if config.target_column and config.target_column in out.columns:
        out = out.dropna(subset=[config.target_column])
    return out.reset_index(drop=True)


def _cast_series(series: pd.Series, cast: str) -> pd.Series:
    cast = str(cast).lower()
    if cast in {"int", "integer"}:
        return pd.to_numeric(series, errors="coerce").round().astype("Int64")
    if cast in {"float", "double", "numeric"}:
        return pd.to_numeric(series, errors="coerce").astype(float)
    if cast in {"str", "string", "category"}:
        return series.astype(str)
    if cast in {"bool", "boolean"}:
        return series.astype(bool)
    raise ValueError(f"Unsupported cast type: {cast}")


def _normalize_binary_target(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        unique = set(numeric.dropna().unique())
        if unique.issubset({0, 1}):
            return numeric.round().astype(int)
    return series


def load_and_prepare_dataset(
    seed_path: Path,
    config: DatasetConfig | None = None,
    dictionary_meta: dict | None = None,
    columns: list[str] | None = None,
    dataset_name: str | None = None,
) -> tuple[pd.DataFrame, DatasetConfig, dict]:
    """
    Load, investigate, infer config (if needed), and transform a seed dataset.

    Returns (prepared_df, config, investigation_profile).
    """
    from .config import build_dataset_config

    raw = read_raw_dataframe(seed_path)
    profile = investigate_dataset(raw)
    name = dataset_name or seed_path.stem

    if config is None:
        if dictionary_meta and dictionary_meta.get("columns"):
            config = build_dataset_config(
                name=name,
                columns=list(dictionary_meta["columns"]),
                meta=dictionary_meta,
            )
        elif columns:
            meta = dictionary_meta or {}
            config = build_dataset_config(name=name, columns=columns, meta=meta)
        elif dictionary_meta:
            inferred_cols = [
                c for c in raw.columns if not _is_id_column(str(c))
            ]
            config = build_dataset_config(name=name, columns=inferred_cols, meta=dictionary_meta)
        else:
            config = infer_fallback_config(raw, name=name, columns=None)

    missing = set(config.columns) - set(raw.columns)
    if missing:
        raise ValueError(f"Seed dataset missing columns: {sorted(missing)}")

    config.infer_column_types(raw[config.columns])
    prepared = apply_column_transforms(raw, config)
    profile["prepared_rows"] = len(prepared)
    profile["config_inferred"] = config.inferred

    logger.info(
        "Prepared dataset: %s rows x %s columns (dropped %s rows during transforms)",
        len(prepared),
        len(config.columns),
        len(raw) - len(prepared),
    )
    return prepared, config, profile


def save_data_profile(profile: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(profile, f, indent=2, default=str)


def random_sample(
    df: pd.DataFrame,
    n: int,
    random_state: int = 42,
    **_,
) -> pd.DataFrame:
    """Draw a random sample without stratification."""
    if n > len(df):
        raise ValueError(f"Sample size {n} exceeds dataset size {len(df)}")
    return df.sample(n=n, random_state=random_state).reset_index(drop=True)


def random_bootstrap(
    df: pd.DataFrame,
    n: int,
    random_state: int = 42,
    **_,
) -> pd.DataFrame:
    """Bootstrap resample with replacement (no stratification)."""
    return df.sample(n=n, replace=True, random_state=random_state).reset_index(drop=True)


def stratified_sample(
    df: pd.DataFrame,
    n: int,
    random_state: int = 42,
    min_fraud_count: int = 1,
    target_column: str | None = None,
) -> pd.DataFrame:
    """Draw a stratified sample preserving class balance."""
    if not target_column:
        return random_sample(df, n, random_state)
    if n > len(df):
        raise ValueError(f"Sample size {n} exceeds dataset size {len(df)}")

    positive = df[df[target_column] == 1]
    negative = df[df[target_column] == 0]

    if len(positive) == 0 or len(negative) == 0:
        return random_sample(df, n, random_state)

    pos_rate = len(positive) / len(df)
    n_pos = max(min_fraud_count, round(n * pos_rate))
    n_pos = min(n_pos, len(positive))
    n_neg = n - n_pos

    if n_neg > len(negative):
        raise ValueError(
            f"Cannot sample {n} rows: only {len(negative)} negative rows available."
        )

    pos_sample = positive.sample(n=n_pos, random_state=random_state)
    neg_sample = negative.sample(n=n_neg, random_state=random_state)
    sample = pd.concat([pos_sample, neg_sample], ignore_index=True)
    return sample.sample(frac=1, random_state=random_state).reset_index(drop=True)


def stratified_bootstrap(
    df: pd.DataFrame,
    n: int,
    random_state: int,
    min_fraud_count: int = 1,
    target_column: str | None = None,
) -> pd.DataFrame:
    """Stratified bootstrap resample with replacement."""
    if not target_column:
        return random_bootstrap(df, n, random_state)

    positive = df[df[target_column] == 1]
    negative = df[df[target_column] == 0]

    if len(positive) == 0 or len(negative) == 0:
        return random_bootstrap(df, n, random_state)

    pos_rate = len(positive) / len(df)
    n_pos = max(min_fraud_count, round(n * pos_rate))
    n_pos = min(n_pos, len(positive))
    n_neg = n - n_pos

    pos_sample = positive.sample(n=n_pos, replace=True, random_state=random_state)
    neg_sample = negative.sample(n=n_neg, replace=True, random_state=random_state + 1)
    sample = pd.concat([pos_sample, neg_sample], ignore_index=True)
    return sample.sample(frac=1, random_state=random_state + 2).reset_index(drop=True)


def create_seed_and_holdout(
    df: pd.DataFrame,
    seed_size: int = 2000,
    holdout_size: int = 5000,
    pool_size: int = 10000,
    random_state: int = 42,
    target_column: str | None = DEFAULT_TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create seed sample, holdout test set, and bootstrap pool."""
    if pool_size + holdout_size > len(df):
        pool_size = min(pool_size, len(df) // 2)
        holdout_size = min(holdout_size, len(df) - pool_size)

    pool_and_holdout = stratified_sample(
        df,
        n=pool_size + holdout_size,
        random_state=random_state,
        target_column=target_column,
    )
    stratify = (
        pool_and_holdout[target_column]
        if target_column and target_column in pool_and_holdout.columns
        else None
    )
    pool_df, holdout_df = train_test_split(
        pool_and_holdout,
        test_size=holdout_size,
        stratify=stratify,
        random_state=random_state,
    )
    pool_df = pool_df.reset_index(drop=True)
    holdout_df = holdout_df.reset_index(drop=True)

    seed_df = stratified_sample(
        pool_df,
        n=seed_size,
        random_state=random_state,
        target_column=target_column,
    )
    return seed_df, holdout_df, pool_df


def aggregate_bootstrap_metrics(
    metrics_list: list[dict],
    ci_low: float = 0.025,
    ci_high: float = 0.975,
) -> pd.DataFrame:
    """Aggregate bootstrap replicate metrics into mean, std, and confidence intervals."""
    df = pd.DataFrame(metrics_list)
    numeric_cols = df.select_dtypes(include="number").columns

    rows = []
    for col in numeric_cols:
        values = df[col].dropna()
        if len(values) == 0:
            continue
        rows.append(
            {
                "metric": col,
                "mean": values.mean(),
                "std": values.std(),
                "ci_low": values.quantile(ci_low),
                "ci_high": values.quantile(ci_high),
                "n": len(values),
            }
        )
    return pd.DataFrame(rows)
