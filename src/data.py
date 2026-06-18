"""Data loading, sampling, and splitting utilities."""

from __future__ import annotations

from pathlib import Path

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
TARGET_COLUMN = "Class"
ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def download_creditcard_dataset() -> Path:
    """Download Kaggle credit card fraud dataset via kagglehub."""
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    dataset_path = Path(path)
    csv_path = dataset_path / "creditcard.csv"
    if not csv_path.exists():
        candidates = list(dataset_path.rglob("creditcard.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"creditcard.csv not found under {dataset_path}. "
                "Ensure Kaggle credentials are configured (~/.kaggle/kaggle.json)."
            )
        csv_path = candidates[0]
    return csv_path


def load_creditcard(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load the credit card fraud dataset."""
    if csv_path is None:
        csv_path = download_creditcard_dataset()
    df = pd.read_csv(csv_path)
    missing = set(ALL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing expected columns: {missing}")
    return df[ALL_COLUMNS].copy()


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
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Draw a stratified sample preserving class balance."""
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
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Stratified bootstrap resample with replacement."""
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create seed sample, holdout test set, and bootstrap pool.

    Returns (seed_df, holdout_df, pool_df) with no row overlap.
    """
    if pool_size + holdout_size > len(df):
        pool_size = min(pool_size, len(df) // 2)
        holdout_size = min(holdout_size, len(df) - pool_size)

    pool_and_holdout = stratified_sample(
        df, n=pool_size + holdout_size, random_state=random_state
    )
    pool_df, holdout_df = train_test_split(
        pool_and_holdout,
        test_size=holdout_size,
        stratify=pool_and_holdout[TARGET_COLUMN],
        random_state=random_state,
    )
    pool_df = pool_df.reset_index(drop=True)
    holdout_df = holdout_df.reset_index(drop=True)

    seed_df = stratified_sample(pool_df, n=seed_size, random_state=random_state)
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
