"""Optional Kaggle dataset helpers (requires kagglehub)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
TARGET_COLUMN = "Class"
ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def _require_kagglehub():
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "kagglehub is not installed. Install optional dependencies with:\n"
            "  pip install -r requirements-kaggle.txt"
        ) from exc
    return kagglehub


def download_creditcard_dataset() -> Path:
    """Download Kaggle credit card fraud dataset via kagglehub."""
    kagglehub = _require_kagglehub()
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
