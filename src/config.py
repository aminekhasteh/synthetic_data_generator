"""Dataset configuration and data-dictionary parsing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import yaml


@dataclass
class DatasetConfig:
    """Configuration for an arbitrary seed dataset."""

    name: str
    columns: list[str]
    target_column: str | None = None
    temporal_column: str | None = None
    categorical_columns: list[str] = field(default_factory=list)
    continuous_columns: list[str] = field(default_factory=list)
    table_name: str = "synthetic_data"

    @property
    def feature_columns(self) -> list[str]:
        if self.target_column and self.target_column in self.columns:
            return [c for c in self.columns if c != self.target_column]
        return list(self.columns)

    def infer_column_types(self, df: pd.DataFrame) -> None:
        """Fill categorical/continuous lists when not provided in the dictionary."""
        if self.categorical_columns and self.continuous_columns:
            return

        inferred_cat = []
        inferred_cont = []
        for col in self.columns:
            if col == self.target_column:
                inferred_cat.append(col)
                continue
            if col in self.categorical_columns:
                continue
            if col in self.continuous_columns:
                continue
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                nunique = series.nunique(dropna=True)
                if nunique <= 20 and nunique / max(len(series), 1) < 0.05:
                    inferred_cat.append(col)
                else:
                    inferred_cont.append(col)
            else:
                inferred_cat.append(col)

        if not self.categorical_columns:
            self.categorical_columns = inferred_cat
        if not self.continuous_columns:
            self.continuous_columns = inferred_cont

    def to_dict(self) -> dict:
        return asdict(self)


def _load_dictionary_file(path: Path) -> dict:
    suffix = path.suffix.lower()
    with open(path) as f:
        if suffix in {".yaml", ".yml"}:
            return yaml.safe_load(f) or {}
        if suffix == ".json":
            return json.load(f)
        raise ValueError(f"Unsupported data dictionary format: {suffix}")


def load_dataset_config(
    seed_path: Path,
    columns: list[str] | None = None,
    dictionary_path: Path | None = None,
    dataset_name: str | None = None,
) -> DatasetConfig:
    """Build a DatasetConfig from CLI arguments and an optional data dictionary."""
    name = dataset_name or seed_path.stem
    meta: dict = {}
    if dictionary_path is not None:
        meta = _load_dictionary_file(dictionary_path)

    if columns is None:
        columns = meta.get("columns")
    if not columns:
        raise ValueError(
            "No columns specified. Pass --columns or include 'columns' in the data dictionary."
        )

    target = meta.get("target_column")
    temporal = meta.get("temporal_column")
    categorical = meta.get("categorical_columns", [])
    continuous = meta.get("continuous_columns", [])
    table_name = meta.get("table_name", name)

    if target and target not in columns:
        raise ValueError(f"target_column '{target}' not in selected columns")
    if temporal and temporal not in columns:
        raise ValueError(f"temporal_column '{temporal}' not in selected columns")

    return DatasetConfig(
        name=name,
        columns=columns,
        target_column=target,
        temporal_column=temporal,
        categorical_columns=list(categorical),
        continuous_columns=list(continuous),
        table_name=table_name,
    )


def load_seed_dataframe(seed_path: Path, config: DatasetConfig) -> pd.DataFrame:
    """Load seed dataset and restrict to configured columns."""
    suffix = seed_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(seed_path)
    elif suffix == ".parquet":
        df = pd.read_parquet(seed_path)
    elif suffix in {".json"}:
        df = pd.read_json(seed_path)
    else:
        raise ValueError(f"Unsupported seed dataset format: {suffix}")

    missing = set(config.columns) - set(df.columns)
    if missing:
        raise ValueError(f"Seed dataset missing columns: {sorted(missing)}")

    return df[config.columns].copy()
