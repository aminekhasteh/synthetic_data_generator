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
    column_specs: dict[str, dict] = field(default_factory=dict)
    table_name: str = "synthetic_data"
    inferred: bool = False

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
            spec = self.column_specs.get(col, {})
            spec_type = spec.get("type", "").lower()
            if spec_type == "categorical":
                inferred_cat.append(col)
                continue
            if spec_type in {"continuous", "numeric", "float", "int"}:
                inferred_cont.append(col)
                continue
            if spec_type == "temporal":
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


def load_dictionary_file(path: Path) -> dict:
    suffix = path.suffix.lower()
    with open(path) as f:
        if suffix in {".yaml", ".yml"}:
            return yaml.safe_load(f) or {}
        if suffix == ".json":
            return json.load(f)
        raise ValueError(f"Unsupported data dictionary format: {suffix}")


def build_dataset_config(
    name: str,
    columns: list[str],
    meta: dict | None = None,
) -> DatasetConfig:
    """Build a DatasetConfig from explicit columns and optional dictionary metadata."""
    meta = meta or {}
    target = meta.get("target_column")
    temporal = meta.get("temporal_column")
    categorical = list(meta.get("categorical_columns") or [])
    continuous = list(meta.get("continuous_columns") or [])
    column_specs = dict(meta.get("column_specs") or {})
    table_name = meta.get("table_name", name)

    if not categorical and not continuous and column_specs:
        for col, spec in column_specs.items():
            if col not in columns:
                continue
            spec_type = str(spec.get("type", "")).lower()
            if spec_type == "categorical":
                categorical.append(col)
            elif spec_type in {"continuous", "numeric", "float", "int", "temporal"}:
                continuous.append(col)

    if target and target not in columns:
        raise ValueError(f"target_column '{target}' not in selected columns")
    if temporal and temporal not in columns:
        raise ValueError(f"temporal_column '{temporal}' not in selected columns")

    return DatasetConfig(
        name=name,
        columns=columns,
        target_column=target,
        temporal_column=temporal,
        categorical_columns=categorical,
        continuous_columns=continuous,
        column_specs=column_specs,
        table_name=table_name,
        inferred=bool(meta.get("_inferred")),
    )


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
        meta = load_dictionary_file(dictionary_path)

    resolved_columns = columns or meta.get("columns")
    if not resolved_columns:
        return build_dataset_config(name=name, columns=[], meta=meta)

    return build_dataset_config(name=name, columns=list(resolved_columns), meta=meta)
