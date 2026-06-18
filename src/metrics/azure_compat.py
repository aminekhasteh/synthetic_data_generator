"""Azure compatibility checklist."""

from __future__ import annotations

import json
import sys
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import yaml

if TYPE_CHECKING:
    from ..config import DatasetConfig


def _python_version_ok() -> bool:
    v = sys.version_info
    return (3, 10) <= (v.major, v.minor) < (3, 15)


def _column_metadata(df: pd.DataFrame, col: str) -> dict:
    series = df[col]
    if pd.api.types.is_numeric_dtype(series):
        nunique = series.nunique(dropna=True)
        if nunique <= 20:
            return {"type": "int", "lower": int(series.min()), "upper": int(series.max())}
        return {
            "type": "float",
            "lower": float(series.min()),
            "upper": float(series.max()),
        }
    return {"type": "string"}


def _build_metadata_yaml(
    df: pd.DataFrame,
    path: Path,
    table_name: str,
) -> None:
    """Write minimal smartnoise-sql metadata."""
    table_meta: dict = {"row_privacy": True, "rows": len(df)}
    for col in df.columns:
        table_meta[col] = _column_metadata(df, col)

    metadata = {table_name: {table_name: {table_name: table_meta}}}
    with open(path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)


def _numeric_column_for_query(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None


def _dp_sql_smoke_test(
    df: pd.DataFrame,
    metadata_path: Path,
    table_name: str,
) -> bool:
    """Run a private SQL aggregate query against synthetic data."""
    num_col = _numeric_column_for_query(df)
    if not num_col:
        return False
    query = f"SELECT AVG({num_col}), COUNT(*) FROM {table_name}.{table_name}"
    try:
        import snsql
        from snsql import Privacy

        privacy = Privacy(epsilon=1.0, delta=1e-5)
        reader = snsql.from_df(
            df,
            privacy=privacy,
            metadata=str(metadata_path),
        )
        result = reader.execute(query)
        return result is not None and len(result) > 0
    except Exception:
        return False


def run_azure_compat_checklist(
    synthetic: pd.DataFrame,
    privacy_budget: dict,
    output_dir: str = "outputs",
    synthesizer_name: str = "mwem",
    config: DatasetConfig | None = None,
) -> dict:
    """Run Azure compatibility checks and write export artifacts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table_name = config.table_name if config else "synthetic_data"

    csv_path = out / "synthetic_export.csv"
    parquet_path = out / "synthetic_export.parquet"
    metadata_path = out / "metadata.yaml"
    report_path = out / "azure_compat_report.json"

    synthetic.to_csv(csv_path, index=False)
    synthetic.to_parquet(parquet_path, index=False)
    _build_metadata_yaml(synthetic, metadata_path, table_name)

    checks = {
        "azure_python_version_ok": _python_version_ok(),
        "azure_csv_export_ok": csv_path.exists(),
        "azure_parquet_export_ok": parquet_path.exists(),
        "azure_metadata_yaml_ok": metadata_path.exists(),
        "azure_dp_sql_smoke_test_ok": _dp_sql_smoke_test(
            synthetic, metadata_path, table_name
        ),
    }

    report = {
        "checks": checks,
        "all_passed": all(checks.values()),
        "privacy_budget": privacy_budget,
        "synthesizer": synthesizer_name,
        "table_name": table_name,
        "python_version": sys.version,
        "packages": {},
        "artifacts": {
            "csv": str(csv_path),
            "parquet": str(parquet_path),
            "metadata": str(metadata_path),
        },
    }
    for pkg in ("smartnoise-synth", "smartnoise-sql", "pandas"):
        try:
            report["packages"][pkg] = version(pkg)
        except Exception:
            report["packages"][pkg] = "unknown"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return {
        "azure_all_passed": float(all(checks.values())),
        "azure_python_version_ok": float(checks["azure_python_version_ok"]),
        "azure_csv_export_ok": float(checks["azure_csv_export_ok"]),
        "azure_parquet_export_ok": float(checks["azure_parquet_export_ok"]),
        "azure_metadata_yaml_ok": float(checks["azure_metadata_yaml_ok"]),
        "azure_dp_sql_smoke_test_ok": float(checks["azure_dp_sql_smoke_test_ok"]),
    }
