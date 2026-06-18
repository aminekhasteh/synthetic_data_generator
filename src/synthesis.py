"""SmartNoise synthesizer wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
from snsynth import Synthesizer
from snsynth.transform import BinTransformer, LabelTransformer, TableTransformer

from .data import DEFAULT_TARGET_COLUMN

if TYPE_CHECKING:
    from .config import DatasetConfig


@dataclass
class SynthesisResult:
    synthetic: pd.DataFrame
    synthesizer_name: str
    epsilon: float
    delta: float
    spent_epsilon: float
    spent_delta: float


def _resolve_config(config: DatasetConfig | None, seed_df: pd.DataFrame):
    if config is None:
        target = DEFAULT_TARGET_COLUMN if DEFAULT_TARGET_COLUMN in seed_df.columns else None
        categorical = [target] if target else []
        continuous = [c for c in seed_df.columns if c not in categorical]
        return target, categorical, continuous
    config.infer_column_types(seed_df)
    target = config.target_column
    categorical = list(config.categorical_columns)
    continuous = [c for c in seed_df.columns if c not in categorical]
    return target, categorical, continuous


def _column_hints(
    seed_df: pd.DataFrame,
    config: DatasetConfig | None = None,
) -> dict:
    target, categorical, continuous = _resolve_config(config, seed_df)
    hints = {
        "continuous_columns": [c for c in continuous if c in seed_df.columns],
        "categorical_columns": [c for c in categorical if c in seed_df.columns],
    }
    if not hints["categorical_columns"] and target and target in seed_df.columns:
        hints["categorical_columns"] = [target]
        hints["continuous_columns"] = [
            c for c in hints["continuous_columns"] if c != target
        ]
    return hints


def _column_bounds(df: pd.DataFrame, col: str) -> tuple[float, float]:
    lower = float(df[col].min())
    upper = float(df[col].max())
    if lower == upper:
        upper = lower + 1.0
    span = upper - lower
    pad = max(span * 0.05, 1e-3)
    return lower - pad, upper + pad


def build_table_transformer(
    df: pd.DataFrame,
    config: DatasetConfig | None = None,
    bins: int = 10,
) -> TableTransformer:
    """Build a reversible transformer with explicit public bounds (no DP budget for bounds)."""
    _, categorical, _ = _resolve_config(config, df)
    categorical_set = set(categorical)
    transformers = []
    for col in df.columns:
        if col in categorical_set:
            transformers.append(LabelTransformer())
        else:
            lower, upper = _column_bounds(df, col)
            transformers.append(BinTransformer(bins=bins, lower=lower, upper=upper))
    return TableTransformer(transformers)


def _read_odometer(synth) -> tuple[float, float]:
    if hasattr(synth, "spent"):
        spent_val = synth.spent
        if spent_val is not None:
            return float(spent_val), 0.0
    if hasattr(synth, "accountant") and synth.accountant:
        return float(sum(synth.accountant)), 0.0
    spent = getattr(synth, "odometer", None)
    if spent is None:
        return 0.0, 0.0
    value = spent.spent
    if isinstance(value, tuple):
        return float(value[0]), float(value[1]) if len(value) > 1 else 0.0
    return float(value), 0.0


def _postprocess_synthetic(
    df: pd.DataFrame,
    target_column: str | None,
) -> pd.DataFrame:
    out = df.copy()
    if target_column and target_column in out.columns:
        series = out[target_column]
        if pd.api.types.is_numeric_dtype(series):
            nunique = series.nunique(dropna=True)
            if nunique <= 2:
                out[target_column] = (
                    pd.to_numeric(series, errors="coerce").round().astype(int)
                )
    return out


def fit_sample_smartnoise(
    seed_df: pd.DataFrame,
    synthesizer_name: str = "mwem",
    epsilon: float = 1.0,
    preprocessor_eps: float = 0.0,
    random_state: int = 42,
    verbose: bool = True,
    config: DatasetConfig | None = None,
    **kwargs,
) -> SynthesisResult:
    """Fit SmartNoise synthesizer and generate synthetic data."""
    hints = _column_hints(seed_df, config)
    target, _, _ = _resolve_config(config, seed_df)
    n_rows = len(seed_df)
    transformer = build_table_transformer(seed_df, config=config)

    create_kwargs = {"epsilon": epsilon, "verbose": verbose}
    if synthesizer_name == "mwem":
        create_kwargs["split_factor"] = kwargs.pop("split_factor", 1)
        create_kwargs["iterations"] = kwargs.pop("iterations", 10)
    if synthesizer_name == "dpctgan":
        create_kwargs.update(
            {
                "batch_size": min(500, max(64, n_rows // 4)),
                "epochs": kwargs.pop("epochs", 50),
            }
        )
    create_kwargs.update(kwargs)

    synth = Synthesizer.create(synthesizer_name, **create_kwargs)
    synthetic = synth.fit_sample(
        seed_df,
        transformer=transformer,
        preprocessor_eps=preprocessor_eps,
        **hints,
    )

    if len(synthetic) != n_rows:
        synthetic = synth.sample(n_rows)

    if isinstance(synthetic, pd.DataFrame):
        synthetic = synthetic[seed_df.columns]
    else:
        synthetic = pd.DataFrame(synthetic, columns=seed_df.columns)

    synthetic = _postprocess_synthetic(synthetic, target)

    spent_eps, spent_delta = _read_odometer(synth)

    return SynthesisResult(
        synthetic=synthetic.reset_index(drop=True),
        synthesizer_name=synthesizer_name,
        epsilon=epsilon,
        delta=create_kwargs.get("delta", 0.0) or 0.0,
        spent_epsilon=spent_eps,
        spent_delta=spent_delta,
    )


def privacy_budget_dict(result: SynthesisResult) -> dict:
    """Return privacy budget as a flat dict for metrics tables."""
    return {
        "privacy_epsilon_configured": result.epsilon,
        "privacy_delta_configured": result.delta,
        "privacy_epsilon_spent": result.spent_epsilon,
        "privacy_delta_spent": result.spent_delta,
        "synthesizer": result.synthesizer_name,
    }
