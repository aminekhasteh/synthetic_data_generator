"""Metric modules for synthetic data evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .fidelity import compute_fidelity_metrics
from .temporal import compute_temporal_metrics
from .privacy import compute_privacy_metrics
from .utility import compute_utility_metrics
from .azure_compat import run_azure_compat_checklist

if TYPE_CHECKING:
    from ..config import DatasetConfig

__all__ = [
    "compute_fidelity_metrics",
    "compute_temporal_metrics",
    "compute_privacy_metrics",
    "compute_utility_metrics",
    "run_azure_compat_checklist",
    "compute_all_metrics",
]


def compute_all_metrics(
    real_train,
    synthetic,
    real_holdout,
    privacy_budget,
    output_dir="outputs",
    synthesizer_name="mwem",
    config: DatasetConfig | None = None,
    write_artifacts: bool = True,
):
    """Run all metric dimensions and return a flat dict."""
    from pathlib import Path

    out = Path(output_dir)
    if write_artifacts:
        out.mkdir(parents=True, exist_ok=True)

    metrics = {}
    metrics.update(compute_fidelity_metrics(real_train, synthetic, config=config))
    metrics.update(compute_temporal_metrics(real_train, synthetic, config=config))
    metrics.update(
        compute_privacy_metrics(
            real_train, synthetic, real_holdout, privacy_budget=privacy_budget
        )
    )
    metrics.update(
        compute_utility_metrics(real_train, synthetic, real_holdout, config=config)
    )
    if write_artifacts:
        metrics.update(
            run_azure_compat_checklist(
                synthetic,
                privacy_budget=privacy_budget,
                output_dir=output_dir,
                synthesizer_name=synthesizer_name,
                config=config,
            )
        )
    return metrics
