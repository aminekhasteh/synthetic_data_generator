"""End-to-end synthetic data pipeline for arbitrary datasets."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config import DatasetConfig, load_dictionary_file
from .data import (
    aggregate_bootstrap_metrics,
    load_and_prepare_dataset,
    random_bootstrap,
    random_sample,
    save_data_profile,
    stratified_bootstrap,
    stratified_sample,
)
from .metrics import compute_all_metrics
from .synthesis import fit_sample_smartnoise, privacy_budget_dict

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Runtime parameters for a pipeline execution."""

    sampling: bool = True
    seed_size: int = 2000
    holdout_size: int = 5000
    pool_size: int = 10000
    bootstrap: bool = True
    bootstrap_n: int = 30
    bootstrap_sample_size: int | None = None
    bootstrap_seed: int | None = None
    bootstrap_ci_low: float = 0.025
    bootstrap_ci_high: float = 0.975
    bootstrap_min_class_count: int = 1
    bootstrap_save_replicates: bool = True
    epsilon: float = 1.0
    synthesizer: str = "mwem"
    random_state: int = 42
    output_base: Path = Path("outputs")

    _RESERVED = frozenset(
        {
            "seed_size",
            "holdout_size",
            "pool_size",
            "bootstrap",
            "bootstrap_n",
            "bootstrap_sample_size",
            "bootstrap_seed",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "bootstrap_min_class_count",
            "bootstrap_save_replicates",
        }
    )

    @property
    def effective_bootstrap_n(self) -> int:
        return self.bootstrap_n if self.bootstrap else 0

    @property
    def effective_bootstrap_sample_size(self) -> int:
        return self.bootstrap_sample_size or self.seed_size

    @property
    def effective_bootstrap_seed(self) -> int:
        return self.bootstrap_seed if self.bootstrap_seed is not None else self.random_state

    @classmethod
    def from_sampling_flag(cls, sampling: bool, **kwargs) -> RunConfig:
        bootstrap = kwargs.get("bootstrap", True)
        defaults = {
            True: dict(seed_size=2000, holdout_size=5000, pool_size=10000, bootstrap_n=5),
            False: dict(seed_size=10000, holdout_size=20000, pool_size=50000, bootstrap_n=30),
        }[sampling]
        passthrough = {k: v for k, v in kwargs.items() if k not in cls._RESERVED}
        return cls(
            sampling=sampling,
            seed_size=kwargs.get("seed_size", defaults["seed_size"]),
            holdout_size=kwargs.get("holdout_size", defaults["holdout_size"]),
            pool_size=kwargs.get("pool_size", defaults["pool_size"]),
            bootstrap=bootstrap,
            bootstrap_n=kwargs.get("bootstrap_n", defaults["bootstrap_n"]),
            bootstrap_sample_size=kwargs.get("bootstrap_sample_size"),
            bootstrap_seed=kwargs.get("bootstrap_seed"),
            bootstrap_ci_low=kwargs.get("bootstrap_ci_low", 0.025),
            bootstrap_ci_high=kwargs.get("bootstrap_ci_high", 0.975),
            bootstrap_min_class_count=kwargs.get("bootstrap_min_class_count", 1),
            bootstrap_save_replicates=kwargs.get("bootstrap_save_replicates", True),
            **passthrough,
        )


def build_output_dir(config: DatasetConfig, run: RunConfig) -> Path:
    """Create output directory: {dataset}_{timestamp}_sampling-{bool}_eps-{e}_synth-{name}."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sampling_tag = "true" if run.sampling else "false"
    bootstrap_tag = (
        f"boot-{run.effective_bootstrap_n}"
        if run.bootstrap
        else "boot-off"
    )
    folder = (
        f"{config.name}_{timestamp}_"
        f"sampling-{sampling_tag}_"
        f"{bootstrap_tag}_"
        f"eps-{run.epsilon}_"
        f"synth-{run.synthesizer}"
    )
    out = run.output_base / folder
    out.mkdir(parents=True, exist_ok=True)
    return out


def _sample_fn(config: DatasetConfig):
    target = config.target_column
    if target:
        return stratified_sample, stratified_bootstrap
    return random_sample, random_bootstrap


def prepare_splits(
    df: pd.DataFrame,
    config: DatasetConfig,
    run: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create seed, holdout, and bootstrap pool from the full seed dataframe."""
    sample_fn, _ = _sample_fn(config)
    kwargs = {"random_state": run.random_state}
    if config.target_column:
        kwargs["target_column"] = config.target_column

    total_needed = run.pool_size + run.holdout_size
    if total_needed > len(df):
        run.pool_size = min(run.pool_size, max(len(df) // 2, 1))
        run.holdout_size = min(run.holdout_size, len(df) - run.pool_size)
        logger.warning(
            "Adjusted pool_size=%s holdout_size=%s to fit dataset rows=%s",
            run.pool_size,
            run.holdout_size,
            len(df),
        )

    from sklearn.model_selection import train_test_split

    pool_and_holdout = sample_fn(df, n=run.pool_size + run.holdout_size, **kwargs)
    stratify = (
        pool_and_holdout[config.target_column]
        if config.target_column
        else None
    )
    pool_df, holdout_df = train_test_split(
        pool_and_holdout,
        test_size=run.holdout_size,
        stratify=stratify,
        random_state=run.random_state,
    )
    pool_df = pool_df.reset_index(drop=True)
    holdout_df = holdout_df.reset_index(drop=True)
    seed_df = sample_fn(pool_df, n=min(run.seed_size, len(pool_df)), **kwargs)
    return seed_df, holdout_df, pool_df


def save_seed_artifacts(
    output_dir: Path,
    seed_df: pd.DataFrame,
    source_path: Path,
    config: DatasetConfig,
    run: RunConfig,
    dictionary_path: Path | None,
    data_profile: dict | None = None,
) -> Path:
    """Persist seed data and run metadata under seed_data/."""
    seed_dir = output_dir / "seed_data"
    seed_dir.mkdir(parents=True, exist_ok=True)

    seed_df.to_csv(seed_dir / "seed.csv", index=False)
    shutil.copy2(source_path, seed_dir / f"source{source_path.suffix}")

    if dictionary_path is not None:
        shutil.copy2(dictionary_path, seed_dir / f"dictionary{dictionary_path.suffix}")

    manifest = {
        "dataset_name": config.name,
        "source_path": str(source_path.resolve()),
        "columns": config.columns,
        "target_column": config.target_column,
        "temporal_column": config.temporal_column,
        "categorical_columns": config.categorical_columns,
        "continuous_columns": config.continuous_columns,
        "sampling": run.sampling,
        "seed_size": run.seed_size,
        "holdout_size": run.holdout_size,
        "pool_size": run.pool_size,
        "bootstrap": run.bootstrap,
        "bootstrap_n": run.bootstrap_n,
        "bootstrap_sample_size": run.effective_bootstrap_sample_size,
        "bootstrap_seed": run.effective_bootstrap_seed,
        "bootstrap_ci_low": run.bootstrap_ci_low,
        "bootstrap_ci_high": run.bootstrap_ci_high,
        "bootstrap_min_class_count": run.bootstrap_min_class_count,
        "bootstrap_save_replicates": run.bootstrap_save_replicates,
        "epsilon": run.epsilon,
        "synthesizer": run.synthesizer,
        "random_state": run.random_state,
        "seed_rows": len(seed_df),
    }
    with open(seed_dir / "run_config.json", "w") as f:
        json.dump(manifest, f, indent=2)

    with open(output_dir / "dataset_config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    if data_profile is not None:
        save_data_profile(data_profile, seed_dir / "data_profile.json")

    logger.info("Saved seed data to %s", seed_dir)
    return seed_dir


def run_pipeline(
    seed_path: Path,
    config: DatasetConfig | None,
    run: RunConfig,
    dictionary_path: Path | None = None,
    columns: list[str] | None = None,
    dataset_name: str | None = None,
) -> Path:
    """Execute synthesis, metrics, and optional bootstrap; return output directory."""
    dictionary_meta = load_dictionary_file(dictionary_path) if dictionary_path else None

    df, config, profile = load_and_prepare_dataset(
        seed_path=seed_path,
        config=config if config and config.columns else None,
        dictionary_meta=dictionary_meta,
        columns=columns,
        dataset_name=dataset_name or (config.name if config else None),
    )

    output_dir = build_output_dir(config, run)
    logger.info("Output directory: %s", output_dir)
    logger.info(
        "Loaded %s rows x %s columns (target=%s, temporal=%s, inferred=%s)",
        len(df),
        len(config.columns),
        config.target_column,
        config.temporal_column,
        config.inferred,
    )

    seed_df, holdout_df, pool_df = prepare_splits(df, config, run)
    save_seed_artifacts(
        output_dir, seed_df, seed_path, config, run, dictionary_path, data_profile=profile
    )

    logger.info("Fitting synthesizer (%s, epsilon=%s)...", run.synthesizer, run.epsilon)
    result = fit_sample_smartnoise(
        seed_df,
        config=config,
        synthesizer_name=run.synthesizer,
        epsilon=run.epsilon,
        verbose=False,
    )
    synthetic_df = result.synthetic
    synthetic_df.to_csv(output_dir / "synthetic.csv", index=False)
    logger.info(
        "Synthesis complete: %s rows, epsilon spent=%.4f",
        len(synthetic_df),
        result.spent_epsilon,
    )

    privacy = privacy_budget_dict(result)
    metrics = compute_all_metrics(
        real_train=seed_df,
        synthetic=synthetic_df,
        real_holdout=holdout_df,
        privacy_budget=privacy,
        output_dir=str(output_dir),
        synthesizer_name=run.synthesizer,
        config=config,
    )
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Single-run metrics saved")

    if run.effective_bootstrap_n > 0:
        logger.info(
            "Starting bootstrap: n=%s sample_size=%s seed=%s ci=[%.3f, %.3f]",
            run.effective_bootstrap_n,
            run.effective_bootstrap_sample_size,
            run.effective_bootstrap_seed,
            run.bootstrap_ci_low,
            run.bootstrap_ci_high,
        )
        _, bootstrap_fn = _sample_fn(config)
        bootstrap_metrics = []

        for b in tqdm(range(run.effective_bootstrap_n), desc="Bootstrap", unit="rep"):
            boot_kwargs = {"random_state": run.effective_bootstrap_seed + b}
            if config.target_column:
                boot_kwargs["target_column"] = config.target_column
                boot_kwargs["min_fraud_count"] = run.bootstrap_min_class_count
            boot_seed = bootstrap_fn(
                pool_df,
                n=min(run.effective_bootstrap_sample_size, len(pool_df)),
                **boot_kwargs,
            )
            boot_result = fit_sample_smartnoise(
                boot_seed,
                config=config,
                synthesizer_name=run.synthesizer,
                epsilon=run.epsilon,
                verbose=False,
            )
            replicate_dir = (
                str(output_dir / f"bootstrap_{b}")
                if run.bootstrap_save_replicates
                else str(output_dir)
            )
            boot_metrics = compute_all_metrics(
                real_train=boot_seed,
                synthetic=boot_result.synthetic,
                real_holdout=holdout_df,
                privacy_budget=privacy_budget_dict(boot_result),
                output_dir=replicate_dir,
                synthesizer_name=run.synthesizer,
                config=config,
                write_artifacts=run.bootstrap_save_replicates,
            )
            boot_metrics["bootstrap_id"] = b
            bootstrap_metrics.append(boot_metrics)

        summary = aggregate_bootstrap_metrics(
            bootstrap_metrics,
            ci_low=run.bootstrap_ci_low,
            ci_high=run.bootstrap_ci_high,
        )
        summary.to_csv(output_dir / "bootstrap_metrics.csv", index=False)
        with open(output_dir / "bootstrap_replicates.json", "w") as f:
            json.dump(bootstrap_metrics, f, indent=2, default=str)
        logger.info("Bootstrap summary saved (%s replicates)", run.effective_bootstrap_n)
    elif not run.bootstrap:
        logger.info("Bootstrap disabled (--bootstrap false)")

    logger.info("Pipeline finished: %s", output_dir)
    return output_dir
