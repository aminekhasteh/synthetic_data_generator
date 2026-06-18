#!/usr/bin/env python3
"""CLI for SmartNoise synthetic data generation on arbitrary seed datasets."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_dataset_config
from src.pipeline import RunConfig, run_pipeline


def _str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "t", "yes", "y", "1"}:
        return True
    if lowered in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got: {value}")


def _parse_columns(value: str) -> list[str]:
    return [c.strip() for c in value.split(",") if c.strip()]


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate differentially private synthetic data with OpenDP SmartNoise "
            "and evaluate statistical fidelity, temporal, privacy, Azure compatibility, "
            "and utility metrics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seed-dataset",
        required=True,
        type=Path,
        help="Path to the seed dataset (CSV or Parquet).",
    )
    parser.add_argument(
        "--columns",
        required=False,
        default=None,
        type=_parse_columns,
        help=(
            "Comma-separated columns to use. Optional if provided in the data dictionary "
            "or when inferring from the dataset (all non-ID columns)."
        ),
    )
    parser.add_argument(
        "--data-dictionary",
        type=Path,
        default=None,
        help="Optional JSON/YAML data dictionary with target/temporal/column types.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name for output folder (defaults to seed file stem).",
    )
    parser.add_argument(
        "--sampling",
        type=_str2bool,
        default=True,
        help="Fast run (true): smaller samples and fewer bootstrap reps. Slow run (false): larger samples and full bootstrap.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Differential privacy epsilon budget.",
    )
    parser.add_argument(
        "--synthesizer",
        type=str,
        default="mwem",
        choices=["mwem", "dpctgan", "mst", "aim"],
        help="SmartNoise synthesizer to use.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Base directory for run outputs.",
    )
    parser.add_argument(
        "--seed-size",
        type=int,
        default=None,
        help="Override seed sample size.",
    )
    parser.add_argument(
        "--holdout-size",
        type=int,
        default=None,
        help="Override holdout size for utility evaluation.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=None,
        help="Override bootstrap pool size.",
    )

    bootstrap = parser.add_argument_group("bootstrap", "Bootstrap uncertainty estimation")
    bootstrap.add_argument(
        "--bootstrap",
        type=_str2bool,
        default=True,
        help="Enable bootstrap replicates (false skips bootstrap entirely).",
    )
    bootstrap.add_argument(
        "--bootstrap-n",
        type=int,
        default=None,
        help="Number of bootstrap replicates (default: 5 if --sampling true, 30 if false).",
    )
    bootstrap.add_argument(
        "--bootstrap-sample-size",
        type=int,
        default=None,
        help="Rows per bootstrap resample (default: same as --seed-size).",
    )
    bootstrap.add_argument(
        "--bootstrap-seed",
        type=int,
        default=None,
        help="Random seed for bootstrap resampling (default: --random-state).",
    )
    bootstrap.add_argument(
        "--bootstrap-ci-low",
        type=float,
        default=0.025,
        help="Lower quantile for bootstrap confidence intervals.",
    )
    bootstrap.add_argument(
        "--bootstrap-ci-high",
        type=float,
        default=0.975,
        help="Upper quantile for bootstrap confidence intervals.",
    )
    bootstrap.add_argument(
        "--bootstrap-min-class-count",
        type=int,
        default=1,
        help="Minimum positive-class rows in stratified bootstrap resamples.",
    )
    bootstrap.add_argument(
        "--bootstrap-save-replicates",
        type=_str2bool,
        default=True,
        help="Save per-replicate outputs under bootstrap_*/ subfolders.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling and synthesis.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    log = logging.getLogger("run_synthetic_pipeline")

    if not args.seed_dataset.exists():
        log.error("Seed dataset not found: %s", args.seed_dataset)
        return 1
    if args.data_dictionary and not args.data_dictionary.exists():
        log.error("Data dictionary not found: %s", args.data_dictionary)
        return 1

    config = None
    if args.columns or args.data_dictionary:
        config = load_dataset_config(
            seed_path=args.seed_dataset,
            columns=args.columns,
            dictionary_path=args.data_dictionary,
            dataset_name=args.dataset_name,
        )

    run_kwargs = {
        "epsilon": args.epsilon,
        "synthesizer": args.synthesizer,
        "random_state": args.random_state,
        "output_base": args.output_dir,
    }
    if args.seed_size is not None:
        run_kwargs["seed_size"] = args.seed_size
    if args.holdout_size is not None:
        run_kwargs["holdout_size"] = args.holdout_size
    if args.pool_size is not None:
        run_kwargs["pool_size"] = args.pool_size
    run_kwargs["bootstrap"] = args.bootstrap
    if args.bootstrap_n is not None:
        run_kwargs["bootstrap_n"] = args.bootstrap_n
    if args.bootstrap_sample_size is not None:
        run_kwargs["bootstrap_sample_size"] = args.bootstrap_sample_size
    if args.bootstrap_seed is not None:
        run_kwargs["bootstrap_seed"] = args.bootstrap_seed
    run_kwargs["bootstrap_ci_low"] = args.bootstrap_ci_low
    run_kwargs["bootstrap_ci_high"] = args.bootstrap_ci_high
    run_kwargs["bootstrap_min_class_count"] = args.bootstrap_min_class_count
    run_kwargs["bootstrap_save_replicates"] = args.bootstrap_save_replicates

    if args.bootstrap_ci_low >= args.bootstrap_ci_high:
        log.error(
            "bootstrap-ci-low (%.4f) must be less than bootstrap-ci-high (%.4f)",
            args.bootstrap_ci_low,
            args.bootstrap_ci_high,
        )
        return 1

    run = RunConfig.from_sampling_flag(args.sampling, **run_kwargs)

    log.info("Run mode: %s", "FAST (sampling=true)" if run.sampling else "FULL (sampling=false)")
    log.info(
        "Sizes: seed=%s holdout=%s pool=%s",
        run.seed_size,
        run.holdout_size,
        run.pool_size,
    )
    log.info(
        "Bootstrap: enabled=%s n=%s sample_size=%s seed=%s ci=[%.3f, %.3f] min_class=%s save_replicates=%s",
        run.bootstrap,
        run.effective_bootstrap_n,
        run.effective_bootstrap_sample_size,
        run.effective_bootstrap_seed,
        run.bootstrap_ci_low,
        run.bootstrap_ci_high,
        run.bootstrap_min_class_count,
        run.bootstrap_save_replicates,
    )

    try:
        output_dir = run_pipeline(
            seed_path=args.seed_dataset,
            config=config,
            run=run,
            dictionary_path=args.data_dictionary,
            columns=args.columns,
            dataset_name=args.dataset_name,
        )
    except Exception:
        log.exception("Pipeline failed")
        return 1

    log.info("Done. Results: %s", output_dir)
    log.info("Seed data: %s", output_dir / "seed_data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
