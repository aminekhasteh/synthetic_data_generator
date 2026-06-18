#!/usr/bin/env python3
"""Run the synthetic data pipeline on all example datasets."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
EXAMPLES = PROJECT_ROOT / "examples"

# (csv_file, dictionary_file, extra CLI args for sizing)
EXAMPLE_RUNS = [
    (
        "club_financials.csv",
        "club_financials_dictionary.yaml",
        ["--seed-size", "300", "--holdout-size", "150", "--pool-size", "400", "--bootstrap-n", "3"],
    ),
    (
        "league_metrics.csv",
        "league_metrics_dictionary.yaml",
        ["--seed-size", "100", "--holdout-size", "40", "--pool-size", "120", "--bootstrap-n", "3"],
    ),
    (
        "player_market_values.csv",
        "player_market_values_dictionary.yaml",
        ["--seed-size", "400", "--holdout-size", "200", "--pool-size", "600", "--bootstrap-n", "3"],
    ),
    (
        "record_transfers.csv",
        "record_transfers_dictionary.yaml",
        ["--seed-size", "40", "--holdout-size", "10", "--pool-size", "45", "--bootstrap-n", "2"],
    ),
    (
        "transfers_history.csv",
        "transfers_history_dictionary.yaml",
        [
            "--seed-size", "500",
            "--holdout-size", "1000",
            "--pool-size", "2000",
            "--bootstrap-n", "2",
            "--bootstrap-save-replicates", "false",
        ],
    ),
    (
        "tiny_creditcard.csv",
        "creditcard_dictionary.yaml",
        [
            "--columns",
            "Time,Amount,V1,V2,V3,V14,V17,Class",
            "--seed-size", "200",
            "--holdout-size", "150",
            "--pool-size", "300",
            "--bootstrap-n", "2",
        ],
    ),
]


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def run_one(
    csv_name: str,
    dict_name: str,
    extra_args: list[str],
    sampling: bool,
    output_dir: Path,
    verbose: bool,
) -> tuple[str, int]:
    csv_path = EXAMPLES / csv_name
    dict_path = EXAMPLES / dict_name
    log = logging.getLogger("run_examples")

    if not csv_path.exists():
        log.error("Missing dataset: %s", csv_path)
        return csv_name, 1
    if not dict_path.exists():
        log.error("Missing dictionary: %s", dict_path)
        return csv_name, 1

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "run_synthetic_pipeline.py"),
        "--seed-dataset",
        str(csv_path),
        "--data-dictionary",
        str(dict_path),
        "--sampling",
        "true" if sampling else "false",
        "--output-dir",
        str(output_dir),
        "--bootstrap",
        "true",
        *extra_args,
    ]
    if verbose:
        cmd.append("--verbose")

    log.info("Running: %s", csv_name)
    log.debug("Command: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    status = "OK" if result.returncode == 0 else "FAILED"
    log.info("%s: %s (exit %s)", csv_name, status, result.returncode)
    return csv_name, result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Test pipeline on all example datasets.")
    parser.add_argument(
        "--dataset",
        action="append",
        help="Run only named CSV(s), e.g. club_financials.csv (repeatable).",
    )
    parser.add_argument(
        "--sampling",
        choices=["true", "false"],
        default="true",
        help="Fast or full pipeline mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "example_runs",
        help="Base output directory for all runs.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    runs = EXAMPLE_RUNS
    if args.dataset:
        names = set(args.dataset)
        runs = [r for r in EXAMPLE_RUNS if r[0] in names]

    results = []
    for csv_name, dict_name, extra in runs:
        name, code = run_one(
            csv_name,
            dict_name,
            extra,
            sampling=args.sampling == "true",
            output_dir=args.output_dir,
            verbose=args.verbose,
        )
        results.append((name, code))

    print("\n" + "=" * 60)
    print("EXAMPLE DATASET TEST SUMMARY")
    print("=" * 60)
    failed = 0
    for name, code in results:
        status = "PASS" if code == 0 else "FAIL"
        print(f"  [{status}] {name}")
        if code != 0:
            failed += 1
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
