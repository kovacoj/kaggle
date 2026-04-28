from __future__ import annotations

import argparse

import polars as pl

from benchmark import PROJECT_ROOT

RESULTS_PATH = PROJECT_ROOT / "results.tsv"
VALID_STATUSES = ("ran", "keep", "discard", "crash")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update the experiment results ledger.")
    parser.add_argument("run_id", help="Run id to update.")
    parser.add_argument("status", choices=VALID_STATUSES, help="New status value.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing {RESULTS_PATH}")

    results = pl.read_csv(RESULTS_PATH, separator="\t")
    matches = results.filter(pl.col("run_id") == args.run_id).height
    if matches == 0:
        raise ValueError(f"Unknown run_id: {args.run_id}")
    if matches > 1:
        raise ValueError(f"Duplicate run_id entries found for: {args.run_id}")

    updated = results.with_columns(
        pl.when(pl.col("run_id") == args.run_id)
        .then(pl.lit(args.status))
        .otherwise(pl.col("status"))
        .alias("status")
    )
    updated.write_csv(RESULTS_PATH, separator="\t")
    print(f"Updated {args.run_id} to status={args.status}")


if __name__ == "__main__":
    main()
