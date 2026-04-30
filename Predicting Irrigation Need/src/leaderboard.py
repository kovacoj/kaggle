from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from benchmark import PROJECT_ROOT, score_prediction_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank benchmark prediction files from the artifacts directory.")
    parser.add_argument("--benchmark", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--glob",
        default="artifacts/**/*.csv",
        help="Glob pattern, relative to the competition root, used to find prediction CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(PROJECT_ROOT.glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"No prediction files matched {args.glob!r}")

    rows: list[dict[str, object]] = []
    for path in paths:
        try:
            predictions = pl.read_csv(path)
            metrics = score_prediction_frame(predictions, args.benchmark)
        except Exception as exc:
            rows.append(
                {
                    "file": str(path.relative_to(PROJECT_ROOT)),
                    "status": f"error: {exc}",
                }
            )
            continue

        rows.append(
            {
                "file": str(path.relative_to(PROJECT_ROOT)),
                "status": "ok",
                **metrics,
            }
        )

    leaderboard = pl.DataFrame(rows)
    if "balanced_accuracy" in leaderboard.columns:
        leaderboard = leaderboard.with_columns(
            pl.col("balanced_accuracy").round(6),
            pl.col("recall_low").round(6),
            pl.col("recall_medium").round(6),
            pl.col("recall_high").round(6),
        ).sort(["status", "balanced_accuracy"], descending=[False, True], nulls_last=True)

    print(leaderboard)


if __name__ == "__main__":
    main()
