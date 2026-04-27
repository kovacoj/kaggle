from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import polars as pl

import experiment
from benchmark import (
    BENCHMARK_NAMES,
    DATA_DIR,
    ID_COLUMN,
    PROJECT_ROOT,
    TARGET,
    ensure_benchmark_exists,
    load_benchmark_part,
    load_train_frame,
    score_prediction_frame,
)

METRIC_NAME = "balanced_accuracy"
METRIC_DIRECTION = "maximize"
RESULTS_PATH = PROJECT_ROOT / "results.tsv"
RESULTS_HEADER = (
    "run_id\tcommit\tbenchmark\tapproach\tmetric_name\tmetric_direction\tmetric_value\t"
    "runtime_seconds\tstatus\tdescription\tsnapshot\n"
)
EXPERIMENT_RUNS_DIR = PROJECT_ROOT / "artifacts" / "experiment_runs"
HISTORY_DIR = PROJECT_ROOT / "history"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed irrigation benchmark harness against src/experiment.py.")
    parser.add_argument("--benchmark", choices=BENCHMARK_NAMES, default="smoke")
    parser.add_argument("--write-submission", action="store_true", help="Also fit on the full training set and write a Kaggle submission.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_benchmark_exists()

    description = os.environ.get("EXPERIMENT_DESCRIPTION", "").strip() or experiment.DESCRIPTION
    commit = git_short_commit(PROJECT_ROOT)
    working_tree_dirty = git_is_dirty(PROJECT_ROOT)
    run_id = build_run_id(commit, working_tree_dirty)

    train_frame = load_benchmark_part(args.benchmark, "train")
    valid_frame = load_benchmark_part(args.benchmark, "valid")

    started_at = time.time()
    validation_predictions = experiment.fit_predict_valid(train_frame, valid_frame, args.benchmark)
    validate_prediction_contract(validation_predictions)

    experiment_runs_dir = EXPERIMENT_RUNS_DIR
    experiment_runs_dir.mkdir(parents=True, exist_ok=True)
    validation_path = experiment_runs_dir / f"{run_id}-{args.benchmark}.csv"
    validation_predictions.write_csv(validation_path)

    metrics = score_prediction_frame(validation_predictions, args.benchmark)
    metric_value = float(metrics[METRIC_NAME])

    submission_file = "-"
    if args.write_submission:
        submission_file = write_submission(run_id)

    runtime_seconds = time.time() - started_at

    summary = {
        "run_id": run_id,
        "commit": commit,
        "working_tree_dirty": working_tree_dirty,
        "benchmark": args.benchmark,
        "approach": experiment.APPROACH,
        "metric_name": METRIC_NAME,
        "metric_direction": METRIC_DIRECTION,
        "metric_value": metric_value,
        "runtime_seconds": runtime_seconds,
        "metrics": metrics,
        "experiment": experiment.DESCRIPTION,
        "description": description,
        "validation_predictions": str(validation_path.relative_to(PROJECT_ROOT)),
        "submission_file": submission_file,
    }
    snapshot_dir = archive_run(run_id, summary, validation_path, submission_file)
    append_result(
        run_id=run_id,
        commit=commit,
        benchmark=args.benchmark,
        approach=experiment.APPROACH,
        metric_value=metric_value,
        runtime_seconds=runtime_seconds,
        status="ran",
        description=description,
        snapshot=str(snapshot_dir.relative_to(PROJECT_ROOT)),
    )

    print("---")
    print(f"run_id:                 {run_id}")
    print(f"benchmark:              {args.benchmark}")
    print(f"metric_name:            {METRIC_NAME}")
    print(f"metric_direction:       {METRIC_DIRECTION}")
    print(f"metric_value:           {metric_value:.6f}")
    print(f"runtime_seconds:        {runtime_seconds:.1f}")
    print(f"validation_predictions: {validation_path.relative_to(PROJECT_ROOT)}")
    print(f"submission_file:        {submission_file}")
    print(f"experiment:             {experiment.DESCRIPTION}")
    print(f"description:            {description}")
    print(f"snapshot:               {snapshot_dir.relative_to(PROJECT_ROOT)}")


def validate_prediction_contract(predictions: pl.DataFrame) -> None:
    if predictions.columns != [ID_COLUMN, TARGET]:
        raise ValueError(f"Predictions must have columns exactly [{ID_COLUMN}, {TARGET}].")


def write_submission(run_id: str) -> str:
    train_frame = load_train_frame()
    test_frame = pl.read_csv(DATA_DIR / "test.csv")
    sample_submission = pl.read_csv(DATA_DIR / "sample_submission.csv")
    submission = experiment.fit_predict_test(train_frame, test_frame)
    validate_prediction_contract(submission)

    if not submission[ID_COLUMN].equals(test_frame[ID_COLUMN]):
        raise ValueError("Submission predictions ids do not match test.csv ids.")

    submission = submission.select(sample_submission.columns)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SUBMISSIONS_DIR / f"submission-{run_id}.csv"
    submission.write_csv(output_path)
    return str(output_path.relative_to(PROJECT_ROOT))


def git_short_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=project_root,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "no-git"

    return result.stdout.strip() or "no-git"


def git_is_dirty(project_root: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
            cwd=project_root,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

    return bool(result.stdout.strip())


def build_run_id(commit: str, dirty: bool) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = "dirty" if dirty else commit
    return f"{timestamp}-{suffix}"


def archive_run(run_id: str, summary: dict[str, object], validation_path: Path, submission_file: str) -> Path:
    run_dir = HISTORY_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    files_to_copy = [
        PROJECT_ROOT / "src" / "experiment.py",
        validation_path,
        PROJECT_ROOT / "artifacts" / "data_profile.md",
    ]
    if submission_file != "-":
        files_to_copy.append(PROJECT_ROOT / submission_file)

    for path in files_to_copy:
        if path.exists():
            shutil.copy2(path, run_dir / path.name)

    summary_path = run_dir / "summary.json"
    summary_with_snapshot = {**summary, "snapshot": str(run_dir.relative_to(PROJECT_ROOT))}
    summary_path.write_text(json.dumps(summary_with_snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return run_dir


def append_result(
    run_id: str,
    commit: str,
    benchmark: str,
    approach: str,
    metric_value: float,
    runtime_seconds: float,
    status: str,
    description: str,
    snapshot: str,
) -> None:
    ensure_results_file()
    row = "\t".join(
        [
            run_id,
            commit,
            benchmark,
            approach,
            METRIC_NAME,
            METRIC_DIRECTION,
            f"{metric_value:.6f}",
            f"{runtime_seconds:.1f}",
            status,
            description.replace("\t", " ").strip(),
            snapshot,
        ]
    )
    with RESULTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{row}\n")


def ensure_results_file() -> None:
    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text(RESULTS_HEADER, encoding="utf-8")


if __name__ == "__main__":
    main()
