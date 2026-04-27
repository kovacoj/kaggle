from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import polars as pl

import experiment


# Human-owned competition configuration. Update these once per competition,
# then freeze this file and let the agent iterate only on src/experiment.py.
TASK_TYPE = "classification"
TARGET_COLUMN = "<replace-target-column>"
ID_COLUMN = "id"
METRIC_NAME = "balanced_accuracy"
METRIC_DIRECTION = "maximize"
VALIDATION_FRACTION = 0.2
WRITE_SUBMISSION = True
RESULTS_HEADER = (
    "run_id\tcommit\tmetric_name\tmetric_direction\tmetric_value\t"
    "runtime_seconds\tstatus\tdescription\tsnapshot\n"
)


@dataclass(frozen=True)
class CompetitionConfig:
    task_type: str
    target_column: str
    id_column: str
    metric_name: str
    metric_direction: str
    prediction_column: str
    validation_fraction: float
    write_submission: bool


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    submission_dir = project_root / "submissions"
    history_dir = project_root / "history"
    results_path = project_root / "results.tsv"
    experiment_note = os.environ.get("EXPERIMENT_DESCRIPTION", "").strip()

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_submission_path = data_dir / "sample_submission.csv"

    require_file(train_path)
    require_file(test_path)
    require_file(sample_submission_path)

    train_df = pl.read_csv(train_path)
    test_df = pl.read_csv(test_path)
    sample_submission = pl.read_csv(sample_submission_path)

    prediction_column = sample_submission.columns[1]
    config = CompetitionConfig(
        task_type=TASK_TYPE,
        target_column=TARGET_COLUMN,
        id_column=ID_COLUMN,
        metric_name=METRIC_NAME,
        metric_direction=METRIC_DIRECTION,
        prediction_column=prediction_column,
        validation_fraction=VALIDATION_FRACTION,
        write_submission=WRITE_SUBMISSION,
    )

    validate_config(train_df, test_df, sample_submission, config)

    started_at = time.time()

    fit_df, valid_df = split_train_validation(train_df, config)
    model = experiment.fit_model(fit_df, config)
    valid_predictions = normalize_predictions(experiment.predict(model, valid_df, config), config, valid_df.height)
    metric_value = score_predictions(valid_df[config.target_column].to_list(), valid_predictions.to_list(), config)

    submission_file = "-"
    if config.write_submission:
        full_model = experiment.fit_model(train_df, config)
        test_predictions = normalize_predictions(experiment.predict(full_model, test_df, config), config, test_df.height)
        submission_file = write_submission(test_df, test_predictions, sample_submission, config, submission_dir)

    runtime_seconds = time.time() - started_at

    commit = git_short_commit(project_root)
    working_tree_dirty = git_is_dirty(project_root)
    run_id = build_run_id(commit, working_tree_dirty)

    snapshot = archive_run(
        project_root=project_root,
        history_dir=history_dir,
        run_id=run_id,
        summary={
            "run_id": run_id,
            "commit": commit,
            "working_tree_dirty": working_tree_dirty,
            "metric_name": config.metric_name,
            "metric_direction": config.metric_direction,
            "metric_value": metric_value,
            "runtime_seconds": runtime_seconds,
            "experiment": experiment.DESCRIPTION,
            "description": experiment_note,
            "submission_file": submission_file,
        },
        files_to_copy=[
            project_root / "src" / "experiment.py",
            project_root / "artifacts" / "data_profile.md",
            project_root / submission_file if submission_file != "-" else None,
        ],
    )

    append_result(
        path=results_path,
        run_id=run_id,
        commit=commit,
        metric_name=config.metric_name,
        metric_direction=config.metric_direction,
        metric_value=metric_value,
        runtime_seconds=runtime_seconds,
        status="ran",
        description=experiment_note,
        snapshot=str(snapshot.relative_to(project_root)),
    )

    print("---")
    print(f"run_id:            {run_id}")
    print(f"metric_name:       {config.metric_name}")
    print(f"metric_direction:  {config.metric_direction}")
    print(f"metric_value:      {metric_value:.6f}")
    print(f"runtime_seconds:   {runtime_seconds:.1f}")
    print(f"submission_file:   {submission_file}")
    print(f"experiment:        {experiment.DESCRIPTION}")
    print(f"description:       {experiment_note or '-'}")
    print(f"snapshot:          {snapshot.relative_to(project_root)}")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


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


def archive_run(
    project_root: Path,
    history_dir: Path,
    run_id: str,
    summary: dict[str, object],
    files_to_copy: list[Path | None],
) -> Path:
    run_dir = history_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    for file_path in files_to_copy:
        if file_path is None or not file_path.exists():
            continue
        shutil.copy2(file_path, run_dir / file_path.name)

    summary_path = run_dir / "summary.json"
    summary_with_snapshot = {**summary, "snapshot": str(run_dir.relative_to(project_root))}
    summary_path.write_text(json.dumps(summary_with_snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return run_dir


def append_result(
    path: Path,
    run_id: str,
    commit: str,
    metric_name: str,
    metric_direction: str,
    metric_value: float,
    runtime_seconds: float,
    status: str,
    description: str,
    snapshot: str,
) -> None:
    ensure_results_file(path)
    row = "\t".join(
        [
            run_id,
            commit,
            metric_name,
            metric_direction,
            f"{metric_value:.6f}",
            f"{runtime_seconds:.1f}",
            status,
            description.replace("\t", " ").strip(),
            snapshot,
        ]
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{row}\n")


def ensure_results_file(path: Path) -> None:
    if not path.exists():
        path.write_text(RESULTS_HEADER, encoding="utf-8")


def validate_config(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    sample_submission: pl.DataFrame,
    config: CompetitionConfig,
) -> None:
    if config.target_column.startswith("<replace-"):
        raise ValueError("Update TARGET_COLUMN in src/evaluate.py before running the harness.")

    if config.task_type not in {"classification", "regression"}:
        raise ValueError("TASK_TYPE must be 'classification' or 'regression'.")

    if config.metric_direction not in {"maximize", "minimize"}:
        raise ValueError("METRIC_DIRECTION must be 'maximize' or 'minimize'.")

    if not 0.0 < config.validation_fraction < 1.0:
        raise ValueError("VALIDATION_FRACTION must be between 0 and 1.")

    for name, frame in (("train.csv", train_df), ("test.csv", test_df)):
        if config.id_column not in frame.columns:
            raise ValueError(f"{config.id_column!r} is missing from {name}.")

    if config.target_column not in train_df.columns:
        raise ValueError(f"{config.target_column!r} is missing from train.csv.")

    if config.target_column in test_df.columns:
        raise ValueError(f"{config.target_column!r} should not appear in test.csv.")

    if sample_submission.width != 2:
        raise ValueError("sample_submission.csv must contain exactly two columns.")

    if sample_submission.columns[0] != config.id_column:
        raise ValueError(
            f"sample_submission.csv should start with {config.id_column!r}; "
            f"found {sample_submission.columns[0]!r}."
        )

    if not sample_submission[config.id_column].equals(test_df[config.id_column]):
        raise ValueError("sample_submission.csv ids do not match test.csv ids.")


def split_train_validation(train_df: pl.DataFrame, config: CompetitionConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    with_buckets = train_df.with_columns(
        pl.col(config.id_column)
        .cast(pl.Utf8)
        .map_elements(stable_bucket, return_dtype=pl.UInt64)
        .mod(10_000)
        .alias("_bucket")
    )

    cutoff = int(config.validation_fraction * 10_000)
    valid_df = with_buckets.filter(pl.col("_bucket") < cutoff).drop("_bucket")
    fit_df = with_buckets.filter(pl.col("_bucket") >= cutoff).drop("_bucket")

    if fit_df.is_empty() or valid_df.is_empty():
        raise ValueError("Validation split produced an empty train or validation set.")

    return fit_df, valid_df


def stable_bucket(value: str) -> int:
    total = 0
    for char in value:
        total = (total * 131 + ord(char)) % 1_000_000_007
    return total


def normalize_predictions(predictions: object, config: CompetitionConfig, expected_rows: int) -> pl.Series:
    if isinstance(predictions, pl.Series):
        series = predictions.rename(config.prediction_column)
    else:
        series = pl.Series(config.prediction_column, predictions)

    if series.len() != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} predictions for {config.prediction_column!r}, "
            f"received {series.len()}."
        )

    return series


def score_predictions(y_true: list[object], y_pred: list[object], config: CompetitionConfig) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("Prediction length does not match ground truth length.")

    metric_name = config.metric_name.lower()

    if metric_name == "accuracy":
        return sum(actual == predicted for actual, predicted in zip(y_true, y_pred)) / len(y_true)

    if metric_name == "balanced_accuracy":
        labels = sorted({label for label in y_true})
        recalls = []
        for label in labels:
            positives = sum(actual == label for actual in y_true)
            true_positives = sum(
                actual == label and predicted == label
                for actual, predicted in zip(y_true, y_pred)
            )
            recalls.append(true_positives / positives)
        return sum(recalls) / len(recalls)

    if metric_name == "mae":
        return sum(abs(float(actual) - float(predicted)) for actual, predicted in zip(y_true, y_pred)) / len(y_true)

    if metric_name == "rmse":
        squared_error = sum((float(actual) - float(predicted)) ** 2 for actual, predicted in zip(y_true, y_pred))
        return math.sqrt(squared_error / len(y_true))

    raise ValueError(
        f"Unsupported METRIC_NAME {config.metric_name!r}. Add it to score_predictions in src/evaluate.py."
    )


def write_submission(
    test_df: pl.DataFrame,
    predictions: pl.Series,
    sample_submission: pl.DataFrame,
    config: CompetitionConfig,
    submission_dir: Path,
) -> str:
    submission_dir.mkdir(parents=True, exist_ok=True)

    submission = pl.DataFrame(
        {
            config.id_column: test_df[config.id_column],
            config.prediction_column: predictions,
        }
    )

    submission = submission.select(sample_submission.columns)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = submission_dir / f"submission-{timestamp}.csv"
    submission.write_csv(output_path)
    return str(output_path.relative_to(Path(__file__).resolve().parents[1]))


if __name__ == "__main__":
    main()
