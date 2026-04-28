from __future__ import annotations

from pathlib import Path

from benchmark import PROJECT_ROOT, ensure_benchmark_exists
import analyze_results
import profile_data

RESULTS_PATH = PROJECT_ROOT / "results.tsv"
RESULTS_HEADER = (
    "run_id\tcommit\tbenchmark\tapproach\tmetric_name\tmetric_direction\tmetric_value\t"
    "runtime_seconds\tstatus\tdescription\tsnapshot\n"
)


def main() -> None:
    ensure_required_data_files()
    ensure_benchmark_exists()
    ensure_results_file()
    profile_data.main()
    if has_logged_runs():
        analyze_results.main()
    else:
        write_placeholder_outputs()


def ensure_required_data_files() -> None:
    data_dir = PROJECT_ROOT / "data"
    required = ["train.csv", "test.csv", "sample_submission.csv"]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(f"Missing required competition files: {missing_str}")


def ensure_results_file() -> None:
    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text(RESULTS_HEADER, encoding="utf-8")


def has_logged_runs() -> bool:
    return len(RESULTS_PATH.read_text(encoding="utf-8").splitlines()) > 1


def write_placeholder_outputs() -> None:
    artifact_dir = PROJECT_ROOT / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "benchmark_summary.md").write_text("- No successful runs logged yet.\n", encoding="utf-8")
    (artifact_dir / "approach_memory.md").write_text("# Approach Memory\n\n- No approaches logged yet.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
