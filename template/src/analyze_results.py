from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results_path = project_root / "results.tsv"
    artifact_dir = project_root / "artifacts"
    readme_path = project_root / "README.md"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}. Initialize it before running this report.")

    results = pl.read_csv(results_path, separator="\t")
    results = results.with_columns(
        pl.col("status").cast(pl.Utf8).str.to_uppercase(),
        pl.col("metric_direction").cast(pl.Utf8).str.to_lowercase(),
        pl.col("metric_value").cast(pl.Float64),
        pl.col("runtime_seconds").cast(pl.Float64),
    )

    if results.is_empty():
        raise ValueError("results.tsv is empty.")

    print(f"Total experiments: {results.height}")
    print("Experiment outcomes:")
    print(results.group_by("status").len().sort("status"))

    kept = results.filter(pl.col("status") == "KEEP")
    decided = results.filter(pl.col("status").is_in(["KEEP", "DISCARD"]))
    if decided.height:
        keep_rate = kept.height / decided.height
        print(f"Keep rate: {kept.height}/{decided.height} = {keep_rate:.1%}")

    metric_name = results.item(0, "metric_name")
    metric_direction = results.item(0, "metric_direction")

    valid = results.filter(pl.col("status") != "CRASH")
    if valid.is_empty():
        raise ValueError("results.tsv contains only crash rows.")

    metric_values = valid["metric_value"].to_list()
    if metric_direction == "maximize":
        running_best = cumulative_best(metric_values, max)
    else:
        running_best = cumulative_best(metric_values, min)

    best_row = select_best_row(valid, metric_direction)

    kept_x = []
    kept_y = []
    kept_labels = []
    discard_x = []
    discard_y = []

    for index, row in enumerate(valid.iter_rows(named=True)):
        if row["status"] == "KEEP":
            kept_x.append(index)
            kept_y.append(row["metric_value"])
            kept_labels.append(row["description"])
        elif row["status"] == "DISCARD":
            discard_x.append(index)
            discard_y.append(row["metric_value"])

    fig, ax = plt.subplots(figsize=(14, 7))
    if discard_x:
        ax.scatter(discard_x, discard_y, s=16, c="#cccccc", alpha=0.5, label="Discarded")
    if kept_x:
        ax.scatter(kept_x, kept_y, s=42, c="#2ecc71", edgecolors="black", linewidths=0.5, label="Kept")
        for index, value, label in zip(kept_x, kept_y, kept_labels):
            short_label = label if len(label) <= 45 else f"{label[:42]}..."
            ax.annotate(short_label, (index, value), textcoords="offset points", xytext=(6, 6), fontsize=8)

    ax.plot(list(range(len(running_best))), running_best, color="#1f7a3d", linewidth=2, label="Running best")
    ax.set_title(f"Experiment Progress ({metric_name}, {metric_direction})")
    ax.set_xlabel("Experiment #")
    ax.set_ylabel(metric_name)
    ax.grid(alpha=0.2)
    ax.legend()

    output_path = artifact_dir / "progress.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    benchmark_summary = render_benchmark_summary(best_row)
    summary_path = artifact_dir / "benchmark_summary.md"
    summary_path.write_text(benchmark_summary, encoding="utf-8")

    readme_updated = update_readme_benchmark_block(readme_path, benchmark_summary)

    print(f"Saved {output_path.relative_to(project_root)}")
    print(f"Saved {summary_path.relative_to(project_root)}")
    if readme_updated:
        print(f"Updated {readme_path.relative_to(project_root)} benchmark block")
    else:
        print("Skipped README benchmark update")


def cumulative_best(values: list[float], chooser) -> list[float]:
    best_values: list[float] = []
    current = values[0]
    for value in values:
        current = chooser(current, value)
        best_values.append(current)
    return best_values


def select_best_row(results: pl.DataFrame, metric_direction: str) -> dict[str, object]:
    descending = metric_direction == "maximize"
    return results.sort("metric_value", descending=descending).row(0, named=True)


def render_benchmark_summary(best_row: dict[str, object]) -> str:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    description = str(best_row["description"]).strip() or "-"
    snapshot = str(best_row["snapshot"]).strip() or "-"
    return "\n".join(
        [
            f"- Metric: `{best_row['metric_name']}` (`{best_row['metric_direction']}`)",
            f"- Best score: `{float(best_row['metric_value']):.6f}`",
            f"- Best run: `{best_row['run_id']}`",
            f"- Status: `{best_row['status']}`",
            f"- Note: {description}",
            f"- Snapshot: `{snapshot}`",
            f"- Last updated: {timestamp}",
            "",
        ]
    )


def update_readme_benchmark_block(readme_path: Path, benchmark_summary: str) -> bool:
    if not readme_path.exists() or not readme_path.is_file():
        return False

    start_marker = "<!-- benchmark:start -->"
    end_marker = "<!-- benchmark:end -->"
    readme_text = readme_path.read_text(encoding="utf-8")
    if start_marker not in readme_text or end_marker not in readme_text:
        return False

    start_index = readme_text.index(start_marker) + len(start_marker)
    end_index = readme_text.index(end_marker)
    replacement = f"\n{benchmark_summary}"
    updated_text = readme_text[:start_index] + replacement + readme_text[end_index:]
    try:
        readme_path.write_text(updated_text, encoding="utf-8")
    except OSError:
        return False
    return True


if __name__ == "__main__":
    main()
