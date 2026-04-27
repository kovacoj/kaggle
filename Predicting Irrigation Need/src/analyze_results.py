from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from benchmark import PROJECT_ROOT

RESULTS_PATH = PROJECT_ROOT / "results.tsv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
README_PATH = PROJECT_ROOT / "README.md"


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing {RESULTS_PATH}")

    results = pl.read_csv(RESULTS_PATH, separator="\t")
    if results.is_empty():
        write_placeholder_outputs()
        print("No experiment rows logged yet.")
        return

    results = results.with_columns(
        pl.col("status").cast(pl.Utf8).str.to_uppercase(),
        pl.col("benchmark").cast(pl.Utf8).str.to_lowercase(),
        pl.col("metric_direction").cast(pl.Utf8).str.to_lowercase(),
        pl.col("metric_value").cast(pl.Float64),
        pl.col("runtime_seconds").cast(pl.Float64),
        pl.int_range(pl.len()).alias("experiment_index"),
    )

    non_crash = results.filter(pl.col("status") != "CRASH")
    accepted = results.filter(pl.col("status").is_in(["KEEP", "RAN"]))
    preferred = accepted if accepted.height else non_crash
    metric_name = results.item(0, "metric_name")
    metric_direction = results.item(0, "metric_direction")

    print(f"Total runs: {results.height}")
    print(results.group_by("status").len().sort("status"))

    plot_progress(non_crash, metric_name, metric_direction)
    benchmark_summary = render_benchmark_summary(preferred, metric_name, metric_direction)
    (ARTIFACT_DIR / "benchmark_summary.md").write_text(benchmark_summary, encoding="utf-8")

    approach_memory = render_approach_memory(results, preferred, metric_direction)
    (ARTIFACT_DIR / "approach_memory.md").write_text(approach_memory, encoding="utf-8")

    if update_readme_benchmark_block(README_PATH, benchmark_summary):
        print(f"Updated {README_PATH.relative_to(PROJECT_ROOT)} benchmark block")
    else:
        print("Skipped README benchmark update")

    print(f"Saved {(ARTIFACT_DIR / 'benchmark_summary.md').relative_to(PROJECT_ROOT)}")
    print(f"Saved {(ARTIFACT_DIR / 'approach_memory.md').relative_to(PROJECT_ROOT)}")
    print(f"Saved {(ARTIFACT_DIR / 'progress.png').relative_to(PROJECT_ROOT)}")


def write_placeholder_outputs() -> None:
    placeholder = "- No successful runs logged yet.\n"
    (ARTIFACT_DIR / "benchmark_summary.md").write_text(placeholder, encoding="utf-8")
    (ARTIFACT_DIR / "approach_memory.md").write_text("# Approach Memory\n\n- No approaches logged yet.\n", encoding="utf-8")
    update_readme_benchmark_block(README_PATH, placeholder)


def plot_progress(results: pl.DataFrame, metric_name: str, metric_direction: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = {"smoke": "#1f77b4", "full": "#d62728"}

    for benchmark_name in sorted(set(results["benchmark"].to_list())):
        subset = results.filter(pl.col("benchmark") == benchmark_name).sort("experiment_index")
        if subset.is_empty():
            continue

        x = subset["experiment_index"].to_list()
        y = subset["metric_value"].to_list()
        running = running_best(y, metric_direction)
        ax.scatter(x, y, s=28, alpha=0.7, label=f"{benchmark_name} score", color=colors.get(benchmark_name))
        ax.plot(x, running, linewidth=2, label=f"{benchmark_name} best", color=colors.get(benchmark_name))

    ax.set_title(f"Benchmark Progress ({metric_name}, {metric_direction})")
    ax.set_xlabel("Experiment #")
    ax.set_ylabel(metric_name)
    ax.grid(alpha=0.2)
    ax.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "progress.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_benchmark_summary(results: pl.DataFrame, metric_name: str, metric_direction: str) -> str:
    lines = [
        f"- Metric: `{metric_name}` (`{metric_direction}`)",
        f"- Total logged runs: `{results.height}`",
    ]

    for benchmark_name in ("full", "smoke"):
        subset = results.filter(pl.col("benchmark") == benchmark_name)
        if subset.is_empty():
            lines.append(f"- Best {benchmark_name}: not run yet")
            continue

        best_row = best_row_for_metric(subset, metric_direction)
        lines.append(
            f"- Best {benchmark_name}: `{float(best_row['metric_value']):.6f}` via `{best_row['approach']}` "
            f"(`{best_row['status'].lower()}`; run `{best_row['run_id']}`)"
        )

    lines.append(f"- Last updated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    return "\n".join(lines)


def render_approach_memory(results: pl.DataFrame, preferred: pl.DataFrame, metric_direction: str) -> str:
    lines = ["# Approach Memory", "", "Read this before the next experiment.", ""]

    lines.extend(["## Best Current Approaches", ""])
    for benchmark_name in ("full", "smoke"):
        subset = preferred.filter(pl.col("benchmark") == benchmark_name)
        if subset.is_empty():
            lines.append(f"- `{benchmark_name}`: no accepted runs yet")
            continue
        best_row = best_row_for_metric(subset, metric_direction)
        lines.append(
            f"- `{benchmark_name}`: `{best_row['approach']}` -> {float(best_row['metric_value']):.6f} "
            f"[{best_row['status'].lower()}] | {best_row['description']} | run `{best_row['run_id']}`"
        )
    lines.append("")

    build_on = results.filter(pl.col("status").is_in(["KEEP", "RAN"]))
    lines.extend(["## Build On", ""])
    if build_on.is_empty():
        lines.append("- No accepted runs yet.")
    else:
        for row in top_rows(build_on, metric_direction, limit=5):
            lines.append(
                f"- `{row['benchmark']}` | `{row['approach']}` | {float(row['metric_value']):.6f} | "
                f"{row['description']} | run `{row['run_id']}`"
            )
    lines.append("")

    avoid = results.filter(pl.col("status").is_in(["DISCARD", "CRASH"])).sort("experiment_index", descending=True)
    lines.extend(["## Avoid Repeating", ""])
    if avoid.is_empty():
        lines.append("- No discarded or crashed runs logged yet.")
    else:
        for row in avoid.head(8).iter_rows(named=True):
            lines.append(
                f"- `{row['benchmark']}` | `{row['approach']}` | {row['status'].lower()} | "
                f"{float(row['metric_value']):.6f} | {row['description']} | run `{row['run_id']}`"
            )
    lines.append("")

    lines.extend(["## Recent Runs", ""])
    for row in results.sort("experiment_index", descending=True).head(10).iter_rows(named=True):
        lines.append(
            f"- `{row['benchmark']}` | `{row['approach']}` | {float(row['metric_value']):.6f} | "
            f"{row['status'].lower()} | {row['description']} | run `{row['run_id']}`"
        )
    lines.append("")
    return "\n".join(lines)


def top_rows(results: pl.DataFrame, metric_direction: str, limit: int) -> list[dict[str, object]]:
    descending = metric_direction == "maximize"
    return results.sort("metric_value", descending=descending).head(limit).iter_rows(named=True)


def best_row_for_metric(results: pl.DataFrame, metric_direction: str) -> dict[str, object]:
    descending = metric_direction == "maximize"
    return results.sort("metric_value", descending=descending).row(0, named=True)


def running_best(values: list[float], metric_direction: str) -> list[float]:
    best_values: list[float] = []
    current = values[0]
    for value in values:
        if metric_direction == "maximize":
            current = max(current, value)
        else:
            current = min(current, value)
        best_values.append(current)
    return best_values


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
    updated_text = readme_text[:start_index] + f"\n{benchmark_summary}" + readme_text[end_index:]
    try:
        readme_path.write_text(updated_text, encoding="utf-8")
    except OSError:
        return False
    return True


if __name__ == "__main__":
    main()
