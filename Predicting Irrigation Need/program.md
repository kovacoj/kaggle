# Predicting Irrigation Need Agent Workflow

This competition uses a Kaggle-style adaptation of the `autoresearch` loop.

The key difference from LLM training repos is that the agent must understand the dataset and the fixed benchmark split before it starts editing model code.

## Files In Scope

- `README.md` — Kaggle brief plus a benchmark block refreshed from `results.tsv`.
- `AGENTS.md` — local operating rules.
- `program.md` — this workflow.
- `src/profile_data.py` — fixed data-understanding script.
- `src/benchmark.py` — fixed benchmark infrastructure.
- `benchmark/holdout_v1/` — fixed benchmark split artifacts.
- `src/evaluate.py` — fixed experiment harness.
- `src/experiment.py` — the primary agent-owned modeling surface.
- `src/baseline_catboost.py` — reference baseline, not the default experiment surface.
- `src/leaderboard.py` — ranks prediction CSVs already written under `artifacts/`.
- `src/analyze_results.py` — refreshes benchmark summaries, progress plots, and approach memory from `results.tsv`.
- `results.tsv` — auto-appended experiment log.
- `history/` — archived per-run snapshots.
- `artifacts/approach_memory.md` — generated memory of tried approaches and their outcomes.
- `artifacts/ideas.md` — agent-editable list of experiment ideas. Append new ideas; mark them `[tried]` once tested.
- `DIARY.md` — human-readable research notes.
- `DOCKER.md` and `run_agent_container.sh` — optional container guardrails.

## Setup

Before the autonomous loop starts:

1. Read `README.md`, `AGENTS.md`, `program.md`, `src/profile_data.py`, `src/benchmark.py`, `src/evaluate.py`, and `src/experiment.py`.
2. Verify the competition data exists:
   - `data/train.csv`
   - `data/test.csv`
   - `data/sample_submission.csv`
3. If you want stronger filesystem guardrails, start the optional container shell with `bash run_agent_container.sh`.
4. Verify the fixed benchmark split exists. If not, initialize it once:

```bash
uv run python src/benchmark.py init
```

5. Inspect the benchmark split:

```bash
uv run python src/benchmark.py describe
```

6. Run the data profile first:

```bash
uv run python src/profile_data.py > logs/profile.log 2>&1
```

7. Read `artifacts/data_profile.md`, `artifacts/approach_memory.md`, and `artifacts/ideas.md` before changing `src/experiment.py`.
8. Run the baseline once on `smoke`:

```bash
EXPERIMENT_DESCRIPTION="baseline" uv run python src/evaluate.py --benchmark smoke > logs/baseline.log 2>&1
```

9. Read the summary lines from the log:

```bash
rg "^(run_id|benchmark|metric_name|metric_direction|metric_value|runtime_seconds|validation_predictions|submission_file|experiment|description|snapshot):" logs/baseline.log
```

10. Refresh the analysis outputs, approach memory, and README benchmark block:

```bash
uv run python src/analyze_results.py
```

## What The Agent Can Edit

- Default to `src/experiment.py`.
- Keep any extra helper modules under `src/` small and justified.

## What The Agent Must Not Edit

- `data/`
- `src/benchmark.py`
- `benchmark/holdout_v1/`
- `src/profile_data.py`
- `src/evaluate.py`
- the human-authored narrative in `README.md`
- `program.md`
- `AGENTS.md`
- root `pyproject.toml`
- root `uv.lock`

## Experiment Loop

Repeat until interrupted:

1. Re-read `artifacts/data_profile.md` and `artifacts/approach_memory.md` — compare the best scores there against your new idea to avoid repeating discarded approaches.
2. Check `artifacts/ideas.md` for untried experiment ideas. Pick one or add your own.
3. If you add a new idea, log it in `artifacts/ideas.md` before coding with a short hypothesis and the signal you expect.
4. Mark the chosen idea `[tried]` when you start, and append a short outcome note after the run so the next agent can build on the result.
5. Make one experiment-sized change in `src/experiment.py`.
   Update `experiment.APPROACH` and `experiment.DESCRIPTION` so the idea is logged clearly.
6. Create a commit for that candidate.
7. Run the harness with a short note:

```bash
EXPERIMENT_DESCRIPTION="your idea here" uv run python src/evaluate.py --benchmark smoke > logs/run.log 2>&1
```

8. Extract the summary:

```bash
rg "^(run_id|benchmark|metric_name|metric_direction|metric_value|runtime_seconds|validation_predictions|submission_file|experiment|description|snapshot):" logs/run.log
```

9. If the summary is missing, inspect the crash:

```bash
tail -n 50 logs/run.log
```

10. Review the auto-recorded row in `results.tsv` and the archived snapshot in `history/<run_id>/`.
11. Update the `status` in `results.tsv` to `keep`, `discard`, or `crash` after the decision.

12. Add a short dated note to `DIARY.md` when the run changes direction, confirms a strong result, or teaches you something reusable.
13. Refresh the analysis outputs, approach memory, and ideas:

```bash
uv run python src/analyze_results.py
```

14. If a `smoke` result looks promising, confirm it on `full` before treating it as the new headline benchmark.
15. Keep the commit only if the benchmark improved enough to justify the complexity.

## Output Contract

`src/evaluate.py` prints a machine-readable summary block like this:

```text
---
run_id:                 20260427-101530-a1b2c3d
benchmark:              smoke
metric_name:            balanced_accuracy
metric_direction:       maximize
metric_value:           0.587170
runtime_seconds:        2.4
validation_predictions: artifacts/experiment_runs/20260427-101530-a1b2c3d-smoke.csv
submission_file:        submissions/submission-20260427-101530-a1b2c3d.csv
experiment:             majority-class baseline
description:            baseline
snapshot:               history/20260427-101530-a1b2c3d
```

## Auto-Recorded Results

The harness auto-appends `results.tsv` using tab-separated values:

```tsv
run_id	commit	benchmark	approach	metric_name	metric_direction	metric_value	runtime_seconds	status	description	snapshot
```

It also writes `history/<run_id>/summary.json` and copies the exact `src/experiment.py` used for that run.

## Approach Memory

`src/analyze_results.py` writes `artifacts/approach_memory.md`, which summarizes:

- best current approaches by benchmark
- approaches worth building on
- discarded or crashed approaches worth avoiding
- recent runs with their evaluation outcomes

Read it before the next iteration so the agent can build on prior wins and avoid repeating the same mistake.

## Idea Logging Rules

- `artifacts/ideas.md` is the handoff file for future agents. Add ideas before implementation, not after.
- Every idea entry should include four things: status, short hypothesis, expected signal, and outcome note.
- Mark ideas `[tried]` as soon as execution starts, even if the run later crashes.
- If a run crashes or underperforms, record the failure mode in the idea note so the next agent does not repeat it blindly.
- If a run works, record what likely caused the gain so the next agent can extend it instead of rediscovering it.

## README Benchmark Block

`README.md` should explain the competition and also show the best current benchmark.

Use this marker block:

```md
<!-- benchmark:start -->
...
<!-- benchmark:end -->
```

`src/analyze_results.py` refreshes that block from `results.tsv` when `README.md` is writable. It always writes `artifacts/benchmark_summary.md`.

## Safety Rules

- Treat `data/` as read-only.
- Do not skip the data-understanding step.
- Treat `src/benchmark.py` and `benchmark/holdout_v1/` as fixed scoring infrastructure.
- Use as many CPU cores as possible whenever the training library supports it, unless there is a concrete reason not to.
- Use `smoke` for quick iteration and `full` to confirm promising changes.
- When the dataset is too large for fast iteration, debug and test on `smoke` or another smaller stratified subset first, then confirm on the full fixed benchmark.
- If you need stronger guardrails, run the loop inside the optional Docker wrapper so protected files are mounted read-only.
- Review the archived snapshot before deciding to keep or discard a run.
- Use `data/sample_submission.csv` as the submission schema.
- Do not create a new submission file unless the run completed successfully.
- Prefer smaller, reviewable edits over large rewrites.
- Each experiment run must complete within **15 minutes** wall-clock time. If a run exceeds this, reduce the training data size or iteration count before retrying.

## Faster Iteration

Training on the full 630K-row dataset is expensive. Use these strategies to keep the feedback loop tight.

### Stratified Micro-Subsets

For rapid feature engineering and hyperparameter experiments, extract a small stratified subset of the training data that preserves the target class distribution. Suggested sizes:

- **1 % subset** (~6 300 rows): for sanity-checking code paths and quick feature iterations.
- **10 % subset** (~63 000 rows): for more reliable signal before committing to a full `smoke` or `full` run.

Create a subset once and cache it (e.g. under `artifacts/`) so repeated runs skip the resampling. Always stratify on `Irrigation_Need` so the severe class imbalance (High ≈ 3.3 %) is faithfully represented.

Workflow:

1. Develop and debug on the 1 % subset.
2. Promising changes → validate on the 10 % subset or `smoke`.
3. Best candidates → confirm on `full`.

### Hardware Acceleration

Prefer libraries that exploit multi-core CPUs and SIMD instructions out of the box (CatBoost, LightGBM, XGBoost already do). For custom feature-engineering or post-processing code that is still a bottleneck:

- **Numba**: JIT-compile pure-Python numeric loops with `@njit(parallel=True)` and `prange`. Especially effective for row-wise feature transforms that Polars expressions cannot express natively.
- **NumPy vectorisation**: replace Python-level loops with broadcast operations before reaching for Numba.
- **Polars lazy**: use `pl.LazyFrame` and `collect()` to let the query optimizer fuse and parallelise transformations.
- **Mixed-precision arithmetic**: Polars defaults to `Float64`. Most features in this dataset do not require double precision. Cast to `Float32` (`.cast(pl.Float32)`) after computation to halve memory and speed up downstream operations. GBM libraries internally use 32-bit floats anyway, so the extra precision is wasted.
- **Chunked processing**: When working with the full dataset, process data in chunks (e.g. via `pl.read_csv(..., n_rows=chunk_size)` in a loop or by slicing an in-memory frame) to keep peak RAM usage bounded. This has not been a bottleneck so far, but it is good hygiene for larger feature sets.

Only introduce Numba (or similar) if profiling shows that the feature-engineering step dominates runtime. Do not add it speculatively.
