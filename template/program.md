# Kaggle Autoresearch Template

This template adapts the `autoresearch` idea to a Kaggle-style tabular competition.

The human owns the competition brief and the evaluation harness. The agent owns the experiment logic.

## Files In Scope

- `README.md` — Kaggle brief plus a small benchmark block. The benchmark block can be refreshed from `results.tsv` by `src/analyze_results.py`.
- `src/profile_data.py` — fixed data-understanding script. Run this before any baseline or experiment loop.
- `src/evaluate.py` — fixed local harness. Human configures the constants once, then freezes it.
- `src/experiment.py` — the file the agent edits during autonomous experimentation.
- `results.tsv` — untracked experiment log, auto-appended by the harness.
- `history/` — archived per-run snapshots, auto-created by the harness.
- `DIARY.md` — short human-readable research notes.
- `DOCKER.md` and `run_agent_container.sh` — optional container-based guardrails.
- `src/analyze_results.py` — optional summary/plotting for `results.tsv`.

## Setup

Before the experiment loop starts:

1. Pick a run tag such as `apr27` or `apr27-gpu0`.
2. Create a dedicated branch such as `autoresearch/<tag>`.
3. Read these files for full context:
   - `README.md`
   - `AGENTS.md`
   - `program.md`
   - `src/profile_data.py`
   - `src/evaluate.py`
   - `src/experiment.py`
4. Verify the competition data exists:
    - `data/train.csv`
    - `data/test.csv`
    - `data/sample_submission.csv`
   If you want file-system guardrails, start the optional container shell with `bash run_agent_container.sh` before the autonomous loop.
5. Run the data profile first:

```bash
uv run src/profile_data.py > logs/profile.log 2>&1
```

6. Read `artifacts/data_profile.md` and identify the basic modeling constraints before touching `src/experiment.py`.
7. Confirm the configuration block at the top of `src/evaluate.py` has been updated for this competition.
8. Run the baseline once. Add a human-readable note with `EXPERIMENT_DESCRIPTION`:

```bash
EXPERIMENT_DESCRIPTION="baseline" uv run src/evaluate.py > logs/baseline.log 2>&1
```

9. Read the summary lines from the log:

```bash
rg "^(run_id|metric_name|metric_direction|metric_value|runtime_seconds|submission_file|experiment|description|snapshot):" logs/baseline.log
```

10. Confirm that `results.tsv` and `history/<run_id>/summary.json` were created automatically.
11. Run the analysis refresh so `artifacts/benchmark_summary.md`, `artifacts/progress.png`, and the README benchmark block stay current:

```bash
uv run src/analyze_results.py
```

## What The Agent Can Edit

- `src/experiment.py`

## What The Agent Must Not Edit

- `data/`
- `src/evaluate.py`
- `src/profile_data.py`
- `README.md`
- `program.md`
- `AGENTS.md`
- root `pyproject.toml`
- root `uv.lock`

## Experiment Loop

Repeat until interrupted:

1. Re-read the latest data profile and inspect the current branch and last winning commit.
2. Make one experiment-sized change in `src/experiment.py`.
3. Create a commit for that candidate.
4. Run the harness with a short note describing the idea:

```bash
EXPERIMENT_DESCRIPTION="your idea here" uv run src/evaluate.py > logs/run.log 2>&1
```

5. Extract the result summary:

```bash
rg "^(run_id|metric_name|metric_direction|metric_value|runtime_seconds|submission_file|experiment|description|snapshot):" logs/run.log
```

6. If the summary is missing, inspect the crash:

```bash
tail -n 50 logs/run.log
```

7. Review the auto-recorded row in `results.tsv` and the archived snapshot in `history/<run_id>/`.
8. Update the `status` in `results.tsv` to `keep`, `discard`, or `crash` after you decide what to do with the run.
9. Run `uv run src/analyze_results.py` to refresh `artifacts/benchmark_summary.md`, `artifacts/progress.png`, and the README benchmark block when writable.
10. If the metric improved in the configured direction, keep the commit.
11. If the metric did not improve, restore the prior winning state before trying the next idea.

## Output Contract

`src/evaluate.py` prints a machine-readable block like this:

```text
---
run_id:            20260427-101530-a1b2c3d
metric_name:       balanced_accuracy
metric_direction:  maximize
metric_value:      0.587170
runtime_seconds:   2.4
submission_file:   submissions/submission-20260427-101530.csv
experiment:        majority-class baseline
description:       baseline
snapshot:          history/20260427-101530-a1b2c3d
```

## Auto-Recorded Results

The harness auto-appends `results.tsv` using tab-separated values:

```tsv
run_id	commit	metric_name	metric_direction	metric_value	runtime_seconds	status	description	snapshot
```

The harness also writes `history/<run_id>/summary.json` and copies the exact `src/experiment.py` used for that run.

## README Benchmark Block

The competition `README.md` should explain the competition and also show the best benchmark seen so far.

Use the marker block in `README.md`:

```md
<!-- benchmark:start -->
...
<!-- benchmark:end -->
```

`src/analyze_results.py` refreshes that block from `results.tsv` when `README.md` is writable. It always writes `artifacts/benchmark_summary.md` even when the README is read-only.

## Diary

After a significant experiment or decision, add a short dated note to `DIARY.md`.

## Safety Rules

- Treat `data/` as read-only.
- Do not skip the data-understanding step. Read the profile before proposing model changes.
- Keep generated artifacts under `artifacts/`, `logs/`, `submissions/`, or `results.tsv`.
- If you need stronger guardrails, run the loop inside the optional Docker wrapper so protected files are mounted read-only.
- Review the archived snapshot before deciding to keep or discard a run.
- Use `data/sample_submission.csv` as the submission schema.
- Do not create a new submission file unless the run completed successfully.
- Prefer smaller, reviewable edits over large rewrites.
