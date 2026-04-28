# AGENTS.md

## Workspace Shape
- This is a Kaggle competition template inside the shared `/home/cady/personal/kaggle` repo.
- Competition folders should keep raw data in `data/`, reproducible experiment code in `src/`, optional human notebooks in `notebooks/`, and generated outputs in `artifacts/`, `logs/`, `history/`, or `submissions/`.
- `README.md` should be replaced with the Kaggle brief for the active competition, and it should include a small benchmark block showing the best current metric.
- `program.md` describes the autonomous experiment loop.
- `src/profile_data.py` is the fixed first step for understanding the dataset before experiments start.
- `src/evaluate.py` is the fixed local harness after the human configures it once; `src/experiment.py` is the agent-owned experimentation surface.
- `DOCKER.md` and `run_agent_container.sh` provide an optional container wrapper that mounts protected files read-only.

## Environment
- Use the shared root environment via `uv sync` from the repo root or `uv run ...` from inside the competition folder.
- The root project requires Python `>=3.13`.
- Prefer Polars and Matplotlib unless the human explicitly approves a new dependency.

## Working Conventions
- Treat `data/` as read-only after the competition files land there.
- Run `src/profile_data.py` and read `artifacts/data_profile.md` before changing `src/experiment.py`.
- Keep generated experiment logs in `logs/`, plots in `artifacts/`, submissions in `submissions/`, archived run snapshots in `history/`, and the run ledger in `results.tsv`.
- Run `src/analyze_results.py` after meaningful runs so the benchmark summary and README benchmark block stay fresh.
- Use `EXPERIMENT_DESCRIPTION` when running the harness so each result row has a readable note.
- Add short dated notes to `DIARY.md` when a run meaningfully changes the direction of the work.
- There is no repo-configured test, lint, typecheck, formatter, task runner, or CI workflow. Do not claim those checks ran unless you add the tooling yourself.
- Prefer script-first automation in `src/`; notebooks are for human exploration, not for the autonomous loop.

## Agent Rules
- Do not rewrite the human-authored competition narrative in `README.md`; only the benchmark marker block may be refreshed from `results.tsv`.
- If the human asks for stronger isolation, prefer the container wrapper so `data/`, `src/evaluate.py`, and `src/profile_data.py` stay read-only at the filesystem level.
- Default autonomous changes to `src/experiment.py`.
- Use `data/sample_submission.csv` as the source of truth for submission column names and order.

## Git Gotchas
- `.gitattributes` at the repo root routes `*.csv` and `*.ipynb` through Git LFS. Preserve that when adding or rewriting datasets or notebooks.
- This template has a local `.gitignore` for `results.tsv`, `artifacts/`, `history/`, `logs/`, and generated submissions. Keep those outputs untracked unless the human asks otherwise.
