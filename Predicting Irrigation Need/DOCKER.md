# Optional Container Isolation

This competition includes an optional Docker runner for stronger guardrails during longer autonomous runs.

The container protects the fixed parts of the workspace:

- `data/` is mounted read-only.
- `README.md` is mounted read-only.
- `AGENTS.md` is mounted read-only.
- `program.md` is mounted read-only.
- `src/profile_data.py` is mounted read-only.
- `src/benchmark.py` is mounted read-only.
- `src/evaluate.py` is mounted read-only.
- `benchmark/holdout_v1/` is mounted read-only.

The writable surfaces stay available to the agent:

- `src/experiment.py`
- `artifacts/`
- `logs/`
- `submissions/`
- `history/`
- `results.tsv`
- `DIARY.md`

## Start The Container

From the competition folder:

```bash
bash run_agent_container.sh
```

This starts an interactive shell after running `uv sync --frozen`.

Inside the container, the environment lives at `./.venv` via `UV_PROJECT_ENVIRONMENT=/workspace/competition/.venv`, so the shared repo root can remain read-only.

## Notes

- This is an optional safety layer, not a replacement for the fixed benchmark harness.
- The current container setup protects files, not network access.
- The benchmark block in `README.md` can still be refreshed outside the container by running `uv run python src/analyze_results.py` locally.
