# Optional Container Isolation

This template includes an optional Docker runner for cases where you want stronger guardrails around the agent.

The container setup is designed to protect the files that should not move during autonomous experimentation:

- `data/` is mounted read-only.
- `README.md` is mounted read-only.
- `program.md` is mounted read-only.
- `AGENTS.md` is mounted read-only.
- `src/profile_data.py` is mounted read-only.
- `src/evaluate.py` is mounted read-only.

The agent can still write where it is supposed to work:

- `src/experiment.py`
- `artifacts/`
- `logs/`
- `submissions/`
- `history/`
- `results.tsv`
- `DIARY.md`

## What The Container Protects

The runner mounts the shared repo root read-only, then overlays the active competition folder as writable, and then overlays the protected files again as read-only.

That means the agent can still use the shared root `pyproject.toml` and `uv.lock`, but it cannot rewrite them from inside the container.

## How To Start The Container

From the competition folder:

```bash
bash run_agent_container.sh
```

This opens an interactive shell inside the container after running `uv sync --frozen`.

Inside the container, a project-local `.venv` is created in the competition folder via `UV_PROJECT_ENVIRONMENT=/workspace/competition/.venv` so the shared repo root can remain mounted read-only.

## Notes

- This container setup protects files, not network access.
- If you later want stricter isolation, the next step is to build a dedicated image and optionally disable network access after the environment is baked.
- The default workflow can still stay local; use the container when you want extra safety for longer autonomous runs.
