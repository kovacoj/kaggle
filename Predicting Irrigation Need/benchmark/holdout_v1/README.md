# Holdout Benchmark v1

- Source of truth: `spec.json` and `splits.csv`
- Metric: balanced accuracy
- Labels: `Low`, `Medium`, `High`
- Contract for predictions: CSV with columns `id,Irrigation_Need`

Usage rules:

- Use `smoke` for fast iteration.
- Use `full` to confirm a promising result.
- Model code should live under `src/`.
- Model outputs should go under `artifacts/`.
- Treat files under `benchmark/holdout_v1/` and `src/benchmark.py` as fixed benchmark infrastructure.

Recommended workflow:

1. Generate or refresh the benchmark once: `uv run python src/benchmark.py init`
2. Inspect the split: `uv run python src/benchmark.py describe`
3. Train a model against `smoke` or `full`
4. Score its validation predictions with `uv run python src/benchmark.py score --benchmark <name> --predictions <csv>`
