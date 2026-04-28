#!/usr/bin/env bash

set -euo pipefail

competition_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${competition_dir}/.." && pwd)"

mkdir -p \
  "${competition_dir}/artifacts" \
  "${competition_dir}/history" \
  "${competition_dir}/logs" \
  "${competition_dir}/submissions"

docker run --rm -it \
  --workdir /workspace/competition \
  --mount type=bind,src="${repo_root}",dst=/workspace,readonly \
  --mount type=bind,src="${competition_dir}",dst=/workspace/competition \
  --mount type=bind,src="${competition_dir}/AGENTS.md",dst=/workspace/competition/AGENTS.md,readonly \
  --mount type=bind,src="${competition_dir}/README.md",dst=/workspace/competition/README.md,readonly \
  --mount type=bind,src="${competition_dir}/program.md",dst=/workspace/competition/program.md,readonly \
  --mount type=bind,src="${competition_dir}/data",dst=/workspace/competition/data,readonly \
  --mount type=bind,src="${competition_dir}/src/evaluate.py",dst=/workspace/competition/src/evaluate.py,readonly \
  --mount type=bind,src="${competition_dir}/src/profile_data.py",dst=/workspace/competition/src/profile_data.py,readonly \
  --mount type=volume,src=kaggle-agent-uv-cache,dst=/root/.cache/uv \
  ghcr.io/astral-sh/uv:python3.13-bookworm \
  bash -lc 'export UV_PROJECT_ENVIRONMENT=/workspace/competition/.venv; uv sync --frozen; exec bash'
