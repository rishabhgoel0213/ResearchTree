#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/codex-uv-cache}"
mkdir -p "$UV_CACHE_DIR"

exec uv run \
  --with 'arxiv>=2.0.0' \
  --with 'pymongo>=4.0.0' \
  -- \
  python3 "$BASE_DIR/arxiv_tool.py" "$@"
