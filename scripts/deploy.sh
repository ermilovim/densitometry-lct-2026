#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# A public bind is convenient for an isolated judging stand. Override it with
# VIEWER_BIND=127.0.0.1 when the service is published through a reverse proxy.
export VIEWER_BIND="${VIEWER_BIND:-0.0.0.0}"
export VIEWER_PORT="${VIEWER_PORT:-8765}"
export GPU_MODE="${GPU_MODE:-auto}"

exec "$script_dir/restart_viewer.sh"
