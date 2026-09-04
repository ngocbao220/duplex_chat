#!/bin/bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
cd "${PROJECT_DIR}"

uv run duplexchat-pipe run --phase diarize_segment "$@"
