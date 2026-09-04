#!/bin/bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
cd "${PROJECT_DIR}"

INPUT=${INPUT:-data/wds}
OUTPUT=${OUTPUT:-data/wds_filtered}
RUN_ID=${RUN_ID:-filter_$(date +%H%M%S)}
LOG_DIR="logs/$(date +%F)/${RUN_ID}"

mkdir -p "${LOG_DIR}"
uv run python3 scripts/filter_shards.py --input "${INPUT}" --output "${OUTPUT}" "$@" \
  2>&1 | tee "${LOG_DIR}/run.log"
