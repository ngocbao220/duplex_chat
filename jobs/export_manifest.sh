#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# DuplexChat metadata export — TEMPLATE. Produces the copyright-safe reconstruction
# manifest (rss_url, audio_url, dialogue span timings — no audio) from the built
# WebDataset shards. Array task i of NUM_TASKS scans shards[i::NUM_TASKS] and writes
# one manifest_part_*.jsonl.gz; merge them afterwards with export_manifest_reduce.py.
#
# Generic template — adapt scheduler directives and `# EDIT:` lines.
# ─────────────────────────────────────────────────────────────────────────────
#PBS -q YOUR_CPU_QUEUE          # EDIT
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -r y
#PBS -J 0-15                    # EDIT: 0-(NUM_TASKS-1)
#PBS -N duplexchat_export
#PBS -j oe

set -euo pipefail

NUM_TASKS=16                    # EDIT: must match the `#PBS -J` range size
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
DATA_ROOT=${PROJECT_DIR}/data   # holds wds_ja_filtered/ and wds_en_filtered/
OUT_DIR=${PROJECT_DIR}/data/manifest_parts

cd "${PROJECT_DIR}"
mkdir -p "${OUT_DIR}"
echo "Export task ${PBS_ARRAY_INDEX}/${NUM_TASKS} on $(hostname) at $(date)"

uv run python3 "${PROJECT_DIR}/analysis/export_manifest_scan.py" \
    --data-root "${DATA_ROOT}" \
    --task-id "${PBS_ARRAY_INDEX}" \
    --num-tasks "${NUM_TASKS}" \
    --procs 64 \
    --out-dir "${OUT_DIR}"

echo "Export task ${PBS_ARRAY_INDEX} finished at $(date)"

# After all array tasks finish, merge + globally dedup:
#   uv run python3 analysis/export_manifest_reduce.py \
#       --parts-dir data/manifest_parts --out-dir data/manifest
