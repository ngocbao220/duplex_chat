#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# DuplexChat filter — TEMPLATE. Post-crawl cleanup on one many-core node:
# drops music-genre feeds, (audio_url, dialogue_idx) duplicates, and dialogues
# that are >= 1/3 of the parent episode (usually diarization failures). Runs
# NUM_RANKS parallel filter processes over inputs[i::NUM_RANKS], then merges
# per-rank stats into stats.json.
#
# Generic template — adapt scheduler directives and `# EDIT:` lines.
# ─────────────────────────────────────────────────────────────────────────────
#PBS -q YOUR_CPU_QUEUE          # EDIT
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -r y
#PBS -N duplexchat_filter
#PBS -j oe

set -euo pipefail

NUM_RANKS=16                    # EDIT: ~ cores / 5
LANG=en                         # EDIT: en or ja
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
INPUT_DIR=${PROJECT_DIR}/data/wds_${LANG}
OUTPUT_DIR=${PROJECT_DIR}/data/wds_${LANG}_filtered

cd "${PROJECT_DIR}"
# EDIT: put `uv` on PATH for batch jobs if needed.
mkdir -p "${PROJECT_DIR}/logs" "${OUTPUT_DIR}"

echo "Filter start $(date): input=${INPUT_DIR} output=${OUTPUT_DIR} ranks=${NUM_RANKS}"

PIDS=()
for i in $(seq 0 $((NUM_RANKS - 1))); do
    uv run python3 "${PROJECT_DIR}/scripts/filter_shards.py" \
        --input "${INPUT_DIR}" \
        --output "${OUTPUT_DIR}" \
        --shard ${i} \
        --num-shards ${NUM_RANKS} \
        --shard-size-gb 3.0 \
        --num-writers 4 \
        > "${PROJECT_DIR}/logs/filter_${LANG}_${i}.log" 2>&1 &
    PIDS+=($!)
done

FAILED=0
for i in $(seq 0 $((NUM_RANKS - 1))); do
    wait "${PIDS[$i]}" || FAILED=$((FAILED + 1))
done
echo "Filter done $(date): failed ranks ${FAILED}/${NUM_RANKS}"
echo "Merge per-rank stats_n*.json into stats.json as needed."
exit ${FAILED}
