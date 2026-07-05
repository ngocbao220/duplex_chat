#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# DuplexChat reconstruction — TEMPLATE. Rebuilds the audio dataset from a released
# metadata manifest. Each array task reconstructs episodes[shard_index::NUM_SHARDS]
# on one GPU (download -> 16kHz mono -> slice span -> DialogueSidon separation ->
# stereo MP3 shards). Requires a GPU + a HuggingFace token with access to the
# gated sarulab-speech/DialogueSidon weights.
#
# Generic template — adapt scheduler directives and `# EDIT:` lines.
# ─────────────────────────────────────────────────────────────────────────────
#PBS -q YOUR_GPU_QUEUE          # EDIT
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -r y
#PBS -J 0-63                    # EDIT: 0-(NUM_SHARDS-1); one GPU task per shard
#PBS -N duplexchat_reconstruct
#PBS -j oe

set -euo pipefail

NUM_SHARDS=64                   # EDIT: must match the `#PBS -J` range size
LANG=en                         # EDIT: en or ja
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
MANIFEST_DIR=${PROJECT_DIR}/data/manifest        # EDIT: dir with duplexchat_manifest_*.jsonl.gz
OUTPUT_DIR=${PROJECT_DIR}/data/reconstructed_${LANG}

cd "${PROJECT_DIR}"
# EDIT: put `uv` (+ CUDA libs) on PATH and ensure `huggingface-cli login` was run.
echo "Reconstruct shard ${PBS_ARRAY_INDEX}/${NUM_SHARDS} (${LANG}) on $(hostname) at $(date)"

uv run python3 "${PROJECT_DIR}/scripts/reconstruct_dataset.py" \
    --language "${LANG}" \
    --manifest-dir "${MANIFEST_DIR}" \
    --output "${OUTPUT_DIR}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${PBS_ARRAY_INDEX}" \
    --num-steps 30 \
    --scratch-dir "${LOCALDIR:-/tmp}"

echo "Reconstruct shard ${PBS_ARRAY_INDEX} finished at $(date)"
