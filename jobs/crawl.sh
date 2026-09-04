#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# DuplexChat crawl — TEMPLATE. Multi-GPU-node crawl with diarization + DialogueSidon
# separation, one MPI rank per node, each node crawling feed_urls[rank::NUM_NODES].
#
# This is a generic template. Adapt the scheduler directives and the marked
# `# EDIT:` lines to your cluster. It is written for PBS + Open MPI but the core
# is just the `duplexchat-pipe run` invocation, which works on any GPU machine.
# For a single node, drop the mpirun wrapper and run the command directly with
# --node-index 0 --num-nodes 1.
# ─────────────────────────────────────────────────────────────────────────────
#PBS -q YOUR_GPU_QUEUE          # EDIT: your GPU queue
#PBS -l select=64               # EDIT: number of nodes
#PBS -l walltime=48:00:00
#PBS -r y
#PBS -N duplexchat_crawl
#PBS -j oe
# EDIT: add your accounting/group directive if required, e.g. #PBS -W group_list=...

set -euo pipefail

NUM_NODES=64                    # EDIT: must match `#PBS -l select`
LANG=vi                         # EDIT: language filter
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)   # repo root; EDIT if submitting elsewhere
OUTPUT_DIR=${PROJECT_DIR}/data/wds/${PBS_JOBID%%.*}

cd "${PROJECT_DIR}"
# EDIT: put `uv` (and any CUDA libs) on PATH for batch jobs, e.g.:
# export PATH="${HOME}/.local/bin:${PATH}"
mkdir -p "${PROJECT_DIR}/logs"

# One rank per node. `set +e` + `exit 0` so a single node crash (e.g. OOM) does
# not tear down the whole MPI world.
mpirun -np ${NUM_NODES} --map-by ppr:1:node --max-restarts 0 bash -c '
    set +e
    NODE_INDEX=${OMPI_COMM_WORLD_RANK}
    SCRATCH=${LOCALDIR:-/tmp}          # node-local NVMe if available

    echo "Node ${NODE_INDEX}/'"${NUM_NODES}"' on $(hostname) at $(date), scratch=${SCRATCH}"

    uv run duplexchat-pipe run \
        --output "'"${OUTPUT_DIR}"'" \
        --cache-dir "'"${PROJECT_DIR}"'/data/cache" \
        --languages '"${LANG}"' \
        --rss-workers 16 \
        --download-workers 4 \
        --process-workers 4 \
        --separation-workers 1 \
        --scratch-dir "${SCRATCH}" \
        --enable-diarization \
        --diarization-device cuda \
        --enable-separation \
        --node-index ${NODE_INDEX} \
        --num-nodes '"${NUM_NODES}"' \
        || echo "Node ${NODE_INDEX} failed (exit $?) at $(date)"

    echo "Node ${NODE_INDEX} finished at $(date)"
    exit 0
'
