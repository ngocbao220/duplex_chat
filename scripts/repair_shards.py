"""Repair truncated webdataset shards.

When a crawl job is killed mid-write (walltime, OOM), the last .tar.gz
per node ends up with a truncated gzip stream. webdataset's
warn_and_continue handler often bails on the entire shard when gzip
decompression fails, losing all subsequent samples.

This script reads each shard using Python's tarfile (which is more
tolerant) and rewrites a clean copy with only the valid samples. The
original shards are left untouched; repaired shards go to --output.

Usage:

    uv run python3 scripts/repair_shards.py \
        --input data/wds_ja \
        --output data/wds_ja_repaired \
        --workers 8
"""

from __future__ import annotations

import argparse
import gzip
import io
import logging
import os
import sys
import tarfile
from multiprocessing import Pool
from pathlib import Path

LOGGER = logging.getLogger("repair_shards")


def iter_tar_members(shard_path: Path):
    """Yield (member, data_bytes) for every readable file in a .tar.gz.

    Stops gracefully at the first unrecoverable read error instead of
    raising — this is the core tolerance improvement over webdataset.
    """
    try:
        with gzip.open(shard_path, "rb") as gz:
            with tarfile.open(fileobj=gz, mode="r|") as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    try:
                        f = tf.extractfile(member)
                        if f is None:
                            continue
                        data = f.read()
                        yield member, data
                    except Exception:
                        # Truncated entry — stop reading this shard.
                        return
    except EOFError:
        # Truncated gzip footer — everything yielded so far is valid.
        return
    except Exception as exc:
        LOGGER.warning("Cannot open %s: %s", shard_path, exc)
        return


def repair_one_shard(args: tuple[Path, Path]) -> tuple[str, int, int, bool]:
    """Repair a single shard. Returns (relative_path, valid_files, dropped, was_truncated)."""
    shard_path, output_path = args
    output_path.parent.mkdir(parents=True, exist_ok=True)

    members_data: list[tuple[tarfile.TarInfo, bytes]] = []
    truncated = False

    # Read all valid members.
    for member, data in iter_tar_members(shard_path):
        members_data.append((member, data))

    # Check if the shard was truncated by comparing to the original size.
    # A fully intact shard will have its gzip footer; a truncated one won't.
    # We detect this indirectly: if the rewritten shard is significantly
    # smaller or if we got an EOFError during read.
    try:
        with gzip.open(shard_path, "rb") as gz:
            while gz.read(1 << 20):
                pass
    except (EOFError, OSError):
        truncated = True

    if not members_data:
        return str(shard_path), 0, 0, truncated

    # Write clean tar.gz.
    with gzip.open(output_path, "wb", compresslevel=6) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tf:
            for member, data in members_data:
                member.size = len(data)
                tf.addfile(member, io.BytesIO(data))

    return str(shard_path), len(members_data), 0, truncated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Root directory with .tar.gz shards (searched recursively).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for repaired shards (mirrors input layout).")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel worker processes (default: 8).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    shards = sorted(args.input.rglob("*.tar.gz"))
    if not shards:
        LOGGER.error("No shards found under %s", args.input)
        sys.exit(1)
    LOGGER.info("Found %d shards under %s", len(shards), args.input)

    # Build (input, output) pairs preserving the relative directory layout.
    tasks: list[tuple[Path, Path]] = []
    for shard in shards:
        rel = shard.relative_to(args.input)
        out = args.output / rel
        tasks.append((shard, out))

    total_files = 0
    truncated_count = 0
    with Pool(processes=args.workers) as pool:
        for i, (path, n_files, n_dropped, was_trunc) in enumerate(
            pool.imap_unordered(repair_one_shard, tasks)
        ):
            total_files += n_files
            if was_trunc:
                truncated_count += 1
            if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                LOGGER.info(
                    "[%d/%d] files=%d truncated_shards=%d",
                    i + 1, len(tasks), total_files, truncated_count,
                )

    LOGGER.info(
        "DONE: %d shards processed, %d were truncated, %d total files recovered",
        len(tasks), truncated_count, total_files,
    )


if __name__ == "__main__":
    main()
