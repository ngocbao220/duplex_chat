#!/usr/bin/env python3
"""Export the reconstruction manifest (no audio) for copyright-safe release.

One PBS array task scans shards[task_id::num_tasks] and writes a gzipped JSONL
of one row per dialogue clip, deduplicated within the task by
(audio_url, dialogue_idx). The reduce step dedups globally across tasks.

Row schema (sufficient to re-derive each clip by re-running the pipeline):
  {language, rss_url, audio_url, dialogue_idx,
   episode_start_sec, episode_end_sec, duration_sec,
   speakers, episode_duration_sec}
episode_start/end are the dialogue span measured within the source episode.
"""
from __future__ import annotations

import argparse
import gzip
import json
import tarfile
from multiprocessing import Pool
from pathlib import Path


def scan_shard(path_str: str) -> list[dict]:
    path = Path(path_str)
    rows: list[dict] = []
    seen: set = set()
    try:
        with gzip.open(path, "rb") as gz, tarfile.open(fileobj=gz, mode="r|") as tf:
            for m in tf:
                if not m.isfile() or not m.name.endswith(".meta.json"):
                    continue
                try:
                    f = tf.extractfile(m)
                    if not f:
                        continue
                    meta = json.loads(f.read())
                except Exception:
                    break
                au = meta.get("audio_url")
                idx = meta.get("dialogue_idx")
                if not au:
                    continue
                key = (au, idx)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "language": meta.get("language"),
                    "rss_url": meta.get("rss_url"),
                    "audio_url": au,
                    "dialogue_idx": idx,
                    "episode_start_sec": meta.get("dialogue_start"),
                    "episode_end_sec": meta.get("dialogue_end"),
                    "duration_sec": meta.get("dialogue_duration_sec"),
                    "speakers": meta.get("dialogue_speakers"),
                    "episode_duration_sec": meta.get("episode_duration_sec"),
                })
    except (EOFError, OSError):
        pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="Dataset root containing wds_ja_filtered/ and wds_en_filtered/.")
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--num-tasks", type=int, required=True)
    ap.add_argument("--procs", type=int, default=64)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    root = Path(args.data_root)
    shards = []
    for lang_dir in ("wds_ja_filtered", "wds_en_filtered"):
        shards.extend(sorted((root / lang_dir).rglob("*.tar.gz")))
    shards = [str(s) for s in shards]
    mine = shards[args.task_id :: args.num_tasks]
    print(f"task {args.task_id}/{args.num_tasks}: {len(mine)} of {len(shards)} shards", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"manifest_part_{args.task_id:03d}.jsonl.gz"
    n = 0
    done = 0
    with gzip.open(out_path, "wt", encoding="utf-8") as out, Pool(args.procs) as pool:
        for rows in pool.imap_unordered(scan_shard, mine, chunksize=1):
            for r in rows:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(mine)} shards, {n:,} rows", flush=True)
    print(f"wrote {out_path}: {n:,} rows (within-task deduped)", flush=True)


if __name__ == "__main__":
    main()
