#!/usr/bin/env python3
"""Merge per-task manifest parts into final per-language manifests.

Streams all manifest_part_*.jsonl.gz, deduplicates globally by
(audio_url, dialogue_idx) via a 64-bit hash set, and writes one gzipped JSONL
per language plus a counts summary.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
from collections import defaultdict


def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "replace"),
                                          digest_size=8).digest(), "big")


def base(lang: str) -> str:
    if lang and lang.startswith("ja"):
        return "ja"
    if lang and lang.startswith("en"):
        return "en"
    return lang or "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    parts = sorted(glob.glob(os.path.join(args.parts_dir, "manifest_part_*.jsonl.gz")))
    if not parts:
        raise SystemExit(f"no manifest_part_*.jsonl.gz in {args.parts_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    writers = {}
    seen: set[int] = set()
    counts = defaultdict(int)
    dur = defaultdict(float)
    dropped = 0

    for p in parts:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                k = h64(f"{r.get('audio_url')}|{r.get('dialogue_idx')}")
                if k in seen:
                    dropped += 1
                    continue
                seen.add(k)
                b = base(r.get("language"))
                if b not in writers:
                    writers[b] = gzip.open(
                        os.path.join(args.out_dir, f"duplexchat_manifest_{b}.jsonl.gz"),
                        "wt", encoding="utf-8")
                writers[b].write(line if line.endswith("\n") else line + "\n")
                counts[b] += 1
                dur[b] += float(r.get("duration_sec") or 0.0)
        print(f"merged {os.path.basename(p)} (running unique={len(seen):,})", flush=True)

    for w in writers.values():
        w.close()

    summary = {b: {"rows": counts[b], "hours": dur[b] / 3600.0} for b in counts}
    summary["_cross_task_duplicates_dropped"] = dropped
    print(json.dumps(summary, indent=2))
    with open(os.path.join(args.out_dir, "manifest_counts.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
