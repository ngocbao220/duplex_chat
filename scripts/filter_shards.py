"""Filter and dedup webdataset shards.

Reads all shards under --input, drops samples that are:
  - music (feed/entry tag term matches a music keyword)
  - too long (dialogue_duration_sec >= 1/3 of episode_duration_sec)
  - duplicate (same (audio_url, dialogue_idx) seen earlier)

and writes survivors as fresh webdataset shards to --output. Writes a
statistics JSON report alongside the output shards.

Run after the podcast crawl finishes:

    uv run python3 scripts/filter_shards.py \
        --input data/wds_ja \
        --output data/wds_ja_filtered \
        --shard-size-gb 3.0
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import queue
import sys
import tarfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Make src/ importable so we can reuse open_shard_writer / write_sample.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from duplexchat_pipe.wds import open_shard_writer, write_sample  # noqa: E402
from duplexchat_pipe.tags import is_music_feed  # noqa: E402


LOGGER = logging.getLogger("filter_shards")


def is_music(meta: dict) -> bool:
    """True if the sample's feed or entry metadata is tagged as music."""
    return is_music_feed(meta.get("feed"), meta.get("entry"))


def is_too_long(meta: dict) -> bool:
    ep = meta.get("episode_duration_sec") or 0.0
    dlg = meta.get("dialogue_duration_sec") or 0.0
    if ep <= 0:
        return False
    return dlg / ep >= 1.0 / 3.0


def sample_tag_terms(meta: dict) -> set[str]:
    terms: set[str] = set()
    for section in ("feed", "entry"):
        for t in (meta.get(section) or {}).get("tags") or []:
            term = (t.get("term") or "").strip()
            if term:
                terms.add(term)
    return terms


def read_shard(shard_path: Path) -> list[dict]:
    """Read one .tar.gz shard using raw gzip + tarfile.

    More tolerant than webdataset: on a truncated gzip stream, yields all
    valid samples up to the corruption point instead of aborting the whole
    shard. Returns a list of dicts, each keyed by file extension (e.g.
    ``{"meta.json": b"...", "audio.mp3": b"...", "__key__": b"..."}``.
    """
    samples: list[dict] = []
    try:
        with gzip.open(shard_path, "rb") as gz:
            with tarfile.open(fileobj=gz, mode="r|") as tf:
                current: dict[str, bytes] = {}
                current_key: str | None = None
                for member in tf:
                    if not member.isfile():
                        continue
                    try:
                        f = tf.extractfile(member)
                        if f is None:
                            continue
                        data = f.read()
                    except Exception:
                        break  # Truncated entry — stop this shard.

                    # Parse webdataset naming: <key>.<ext> where ext can
                    # contain dots (e.g. "audio.mp3", "meta.json").
                    name = member.name
                    dot = name.find(".")
                    if dot < 0:
                        continue
                    key = name[:dot]
                    ext = name[dot + 1:]  # e.g. "audio.mp3", "meta.json"

                    if key != current_key:
                        if current_key is not None and current:
                            current["__key__"] = current_key.encode()
                            samples.append(current)
                        current = {}
                        current_key = key

                    current[ext] = data

                # Flush last sample.
                if current_key is not None and current:
                    current["__key__"] = current_key.encode()
                    samples.append(current)
    except (EOFError, OSError):
        pass  # Truncated gzip footer — samples collected so far are valid.
    return samples


def iter_shards(input_dir: Path):
    # Accept either a directory that directly contains tar.gz files, or a
    # nested layout like data/wds_ja/{JOB_ID}/{NODE_INDEX}/NNNNNN.tar.gz.
    shards = sorted(input_dir.rglob("*.tar.gz"))
    return shards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Input directory with webdataset shards (searched recursively).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for filtered shards.")
    parser.add_argument("--shard-size-gb", type=float, default=3.0)
    parser.add_argument("--stats-json", type=Path, default=None,
                        help="Where to write the JSON stats report (default: <output>/stats.json).")
    parser.add_argument("--progress-every", type=int, default=10_000,
                        help="Print progress every N samples (default: 10000).")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Unused (kept for CLI compatibility).")
    parser.add_argument("--write-queue-size", type=int, default=64,
                        help="Max pending samples per writer thread (default: 64).")
    parser.add_argument("--num-writers", type=int, default=8,
                        help="Number of parallel writer threads (default: 8). "
                             "Each writes to its own subdirectory. More writers "
                             "= better Lustre OST distribution.")
    parser.add_argument("--shard", type=int, default=0,
                        help="This rank's index when sharding inputs across nodes (0-based).")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of ranks sharding the input. When >1, this "
                             "rank processes inputs[shard::num_shards] and dedup is "
                             "disabled (each rank only sees its slice).")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Skip cross-shard (audio_url, dialogue_idx) dedup.")
    args = parser.parse_args()

    skip_dedup = args.no_dedup or args.num_shards > 1

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    shards = iter_shards(args.input)
    if not shards:
        LOGGER.error("No shards found under %s", args.input)
        sys.exit(1)
    LOGGER.info("Found %d shards under %s", len(shards), args.input)

    if args.num_shards > 1:
        shards = shards[args.shard :: args.num_shards]
        output_dir = args.output / f"n{args.shard:03d}"
        default_stats = args.output / f"stats_n{args.shard:03d}.json"
        LOGGER.info("rank %d/%d: processing %d shards (dedup disabled)",
                    args.shard, args.num_shards, len(shards))
    else:
        output_dir = args.output
        default_stats = args.output / "stats.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.stats_json or default_stats

    seen: set[tuple[str, int]] = set()
    counts = {"total": 0, "music": 0, "dup": 0, "too_long": 0, "kept": 0}
    durations = {"total": 0.0, "music": 0.0, "dup": 0.0, "too_long": 0.0, "kept": 0.0}
    kept_tags: Counter[str] = Counter()
    music_tags: Counter[str] = Counter()
    dup_tags: Counter[str] = Counter()
    too_long_tags: Counter[str] = Counter()

    # ── Parallel writers ────────────────────────────────────────────────
    # N writer threads, each with its own ShardWriter and queue.  gzip
    # compression in zlib releases the GIL, so N threads compress in
    # parallel.  Main thread round-robins kept samples across queues.
    num_writers = args.num_writers
    _SENTINEL = object()
    writer_errors: list[BaseException] = []

    writers = []
    write_queues: list[queue.Queue] = []
    writer_threads: list[threading.Thread] = []

    for wi in range(num_writers):
        out_dir = output_dir / f"w{wi}"
        out_dir.mkdir(parents=True, exist_ok=True)
        w = open_shard_writer(out_dir, args.shard_size_gb)
        q: queue.Queue = queue.Queue(maxsize=args.write_queue_size)
        writers.append(w)
        write_queues.append(q)

        def _writer_loop(writer=w, wq=q) -> None:
            try:
                while True:
                    item = wq.get()
                    if item is _SENTINEL:
                        return
                    key, audio_bytes, meta, diarization = item
                    write_sample(writer, key, audio_bytes, meta, diarization, None)
            except BaseException as exc:  # noqa: BLE001
                writer_errors.append(exc)
                while True:
                    try:
                        if wq.get_nowait() is _SENTINEL:
                            return
                    except queue.Empty:
                        return

        t = threading.Thread(target=_writer_loop, name=f"shard-writer-{wi}", daemon=True)
        t.start()
        writer_threads.append(t)

    next_writer = 0

    try:
        for shard_idx, shard_path in enumerate(shards):
            try:
                samples = read_shard(shard_path)
            except Exception as exc:
                LOGGER.warning("shard [%d/%d] %s failed: %s", shard_idx + 1, len(shards), shard_path, exc)
                continue
            if (shard_idx + 1) % 50 == 0 or shard_idx == 0:
                LOGGER.info("shard [%d/%d] %d samples, total_kept=%d kept_hours=%.1f",
                            shard_idx + 1, len(shards), len(samples),
                            counts["kept"], durations["kept"] / 3600)

            for sample in samples:
                    meta_raw = sample.get("meta.json") or sample.get(b"meta.json")
                    if not meta_raw:
                        continue
                    if isinstance(meta_raw, bytes):
                        meta_raw = meta_raw.decode("utf-8", errors="replace")
                    try:
                        meta = json.loads(meta_raw)
                    except Exception:
                        continue

                    dlg_dur = float(meta.get("dialogue_duration_sec") or 0.0)
                    counts["total"] += 1
                    durations["total"] += dlg_dur

                    terms = sample_tag_terms(meta)

                    if not skip_dedup:
                        key = (meta.get("audio_url", ""), int(meta.get("dialogue_idx", -1)))
                        if key in seen:
                            counts["dup"] += 1
                            durations["dup"] += dlg_dur
                            for t in terms:
                                dup_tags[t] += 1
                            continue
                        seen.add(key)

                    if is_music(meta):
                        counts["music"] += 1
                        durations["music"] += dlg_dur
                        for t in terms:
                            music_tags[t] += 1
                        continue

                    if is_too_long(meta):
                        counts["too_long"] += 1
                        durations["too_long"] += dlg_dur
                        for t in terms:
                            too_long_tags[t] += 1
                        continue

                    for t in terms:
                        kept_tags[t] += 1

                    audio_bytes = (
                        sample.get("audio.mp3") or sample.get("mp3") or
                        sample.get(b"audio.mp3") or sample.get(b"mp3")
                    )
                    diarization = None
                    diar_raw = (
                        sample.get("diarization.json") or
                        sample.get(b"diarization.json")
                    )
                    if diar_raw:
                        try:
                            if isinstance(diar_raw, bytes):
                                diar_raw = diar_raw.decode("utf-8", errors="replace")
                            diarization = json.loads(diar_raw)
                        except Exception:
                            diarization = None

                    if writer_errors:
                        raise writer_errors[0]

                    sample_key = sample.get("__key__", b"")
                    if isinstance(sample_key, bytes):
                        sample_key = sample_key.decode()
                    sample_key = sample_key or f"sample_{counts['kept']:010d}"

                    write_queues[next_writer].put((
                        sample_key,
                        audio_bytes,
                        meta,
                        diarization,
                    ))
                    next_writer = (next_writer + 1) % num_writers
                    counts["kept"] += 1
                    durations["kept"] += dlg_dur

                    if counts["total"] % args.progress_every == 0:
                        qsizes = [q.qsize() for q in write_queues]
                        LOGGER.info(
                            "progress total=%d kept=%d music=%d dup=%d too_long=%d "
                            "kept_hours=%.1f qsizes=%s",
                            counts["total"], counts["kept"], counts["music"],
                            counts["dup"], counts["too_long"],
                            durations["kept"] / 3600,
                            qsizes,
                        )
    finally:
        for q in write_queues:
            q.put(_SENTINEL)
        for t in writer_threads:
            t.join()
        for w in writers:
            w.close()
        if writer_errors:
            raise writer_errors[0]

    rejected_count = counts["music"] + counts["dup"] + counts["too_long"]
    rejected_hours = (
        durations["music"] + durations["dup"] + durations["too_long"]
    ) / 3600

    report = {
        "counts": counts,
        "counts_rejected_total": rejected_count,
        "hours": {k: v / 3600 for k, v in durations.items()},
        "hours_rejected_total": rejected_hours,
        "top_kept_tags": kept_tags.most_common(50),
        "top_music_tags": music_tags.most_common(50),
        "top_dup_tags": dup_tags.most_common(50),
        "top_too_long_tags": too_long_tags.most_common(50),
    }

    with open(stats_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)

    LOGGER.info("=" * 60)
    LOGGER.info("DONE")
    LOGGER.info("total=%d kept=%d rejected=%d (music=%d dup=%d too_long=%d)",
                counts["total"], counts["kept"], rejected_count,
                counts["music"], counts["dup"], counts["too_long"])
    LOGGER.info("hours: kept=%.1fh rejected=%.1fh (music=%.1fh dup=%.1fh too_long=%.1fh)",
                durations["kept"] / 3600, rejected_hours,
                durations["music"] / 3600, durations["dup"] / 3600,
                durations["too_long"] / 3600)
    LOGGER.info("stats written to %s", stats_path)


if __name__ == "__main__":
    main()
