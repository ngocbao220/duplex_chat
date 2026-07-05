#!/usr/bin/env python3
"""Reconstruct the DuplexChat audio dataset from the released metadata manifest.

The DuplexChat corpus is distributed as **metadata only** (see the HuggingFace
dataset ``sarulab-speech/DuplexChat``): each row points at a public source
episode (``audio_url``) and the dialogue span within it. This script turns that
metadata back into the actual audio dataset by, for every dialogue clip:

  1. downloading the source episode (once per ``audio_url``),
  2. transcoding it to 16 kHz mono,
  3. slicing the ``[episode_start_sec, episode_end_sec]`` span,
  4. running DialogueSidon two-speaker separation, and
  5. writing a stereo MP3 (L = speaker 0, R = speaker 1) into WebDataset shards
     matching the layout of the original dataset.

It reuses the construction pipeline (``duplexchat_pipe``) directly, so the output
is byte-faithful to the released corpus.

Requirements: a CUDA GPU, ffmpeg/ffprobe on PATH, a HuggingFace token with
access to the gated ``sarulab-speech/DialogueSidon`` weights (``huggingface-cli
login``). ``pyannote`` is *not* needed — diarization is not re-run, because the
dialogue spans are already in the manifest.

Distributed use: shard the manifest across N workers with
``--num-shards N --shard-index i``. Episodes are assigned to shards by a stable
hash of ``audio_url``, so every dialogue of an episode lands on the same worker
(each episode is downloaded exactly once). Output shards are written under
``<output>/<shard-index>/`` so workers never collide.

Example
-------
    python scripts/reconstruct_dataset.py \
        --manifest duplexchat_manifest_en.jsonl.gz \
        --output ./data/reconstructed_en \
        --num-shards 64 --shard-index 0
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import sqlite3
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

# Package modules (installed as `duplexchat-pipe`).
from duplexchat_pipe import audio, rss, separate, wds

LOGGER = logging.getLogger("reconstruct")

SR = 16_000  # pipeline transcodes every episode to 16 kHz mono before slicing


# ── resume ledger ──────────────────────────────────────────────────────────────

class DoneDB:
    """Tiny SQLite ledger of completed dialogue keys, for resumable runs."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS done (dlg_key TEXT PRIMARY KEY)"
        )
        self.conn.commit()
        self._cache = {
            row[0] for row in self.conn.execute("SELECT dlg_key FROM done")
        }

    def is_done(self, dlg_key: str) -> bool:
        return dlg_key in self._cache

    def mark(self, dlg_key: str) -> None:
        if dlg_key in self._cache:
            return
        self._cache.add(dlg_key)
        self.conn.execute("INSERT OR IGNORE INTO done VALUES (?)", (dlg_key,))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


# ── manifest helpers ────────────────────────────────────────────────────────────

def episode_key(audio_url: str) -> str:
    """Same key the pipeline uses: sha1 of the normalized audio URL."""
    return hashlib.sha1(rss.normalize_url(audio_url).encode("utf-8")).hexdigest()


def dialogue_key(audio_url: str, dialogue_idx) -> str:
    try:
        idx = int(dialogue_idx)
    except (TypeError, ValueError):
        idx = 0
    return f"{episode_key(audio_url)}_{idx:04d}"


def shard_of(audio_url: str, num_shards: int) -> int:
    """Stable assignment of an episode to a shard (independent of PYTHONHASHSEED)."""
    h = hashlib.blake2b(
        rss.normalize_url(audio_url).encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(h, "big") % num_shards


def load_episode_groups(
    manifest: Path, num_shards: int, shard_index: int
) -> dict[str, list[dict]]:
    """Stream the manifest, keep only rows for this shard, group by audio_url."""
    groups: dict[str, list[dict]] = defaultdict(list)
    n_total = n_mine = 0
    with gzip.open(manifest, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            row = json.loads(line)
            au = row.get("audio_url")
            if not au:
                continue
            if num_shards > 1 and shard_of(au, num_shards) != shard_index:
                continue
            groups[au].append(row)
            n_mine += 1
    LOGGER.info(
        "manifest %s: %d rows total, %d rows for shard %d/%d across %d episodes",
        manifest.name, n_total, n_mine, shard_index, num_shards, len(groups),
    )
    return groups


def build_meta(row: dict, sep_sr: int) -> dict:
    """Rebuild a per-clip meta.json from a manifest row (mirrors the pipeline).

    Per-clip diarization segments are intentionally absent from the released
    metadata, so no diarization.json is emitted.
    """
    return {
        "rss_url": row.get("rss_url"),
        "audio_url": row.get("audio_url"),
        "language": row.get("language"),
        "episode_duration_sec": row.get("episode_duration_sec"),
        "dialogue_idx": row.get("dialogue_idx"),
        "dialogue_start": row.get("episode_start_sec"),
        "dialogue_end": row.get("episode_end_sec"),
        "dialogue_duration_sec": row.get("duration_sec"),
        "dialogue_speakers": row.get("speakers"),
        "separated": True,
        "separation_model": separate.REPO_ID,
        "separation_sample_rate": sep_sr,
        "channels": 2,
        "reconstructed": True,
    }


# ── shard writer with resume-safe numbering ──────────────────────────────────────

def open_writer(out_dir: Path, shard_size_gb: float):
    """WebDataset ShardWriter that appends after any pre-existing shards."""
    import webdataset as wds_lib

    out_dir.mkdir(parents=True, exist_ok=True)
    start_shard = len(list(out_dir.glob("[0-9]" * 6 + ".tar.gz")))
    pattern = str(out_dir / "%06d.tar.gz")
    return wds_lib.ShardWriter(
        pattern, maxsize=int(shard_size_gb * 1024 ** 3), start_shard=start_shard
    )


# ── main ─────────────────────────────────────────────────────────────────────────

def process_episode(
    audio_url: str,
    rows: list[dict],
    *,
    models: dict,
    writer,
    done: DoneDB,
    scratch: Path,
    num_steps: int,
    bitrate_kbps: int,
    timeout_seconds: int,
    keep_audio: bool,
) -> tuple[int, int]:
    """Reconstruct every pending dialogue of one episode. Returns (written, failed)."""
    pending = [r for r in rows if not done.is_done(dialogue_key(audio_url, r.get("dialogue_idx")))]
    if not pending:
        return 0, 0

    key = episode_key(audio_url)
    ext = Path(rss.normalize_url(audio_url).split("?")[0]).suffix or ".bin"
    raw_path = scratch / f"{key}{ext}"
    wav_path = scratch / f"{key}.16k.wav"
    written = failed = 0

    try:
        audio.download_audio(audio_url, raw_path, timeout_seconds)
        audio.transcode_to_wav_16k_mono(raw_path, wav_path)
        wav_tensor, _ = audio.load_wav_tensor(wav_path)
    except Exception as exc:  # dead URL, unreadable audio, etc. — skip the episode
        LOGGER.warning("episode %s download/transcode failed: %s", audio_url, exc)
        _cleanup(raw_path, wav_path, keep_audio)
        return 0, len(pending)

    for row in pending:
        dlg_key = dialogue_key(audio_url, row.get("dialogue_idx"))
        try:
            start = float(row["episode_start_sec"])
            end = float(row["episode_end_sec"])
            dlg_wav = wav_tensor[:, int(start * SR):int(end * SR)].clone()
            if dlg_wav.shape[-1] <= 0:
                raise ValueError(f"empty span [{start}, {end}]")
            spk0, spk1, sep_sr = separate.run_separation(dlg_wav, SR, num_steps, models)
            audio_bytes = audio.tensors_to_stereo_mp3_bytes(spk0, spk1, sep_sr, bitrate_kbps)
            wds.write_sample(
                writer,
                key=dlg_key,
                audio_bytes=audio_bytes,
                meta=build_meta(row, sep_sr),
                diarization=None,
                diarization_error=None,
            )
            done.mark(dlg_key)
            written += 1
        except Exception as exc:
            LOGGER.warning("dialogue %s failed: %s", dlg_key, exc)
            failed += 1

    del wav_tensor
    _cleanup(raw_path, wav_path, keep_audio)
    return written, failed


def _cleanup(raw_path: Path, wav_path: Path, keep_audio: bool) -> None:
    if keep_audio:
        return
    for p in (raw_path, wav_path):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


DEFAULT_MANIFEST = {
    "en": "duplexchat_manifest_en.jsonl.gz",
    "ja": "duplexchat_manifest_ja.jsonl.gz",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, help="Path to a manifest .jsonl.gz.")
    ap.add_argument("--language", choices=sorted(DEFAULT_MANIFEST),
                    help="Pick the standard manifest filename (with --manifest-dir).")
    ap.add_argument("--manifest-dir", type=Path, default=Path("."),
                    help="Directory holding the standard manifest files (with --language).")
    ap.add_argument("--output", type=Path, required=True, help="Output dataset directory.")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-steps", type=int, default=30, help="Diffusion steps (default 30).")
    ap.add_argument("--mp3-bitrate-kbps", type=int, default=128)
    ap.add_argument("--shard-size-gb", type=float, default=3.0)
    ap.add_argument("--timeout-seconds", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scratch-dir", type=Path, default=None,
                    help="Fast local dir for temp episode audio (default: a temp dir).")
    ap.add_argument("--limit-episodes", type=int, default=0,
                    help="Process at most N episodes (0 = all). Useful for smoke tests.")
    ap.add_argument("--keep-audio", action="store_true",
                    help="Keep downloaded/transcoded source audio instead of deleting it.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.manifest:
        manifest = args.manifest
    elif args.language:
        manifest = args.manifest_dir / DEFAULT_MANIFEST[args.language]
    else:
        ap.error("provide --manifest PATH or --language {en,ja} [--manifest-dir DIR]")
    if not manifest.exists():
        ap.error(f"manifest not found: {manifest}")
    if not (0 <= args.shard_index < args.num_shards):
        ap.error("--shard-index must be in [0, --num-shards)")

    audio.ensure_ffmpeg()

    out_dir = args.output / f"{args.shard_index:03d}"
    done = DoneDB(args.output / "cache" / f"done_{args.shard_index:03d}.sqlite")

    groups = load_episode_groups(manifest, args.num_shards, args.shard_index)
    episodes = sorted(groups)
    if args.limit_episodes:
        episodes = episodes[: args.limit_episodes]
        LOGGER.info("limiting to first %d episodes", len(episodes))

    LOGGER.info("loading DialogueSidon separation models on %s ...", args.device)
    models = separate.load_separation_models(args.device)

    scratch_ctx = None
    if args.scratch_dir:
        args.scratch_dir.mkdir(parents=True, exist_ok=True)
        scratch = args.scratch_dir
    else:
        scratch_ctx = tempfile.TemporaryDirectory(prefix="duplexchat_recon_")
        scratch = Path(scratch_ctx.name)

    writer = open_writer(out_dir, args.shard_size_gb)
    n_written = n_failed = n_ep = 0
    t0 = time.monotonic()
    try:
        for audio_url in episodes:
            w, f = process_episode(
                audio_url, groups[audio_url],
                models=models, writer=writer, done=done, scratch=scratch,
                num_steps=args.num_steps, bitrate_kbps=args.mp3_bitrate_kbps,
                timeout_seconds=args.timeout_seconds, keep_audio=args.keep_audio,
            )
            n_written += w
            n_failed += f
            n_ep += 1
            if n_ep % 20 == 0:
                LOGGER.info(
                    "%d/%d episodes, %d clips written, %d failed, %.1f min elapsed",
                    n_ep, len(episodes), n_written, n_failed, (time.monotonic() - t0) / 60,
                )
    finally:
        writer.close()
        done.close()
        if scratch_ctx is not None:
            scratch_ctx.cleanup()

    LOGGER.info(
        "done: %d episodes, %d clips written, %d failed -> %s",
        n_ep, n_written, n_failed, out_dir,
    )
    if n_written == 0 and n_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
