from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import urlparse

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import torch
from tqdm import tqdm

from duplexchat_pipe import (
    audio,
    db,
    dialogue as dialogue_mod,
    diarize,
    outputs as outputs_mod,
    rss,
    separate as separate_mod,
    sources,
    tags as tag_filter,
    wds,
    youtube,
)
from duplexchat_pipe.devices import pick_task_device, resolve_device, validate_multi_gpu
from duplexchat_pipe.logging_utils import (
    append_error,
    append_phase_tree,
    append_stats_table,
    setup_run_logging,
    write_artifacts,
    write_resolved_config,
)

REPO_ID_SIDON = "sarulab-speech/DialogueSidon"
from duplexchat_pipe.config import Config


LOGGER = logging.getLogger(__name__)
DB_COMMIT_INTERVAL = 200
FEED_FUTURE_BUFFER_MULTIPLIER = 4
TASK_BUFFER_MULTIPLIER = 4


@dataclass
class AudioItem:
    audio_url: str
    rss_url: str
    language: str
    feed_meta: dict
    entry_meta: dict
    source_type: str = "rss"


@dataclass
class ProcessedResult:
    key: str
    status: str
    meta: dict | None
    audio_bytes: bytes | None
    diarization: dict | None
    diarization_error: dict | None
    error: str | None = None
    raw_path: Path | None = None
    # When True, this result only marks the episode key in ProcessedDB and is not
    # written to the WebDataset (used when an episode produces multiple dialogue samples).
    is_marker: bool = False


@dataclass
class DownloadedItem:
    item: AudioItem
    key: str
    raw_path: Path
    duration: float
    sample_rate: int | None = None
    bit_rate: int | None = None


@dataclass
class SeparationTask:
    """A single dialogue to be separated, produced by the diarize stage."""
    dlg_key: str
    episode_key: str
    dlg_wav: torch.Tensor  # (1, T) float32 at 16kHz
    dlg_start: float
    dlg_end: float
    dialogue: object  # dialogue_mod.Dialogue
    item: AudioItem
    episode_duration: float
    cfg: Config
    device: str


@dataclass
class DiarizeResult:
    """Output of the diarize stage for one episode."""
    episode_key: str
    item: AudioItem
    duration: float
    raw_path: Path
    separation_tasks: list[SeparationTask] = field(default_factory=list)
    # Non-separation dialogue results (when separation is disabled)
    results: list[ProcessedResult] = field(default_factory=list)
    status: str = "ok"
    error: str | None = None


@dataclass
class Stats:
    total_duration_sec: float = 0.0
    written: int = 0
    skipped: int = 0
    errors: int = 0


class ProcessedDB:
    def __init__(self, path: Path, commit_interval: int = DB_COMMIT_INTERVAL) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.commit_interval = max(1, commit_interval)
        self.pending_writes = 0
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS processed (
                url_hash TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                error TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def get_status(self, url_hash: str) -> str | None:
        cursor = self.conn.execute(
            "SELECT status FROM processed WHERE url_hash = ?", (url_hash,)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def is_done(self, url_hash: str) -> bool:
        status = self.get_status(url_hash)
        return status in {
            "ok",
            "skipped_duration",
            "skipped_no_dialogues",
            "skipped_low_quality",
        }

    def mark(self, url_hash: str, status: str, error: str | None = None) -> None:
        updated_at = dt.datetime.utcnow().isoformat()
        self.conn.execute(
            """
            INSERT INTO processed (url_hash, status, error, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(url_hash) DO UPDATE SET
                status=excluded.status,
                error=excluded.error,
                updated_at=excluded.updated_at
            """,
            (url_hash, status, error, updated_at),
        )
        self.pending_writes += 1
        if self.pending_writes >= self.commit_interval:
            self.flush()

    def flush(self) -> None:
        if self.pending_writes > 0:
            self.conn.commit()
            self.pending_writes = 0

    def close(self) -> None:
        self.flush()
        self.conn.close()


def _hash_url(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def _scratch_dir(cfg: Config) -> Path:
    """Return the fast scratch directory (local NVMe if available, else cache_dir)."""
    if cfg.scratch_dir is not None:
        cfg.scratch_dir.mkdir(parents=True, exist_ok=True)
        return cfg.scratch_dir
    return cfg.cache_dir


def _target_hours_reached(cfg: Config, stats: Stats) -> bool:
    return cfg.target_hours is not None and stats.total_duration_sec / 3600 >= cfg.target_hours


def _metadata_language(cfg: Config, source_language: str | None) -> str:
    return str(cfg.metadata_language or source_language or "vi").lower()


def _collect_feed_items(
    rss_url: str,
    language: str,
    cfg: Config,
) -> list[AudioItem]:
    feed = rss.fetch_rss(rss_url, cfg.timeout_seconds)
    feed_meta = rss.to_jsonable(feed.get("feed", {}))

    # Skip entire feed if it's tagged as music — saves downloads and GPU time.
    if tag_filter.is_music_feed(feed_meta):
        LOGGER.info("Skipping music feed: %s", rss_url)
        return []

    items: list[AudioItem] = []
    entry_count = 0
    for entry in feed.get("entries", []) or []:
        if cfg.episode_limit_per_feed is not None and entry_count >= cfg.episode_limit_per_feed:
            break
        entry_count += 1
        entry_meta = rss.to_jsonable(entry)
        # Per-entry music check — rare but some feeds tag individual episodes.
        if tag_filter.is_music_feed(None, entry_meta):
            continue
        for audio_url in rss._iter_entry_audio_urls(entry):
            if not rss.is_audio_url(audio_url):
                continue
            items.append(
                AudioItem(
                    audio_url=audio_url,
                    rss_url=rss_url,
                    language=language,
                    feed_meta=feed_meta,
                    entry_meta=entry_meta,
                )
            )
    return items


def _download_audio_item(
    item: AudioItem,
    cfg: Config,
    key: str,
) -> DownloadedItem | ProcessedResult:
    parsed = urlparse(item.audio_url)
    suffix = Path(parsed.path).suffix
    if not suffix:
        suffix = ".bin"

    # Download to local NVMe scratch for speed
    scratch = _scratch_dir(cfg)
    audio_dir = scratch / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    raw_path = audio_dir / f"{key}{suffix}"

    try:
        t0 = time.monotonic()
        if item.source_type == "youtube":
            raw_path = youtube.download_audio(item.audio_url, audio_dir, key)
        else:
            audio.download_audio(item.audio_url, raw_path, cfg.timeout_seconds)
        t_download = time.monotonic() - t0
        info = audio.probe_audio_info(raw_path)
        duration = info.duration
        LOGGER.info("PROFILE download %s %.2fs (%.1fs audio)", key, t_download, duration)
        if duration <= 0:
            return ProcessedResult(
                key=key, status="error", meta=None, audio_bytes=None,
                diarization=None, diarization_error=None,
                error="ffprobe returned empty duration", raw_path=raw_path,
            )
        if duration > cfg.max_duration_minutes * 60:
            return ProcessedResult(
                key=key, status="skipped_duration", meta=None, audio_bytes=None,
                diarization=None, diarization_error=None,
                error=f"duration {duration:.2f}s exceeds limit", raw_path=raw_path,
            )
        if info.sample_rate is not None and info.sample_rate < cfg.min_original_sample_rate:
            return ProcessedResult(
                key=key, status="skipped_low_quality", meta=None, audio_bytes=None,
                diarization=None, diarization_error=None,
                error=f"sample_rate {info.sample_rate} below {cfg.min_original_sample_rate}",
                raw_path=raw_path,
            )
        if info.bit_rate is not None and info.bit_rate < cfg.min_original_bitrate:
            return ProcessedResult(
                key=key, status="skipped_low_quality", meta=None, audio_bytes=None,
                diarization=None, diarization_error=None,
                error=f"bit_rate {info.bit_rate} below {cfg.min_original_bitrate}",
                raw_path=raw_path,
            )
        return DownloadedItem(
            item=item, key=key, raw_path=raw_path, duration=duration,
            sample_rate=info.sample_rate, bit_rate=info.bit_rate,
        )
    except Exception as exc:  # noqa: BLE001
        return ProcessedResult(
            key=key, status="error", meta=None, audio_bytes=None,
            diarization=None, diarization_error=None, error=str(exc), raw_path=raw_path,
        )


def _make_error_result(key: str, error: str, raw_path: Path | None) -> ProcessedResult:
    return ProcessedResult(
        key=key, status="error", meta=None, audio_bytes=None,
        diarization=None, diarization_error=None, error=error, raw_path=raw_path,
    )


def _debug_episode_dir(cfg: Config, episode_key: str) -> Path:
    return Path(cfg.debug_outputs_dir) / outputs_mod.safe_name(episode_key)


def _write_debug_diarization_outputs(
    cfg: Config,
    episode_key: str,
    wav_path: Path,
    segments: list[dict],
    valid_dialogues: list,
    duration_sec: float,
) -> None:
    if not cfg.debug_outputs_enabled:
        return
    try:
        out_dir = _debug_episode_dir(cfg, episode_key)
        outputs_mod.write_json(
            out_dir / "phase_00_input" / "input.json",
            {"episode_key": episode_key},
        )
        outputs_mod.copy_file(
            wav_path,
            out_dir / "phase_01_preprocess" / "audio_16k_mono.wav",
        )
        outputs_mod.write_diarization_phase(
            out_dir,
            segments,
            duration_sec=duration_sec,
            model=cfg.diarization_model,
            backend=cfg.diarization_backend,
        )
        outputs_mod.write_dialogues_phase(out_dir, valid_dialogues)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to write debug diarization outputs for %s: %s", episode_key, exc)


def _write_debug_separation_outputs(
    task: SeparationTask,
    spk0: torch.Tensor,
    spk1: torch.Tensor,
    sample_rate: int,
    model_id: str | None,
) -> None:
    cfg = task.cfg
    if not cfg.debug_outputs_enabled:
        return
    try:
        phase_dir = _debug_episode_dir(cfg, task.episode_key) / "phase_04_separation" / task.dlg_key
        outputs_mod.save_wav(phase_dir / "speaker_A.wav", spk0, sample_rate)
        outputs_mod.save_wav(phase_dir / "speaker_B.wav", spk1, sample_rate)
        outputs_mod.write_json(
            phase_dir / "separation.json",
            {
                "dialogue_key": task.dlg_key,
                "dialogue_start": task.dlg_start,
                "dialogue_end": task.dlg_end,
                "backend": cfg.separation_backend,
                "model": model_id,
                "sample_rate": sample_rate,
                "device": task.device,
            },
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to write debug separation outputs for %s: %s", task.dlg_key, exc)


# ── Stage 1: Diarize ────────────────────────────────────────────────────────

def _diarize_episode(
    downloaded: DownloadedItem,
    cfg: Config,
    diarization_pipeline,
    diarize_lock: threading.Lock | None,
    task_device_ids: list[int] | None = None,
) -> DiarizeResult:
    """Transcode → diarize → extract dialogues → load WAV tensor. No separation."""
    key = downloaded.key
    item = downloaded.item
    duration = downloaded.duration

    scratch = _scratch_dir(cfg)
    tmp_dir = scratch / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    wav_path = tmp_dir / f"{key}.wav"

    try:
        t0 = time.monotonic()
        audio.transcode_to_wav_16k_mono(downloaded.raw_path, wav_path)
        t_transcode = time.monotonic() - t0

        try:
            t0 = time.monotonic()
            if diarize_lock is None:
                segments = diarize.run_diarization(diarization_pipeline, wav_path)
            else:
                with diarize_lock:
                    segments = diarize.run_diarization(diarization_pipeline, wav_path)
            t_diarize = time.monotonic() - t0
        except Exception as exc:  # noqa: BLE001
            return DiarizeResult(
                episode_key=key, item=item, duration=duration,
                raw_path=downloaded.raw_path, status="error",
                error=f"diarization failed: {exc}",
            )

        valid_dialogues = dialogue_mod.extract_valid_dialogues(
            segments,
            gap_seconds=cfg.dialogue_gap_seconds,
            max_single_speaker_ratio=cfg.dialogue_max_single_speaker_ratio,
            min_duration_seconds=cfg.dialogue_min_duration_seconds,
            max_duration_seconds=cfg.dialogue_max_duration_seconds,
        )
        _write_debug_diarization_outputs(
            cfg,
            key,
            wav_path,
            segments,
            valid_dialogues,
            duration,
        )

        if len(valid_dialogues) < 4:
            LOGGER.info(
                "PROFILE %s dur=%.1fs transcode=%.2fs diarize=%.2fs dialogues=%d (skipped)",
                key, duration, t_transcode, t_diarize, len(valid_dialogues),
            )
            return DiarizeResult(
                episode_key=key, item=item, duration=duration,
                raw_path=downloaded.raw_path, status="skipped_no_dialogues",
            )

        # Build separation tasks or direct results
        result = DiarizeResult(
            episode_key=key, item=item, duration=duration,
            raw_path=downloaded.raw_path,
        )

        if cfg.enable_separation:
            t0 = time.monotonic()
            wav_tensor, _ = audio.load_wav_tensor(wav_path)
            t_load_wav = time.monotonic() - t0

            for idx, dlg in enumerate(valid_dialogues):
                dlg_key = f"{key}_{idx:04d}"
                task_device = pick_task_device(
                    cfg.diarization_device, idx, task_device_ids or []
                )
                start_sample = int(dlg.start * 16_000)
                end_sample = int(dlg.end * 16_000)
                # Clone the slice so the full wav_tensor can be freed
                dlg_wav = wav_tensor[:, start_sample:end_sample].clone()
                result.separation_tasks.append(SeparationTask(
                    dlg_key=dlg_key, episode_key=key,
                    dlg_wav=dlg_wav, dlg_start=dlg.start, dlg_end=dlg.end,
                    dialogue=dlg, item=item, episode_duration=duration, cfg=cfg,
                    device=task_device,
                ))
            # Release the full wav tensor now that dialogue slices are copied
            del wav_tensor
            LOGGER.info(
                "PROFILE %s dur=%.1fs transcode=%.2fs diarize=%.2fs load_wav=%.2fs "
                "dialogues=%d (queued for separation)",
                key, duration, t_transcode, t_diarize, t_load_wav, len(valid_dialogues),
            )
        else:
            # No separation: extract audio segments directly
            for idx, dlg in enumerate(valid_dialogues):
                dlg_key = f"{key}_{idx:04d}"
                dlg_mp3 = tmp_dir / f"{dlg_key}.mp3"
                try:
                    audio.extract_audio_segment(
                        wav_path, dlg_mp3,
                        start=dlg.start, end=dlg.end,
                        sample_rate=cfg.audio_sample_rate,
                        channels=cfg.audio_channels,
                        bitrate_kbps=cfg.mp3_bitrate_kbps,
                    )
                    dlg_audio_bytes = dlg_mp3.read_bytes() if dlg_mp3.exists() else None
                    meta = _build_dialogue_meta(cfg, item, duration, idx, dlg)
                    diarization_data = {
                        "segments": dlg.segments,
                        "model": cfg.diarization_model,
                        "duration_sec": dlg.duration,
                    }
                    result.results.append(ProcessedResult(
                        key=dlg_key, status="ok", meta=meta,
                        audio_bytes=dlg_audio_bytes,
                        diarization=diarization_data, diarization_error=None,
                    ))
                except Exception as exc:  # noqa: BLE001
                    result.results.append(_make_error_result(dlg_key, str(exc), None))
                finally:
                    dlg_mp3.unlink(missing_ok=True)

            LOGGER.info(
                "PROFILE %s dur=%.1fs transcode=%.2fs diarize=%.2fs dialogues=%d",
                key, duration, t_transcode, t_diarize, len(valid_dialogues),
            )

        return result

    except Exception as exc:  # noqa: BLE001
        return DiarizeResult(
            episode_key=key, item=item, duration=duration,
            raw_path=downloaded.raw_path, status="error", error=str(exc),
        )
    finally:
        wav_path.unlink(missing_ok=True)


def _build_dialogue_meta(cfg: Config, item: AudioItem, duration: float, idx: int, dlg) -> dict:
    return {
        "rss_url": item.rss_url,
        "audio_url": item.audio_url,
        "source_type": item.source_type,
        "language": _metadata_language(cfg, item.language),
        "episode_duration_sec": duration,
        "dialogue_idx": idx,
        "dialogue_start": dlg.start,
        "dialogue_end": dlg.end,
        "dialogue_duration_sec": dlg.duration,
        "dialogue_speakers": dlg.speakers,
        "feed": item.feed_meta,
        "entry": item.entry_meta,
        "processed_at": dt.datetime.utcnow().isoformat(),
    }


# ── Stage 2: Separate (runs independently from diarize) ─────────────────────

def _separate_dialogue(
    task: SeparationTask,
    separation_models: dict,
    separation_lock: threading.Lock | None,
) -> ProcessedResult:
    """Run speaker separation on a single dialogue and return a ProcessedResult."""
    cfg = task.cfg
    dlg = task.dialogue
    try:
        models_for_task = separation_models
        if str(separation_models.get("device")) != task.device:
            models_for_task = separate_mod.load_separation_models(
                task.device, cfg.separation_backend, cfg.separation_model
            )
        t0 = time.monotonic()
        if separation_lock is None:
            spk0, spk1, sep_sr = separate_mod.run_separation(
                task.dlg_wav, 16_000, cfg.separation_num_steps, models_for_task
            )
        else:
            with separation_lock:
                spk0, spk1, sep_sr = separate_mod.run_separation(
                    task.dlg_wav, 16_000, cfg.separation_num_steps, models_for_task
                )
        t_sep = time.monotonic() - t0

        t0 = time.monotonic()
        dlg_audio_bytes = audio.tensors_to_stereo_mp3_bytes(
            spk0, spk1, sep_sr, cfg.mp3_bitrate_kbps
        )
        t_enc = time.monotonic() - t0

        LOGGER.info(
            "PROFILE separation %s dur=%.1fs sep=%.2fs encode=%.2fs device=%s",
            task.dlg_key, dlg.duration, t_sep, t_enc, task.device,
        )

        meta = _build_dialogue_meta(cfg, task.item, task.episode_duration, 0, dlg)
        meta["dialogue_idx"] = int(task.dlg_key.rsplit("_", 1)[-1])
        meta["separated"] = True
        meta["separation_backend"] = cfg.separation_backend
        meta["separation_model"] = models_for_task.get("model_id") or cfg.separation_model or REPO_ID_SIDON
        meta["separation_sample_rate"] = sep_sr
        meta["device"] = task.device
        meta["gpu_id"] = int(task.device.split(":", 1)[1]) if task.device.startswith("cuda:") else None
        meta["channels"] = 2
        _write_debug_separation_outputs(task, spk0, spk1, sep_sr, meta["separation_model"])

        diarization_data = {
            "segments": dlg.segments,
            "model": cfg.diarization_model,
            "duration_sec": dlg.duration,
        }

        return ProcessedResult(
            key=task.dlg_key, status="ok", meta=meta,
            audio_bytes=dlg_audio_bytes,
            diarization=diarization_data, diarization_error=None,
        )
    except Exception as exc:  # noqa: BLE001
        return _make_error_result(task.dlg_key, str(exc), None)


# ── Non-diarization path ────────────────────────────────────────────────────

def _process_no_diarization(
    downloaded: DownloadedItem,
    cfg: Config,
) -> ProcessedResult:
    scratch = _scratch_dir(cfg)
    tmp_dir = scratch / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mp3_path = tmp_dir / f"{downloaded.key}.mp3"
    try:
        audio.transcode_to_mp3_cbr(
            downloaded.raw_path, mp3_path,
            sample_rate=cfg.audio_sample_rate,
            channels=cfg.audio_channels,
            bitrate_kbps=cfg.mp3_bitrate_kbps,
        )
        meta = {
            "rss_url": downloaded.item.rss_url,
            "audio_url": downloaded.item.audio_url,
            "source_type": downloaded.item.source_type,
            "language": _metadata_language(cfg, downloaded.item.language),
            "duration_sec": downloaded.duration,
            "feed": downloaded.item.feed_meta,
            "entry": downloaded.item.entry_meta,
            "processed_at": dt.datetime.utcnow().isoformat(),
        }
        return ProcessedResult(
            key=downloaded.key, status="ok", meta=meta,
            audio_bytes=mp3_path.read_bytes(),
            diarization=None, diarization_error=None,
            raw_path=downloaded.raw_path,
        )
    except Exception as exc:  # noqa: BLE001
        return _make_error_result(downloaded.key, str(exc), downloaded.raw_path)
    finally:
        mp3_path.unlink(missing_ok=True)


# ── Feed iteration ───────────────────────────────────────────────────────────

def _rss_source(source_id: str, url: str, language: str, priority: int = 0) -> sources.SourceRecord:
    return sources.SourceRecord(
        source_id=source_id,
        source_type="rss",
        url=url,
        language=language,
        priority=priority,
    )


def _iter_sources(cfg: Config) -> list[sources.SourceRecord]:
    source_records: list[sources.SourceRecord] = []
    seen: set[str] = set()

    for record in sorted(
        sources.load_allowlist(cfg.youtube_allowlist),
        key=lambda item: item.priority,
        reverse=True,
    ):
        if record.source_type != "youtube" or record.url in seen:
            continue
        seen.add(record.url)
        source_records.append(record)

    if cfg.youtube_only:
        return source_records

    tgz_path = db.download_feeds_db(cfg.feeds_db_url, cfg.cache_dir)
    db_path = db.extract_sqlite_db(tgz_path, cfg.cache_dir / "db")

    for rss_url, lang in db.iter_feed_urls(db_path, cfg.languages):
        if rss_url in seen:
            continue
        seen.add(rss_url)
        source_records.append(_rss_source(_hash_url(rss_url), rss_url, lang))

    for record in sorted(
        sources.load_allowlist(cfg.feed_allowlist),
        key=lambda item: item.priority,
        reverse=True,
    ):
        if record.source_type != "rss" or record.url in seen:
            continue
        seen.add(record.url)
        source_records.append(record)

    return source_records


def _iter_feed_urls(cfg: Config) -> list[tuple[str, str]]:
    return [
        (record.url, record.language)
        for record in _iter_sources(cfg)
        if record.source_type == "rss"
    ]


def _collect_source_items(record: sources.SourceRecord, cfg: Config) -> list[AudioItem]:
    if record.source_type == "rss":
        return _collect_feed_items(record.url, record.language, cfg)
    if record.source_type == "youtube":
        target_duration_sec = cfg.target_hours * 3600 if cfg.youtube_only and cfg.target_hours is not None else None
        feed_meta = {
            "source_id": record.source_id,
            "source_type": "youtube",
            "url": record.url,
            "title": record.show_name,
            "license_notes": record.license_notes,
            "priority": record.priority,
        }
        return [
            AudioItem(
                audio_url=entry["url"],
                rss_url=record.url,
                language=record.language,
                feed_meta=feed_meta,
                entry_meta=entry,
                source_type="youtube",
            )
            for entry in youtube.iter_entries(record.url, cfg.episode_limit_per_feed, target_duration_sec)
        ]
    raise ValueError(f"Unsupported source_type {record.source_type!r}")


def _submit_feed_futures(
    feed_iter: Iterator[sources.SourceRecord],
    feed_futures: dict,
    rss_pool: ThreadPoolExecutor,
    cfg: Config,
    max_pending_feeds: int,
    stats: Stats | None = None,
) -> None:
    while len(feed_futures) < max_pending_feeds:
        if stats is not None and _target_hours_reached(cfg, stats):
            break
        try:
            record = next(feed_iter)
        except StopIteration:
            break
        future = rss_pool.submit(_collect_source_items, record, cfg)
        feed_futures[future] = record


# ── Main orchestration ───────────────────────────────────────────────────────

def crawl_and_build_dataset(cfg: Config) -> None:
    cfg.runtime_device = resolve_device(cfg.runtime_device, cfg.allow_cpu_fallback)
    if cfg.diarization_device in {"auto", "gpu"}:
        cfg.diarization_device = cfg.runtime_device
    gpu_ids = validate_multi_gpu(cfg.multi_gpu_enabled, cfg.multi_gpu_device_ids)
    run_dir = setup_run_logging(cfg.log_root, cfg.run_id)
    cfg.run_id = run_dir.name
    cfg.run_dir = run_dir
    write_resolved_config(run_dir, cfg)
    append_phase_tree(
        run_dir,
        "end2end",
        [
            ("languages", ",".join(cfg.languages)),
            ("diarization", f"{cfg.enable_diarization} {cfg.diarization_model}"),
            ("separation", f"{cfg.enable_separation} {cfg.separation_backend}"),
            ("device", cfg.diarization_device),
            ("multi_gpu", ",".join(map(str, gpu_ids)) if gpu_ids else "disabled"),
            ("debug_outputs", str(Path(cfg.debug_outputs_dir)) if cfg.debug_outputs_enabled else "disabled"),
            ("target_hours", str(cfg.target_hours or "")),
        ],
    )
    audio.ensure_ffmpeg()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    started_at = dt.datetime.utcnow()
    if cfg.num_nodes > 1:
        cfg.cache_dir = cfg.cache_dir / str(cfg.node_index)
        cfg.output_dir = cfg.output_dir / str(cfg.node_index)
        cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
    feed_urls = _iter_sources(cfg)
    if cfg.num_nodes > 1:
        feed_urls = feed_urls[cfg.node_index::cfg.num_nodes]
    LOGGER.info(
        "Node %d/%d: processing %d sources (scratch=%s)",
        cfg.node_index, cfg.num_nodes, len(feed_urls),
        cfg.scratch_dir or "cache_dir",
    )

    writer = wds.open_shard_writer(cfg.output_dir, cfg.shard_size_gb)
    processed_db = ProcessedDB(cfg.cache_dir / "processed.sqlite")

    diarization_pipeline = None
    separation_models = None
    diarize_lock = None
    separation_lock = None
    if cfg.enable_diarization:
        diarization_pipeline = diarize.load_diarization_pipeline(
            cfg.diarization_model, cfg.diarization_device, cfg.diarization_backend
        )
        diarize_lock = threading.Lock()
    if cfg.enable_separation and cfg.enable_diarization:
        separation_models = separate_mod.load_separation_models(
            cfg.diarization_device, cfg.separation_backend, cfg.separation_model
        )
        separation_lock = threading.Lock()
    elif cfg.enable_separation:
        LOGGER.warning("enable_separation requires enable_diarization; separation disabled.")

    seen_hashes: set[str] = set()
    pending_downloads: set = set()
    pending_diarize: set = set()
    pending_separate: set = set()

    rss_pool = ThreadPoolExecutor(max_workers=cfg.rss_workers)
    download_pool = ThreadPoolExecutor(max_workers=max(1, cfg.download_workers))
    diarize_pool = ThreadPoolExecutor(max_workers=max(1, cfg.process_workers))
    separate_pool = ThreadPoolExecutor(max_workers=max(1, cfg.separation_workers))
    feed_pbar = None
    audio_pbar = None

    interrupted = False
    stats = Stats()

    def _drain_separations() -> None:
        """Drain completed separation futures."""
        if not pending_separate:
            return
        done, remaining = wait(pending_separate, return_when=FIRST_COMPLETED)
        pending_separate.clear()
        pending_separate.update(remaining)
        for future in done:
            result = future.result()
            _handle_processed_result(result, cfg, writer, processed_db, audio_pbar, stats)
            if _target_hours_reached(cfg, stats):
                break

    def _drain_diarizations() -> None:
        """Drain completed diarize futures → feed separation pool."""
        if not pending_diarize:
            return
        done, remaining = wait(pending_diarize, return_when=FIRST_COMPLETED)
        pending_diarize.clear()
        pending_diarize.update(remaining)
        for future in done:
            dr: DiarizeResult = future.result()
            if dr.status != "ok":
                # Skip or error — mark in DB and clean up
                _handle_processed_result(
                    ProcessedResult(
                        key=dr.episode_key, status=dr.status, meta=None,
                        audio_bytes=None, diarization=None, diarization_error=None,
                        error=dr.error, raw_path=dr.raw_path,
                    ),
                    cfg, writer, processed_db, audio_pbar, stats,
                )
                continue

            # Enqueue separation tasks, draining the separation pool when
            # the in-flight queue grows too large so dlg_wav tensors don't
            # accumulate in memory.
            sep_buffer_limit = max(1, cfg.separation_workers) * TASK_BUFFER_MULTIPLIER
            for task in dr.separation_tasks:
                if _target_hours_reached(cfg, stats):
                    break
                while len(pending_separate) >= sep_buffer_limit:
                    _drain_separations()
                    if _target_hours_reached(cfg, stats):
                        break
                if _target_hours_reached(cfg, stats):
                    break
                sep_future = separate_pool.submit(
                    _separate_dialogue, task, separation_models, separation_lock,
                )
                pending_separate.add(sep_future)

            # Handle direct results (non-separation path)
            for result in dr.results:
                _handle_processed_result(result, cfg, writer, processed_db, audio_pbar, stats)
                if _target_hours_reached(cfg, stats):
                    break

            # Episode marker
            processed_db.mark(dr.episode_key, "ok")
            if cfg.cleanup_audio_cache and dr.raw_path is not None:
                dr.raw_path.unlink(missing_ok=True)

    def _drain_downloads() -> None:
        """Drain completed download futures → feed diarize or process pool."""
        if not pending_downloads:
            return
        done, remaining = wait(pending_downloads, return_when=FIRST_COMPLETED)
        pending_downloads.clear()
        pending_downloads.update(remaining)
        for future in done:
            if _target_hours_reached(cfg, stats):
                break
            result = future.result()
            if isinstance(result, ProcessedResult):
                _handle_processed_result(result, cfg, writer, processed_db, audio_pbar, stats)
                continue

            if cfg.enable_diarization and diarization_pipeline is not None:
                diarize_future = diarize_pool.submit(
                    _diarize_episode, result, cfg, diarization_pipeline, diarize_lock, gpu_ids,
                )
                pending_diarize.add(diarize_future)
            else:
                # Non-diarization: process directly
                proc_result = _process_no_diarization(result, cfg)
                _handle_processed_result(proc_result, cfg, writer, processed_db, audio_pbar, stats)

    try:
        max_pending_feeds = max(1, cfg.rss_workers) * FEED_FUTURE_BUFFER_MULTIPLIER
        feed_iter = iter(feed_urls)
        feed_futures: dict = {}
        _submit_feed_futures(feed_iter, feed_futures, rss_pool, cfg, max_pending_feeds, stats)

        feed_pbar = tqdm(total=len(feed_urls), desc="sources", unit="source", leave=False)
        audio_pbar = tqdm(desc="audio", unit="item", leave=False)

        while feed_futures:
            if _target_hours_reached(cfg, stats):
                LOGGER.info("target_hours %.2f reached; stopping feed submission", cfg.target_hours)
                break
            done_feeds, _ = wait(set(feed_futures), return_when=FIRST_COMPLETED)
            for future in done_feeds:
                feed_pbar.update(1)
                record = feed_futures.pop(future)
                try:
                    items = future.result()
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Failed source fetch: %s (%s)", record.url, exc)
                    continue

                for item in items:
                    if _target_hours_reached(cfg, stats):
                        break
                    norm_url = rss.normalize_url(item.audio_url)
                    key = _hash_url(norm_url)
                    if key in seen_hashes:
                        continue
                    seen_hashes.add(key)
                    if processed_db.is_done(key):
                        continue

                    download_future = download_pool.submit(
                        _download_audio_item, item, cfg, key,
                    )
                    pending_downloads.add(download_future)

                    # Drain pools when buffers are full
                    if len(pending_downloads) >= max(1, cfg.download_workers) * TASK_BUFFER_MULTIPLIER:
                        _drain_downloads()
                    if len(pending_diarize) >= max(1, cfg.process_workers) * TASK_BUFFER_MULTIPLIER:
                        _drain_diarizations()
                    if len(pending_separate) >= max(1, cfg.separation_workers) * TASK_BUFFER_MULTIPLIER:
                        _drain_separations()

            _submit_feed_futures(feed_iter, feed_futures, rss_pool, cfg, max_pending_feeds, stats)

        # Drain remaining work
        while pending_downloads or pending_diarize or pending_separate:
            if pending_downloads:
                _drain_downloads()
            if pending_diarize:
                _drain_diarizations()
            if pending_separate:
                _drain_separations()
            if _target_hours_reached(cfg, stats):
                LOGGER.info("target_hours %.2f reached; cancelling remaining work", cfg.target_hours)
                break
    except KeyboardInterrupt:
        interrupted = True
        LOGGER.warning("Keyboard interrupt received; shutting down.")
    finally:
        rss_pool.shutdown(wait=False, cancel_futures=True)
        download_pool.shutdown(wait=False, cancel_futures=True)
        diarize_pool.shutdown(wait=False, cancel_futures=True)
        separate_pool.shutdown(wait=False, cancel_futures=True)
        if feed_pbar is not None:
            feed_pbar.close()
        if audio_pbar is not None:
            audio_pbar.close()
        writer.close()
        processed_db.close()
        if interrupted:
            _cleanup_partial_downloads(_scratch_dir(cfg) / "audio")
        finished_at = dt.datetime.utcnow()
        if stats.total_duration_sec > 0:
            hours = stats.total_duration_sec / 3600
            LOGGER.info(
                "Total duration written: %.2f seconds (%.2f hours)",
                stats.total_duration_sec, hours,
            )

        summary = {
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "interrupted": interrupted,
            "total_duration_sec": stats.total_duration_sec,
            "total_duration_hours": stats.total_duration_sec / 3600,
            "written": stats.written,
            "skipped": stats.skipped,
            "errors": stats.errors,
            "output_dir": str(cfg.output_dir),
        }
        summary_path = cfg.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        append_stats_table(run_dir, "end2end", summary)
        write_artifacts(run_dir, {"summary": str(summary_path), "output_dir": str(cfg.output_dir)})


def _handle_processed_result(
    result: ProcessedResult,
    cfg: Config,
    writer,
    processed_db: ProcessedDB,
    audio_pbar,
    stats: Stats,
) -> None:
    # Episode markers only update the DB and optionally clean up raw audio.
    if result.is_marker:
        processed_db.mark(result.key, result.status, result.error)
        if cfg.cleanup_audio_cache and result.raw_path is not None:
            result.raw_path.unlink(missing_ok=True)
        return

    audio_pbar.update(1)

    if result.status == "ok":
        if _target_hours_reached(cfg, stats):
            if cfg.cleanup_audio_cache and result.raw_path is not None:
                result.raw_path.unlink(missing_ok=True)
            stats.skipped += 1
            audio_pbar.set_postfix_str(
                f"hours={stats.total_duration_sec / 3600:.2f} "
                f"ok={stats.written} skip={stats.skipped} err={stats.errors}",
                refresh=True,
            )
            return
        wds.write_sample(
            writer, result.key, result.audio_bytes,
            result.meta or {}, result.diarization, result.diarization_error,
        )
        processed_db.mark(result.key, "ok")
        if cfg.cleanup_audio_cache and result.raw_path is not None:
            result.raw_path.unlink(missing_ok=True)
        if result.meta is not None:
            duration_key = "dialogue_duration_sec" if "dialogue_idx" in result.meta else "duration_sec"
            stats.total_duration_sec += float(result.meta.get(duration_key, 0.0))
        stats.written += 1
    elif result.status in {"skipped_duration", "skipped_no_dialogues", "skipped_low_quality"}:
        processed_db.mark(result.key, result.status, result.error)
        stats.skipped += 1
    else:
        LOGGER.warning("Error processing %s: %s", result.key, result.error)
        if cfg.run_dir is not None:
            append_error(
                cfg.run_dir,
                {"key": result.key, "status": result.status, "error": result.error},
            )
        processed_db.mark(result.key, "error", result.error)
        stats.errors += 1

    audio_pbar.set_postfix_str(
        f"hours={stats.total_duration_sec / 3600:.2f} "
        f"ok={stats.written} skip={stats.skipped} err={stats.errors}",
        refresh=True,
    )


def _cleanup_partial_downloads(audio_dir: Path) -> None:
    if not audio_dir.exists():
        return
    for part_path in audio_dir.rglob("*.part"):
        part_path.unlink(missing_ok=True)
