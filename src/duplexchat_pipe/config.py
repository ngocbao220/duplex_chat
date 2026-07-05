from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    feeds_db_url: str = "https://public.podcastindex.org/podcastindex_feeds.db.tgz"
    languages: list[str] = field(default_factory=lambda: ["en", "ja"])
    output_dir: Path = Path("./data/wds")
    cache_dir: Path = Path("./data/cache")
    shard_size_gb: float = 3.0
    rss_workers: int = 64
    download_workers: int = 64
    process_workers: int = 4
    max_duration_minutes: int = 180
    episode_limit_per_feed: int | None = None
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    mp3_bitrate_kbps: int = 128
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    diarization_device: str = "cuda"
    enable_diarization: bool = False
    dialogue_gap_seconds: float = 5.0
    dialogue_min_duration_seconds: float = 10.0
    dialogue_max_duration_seconds: float = 600.0
    dialogue_max_single_speaker_ratio: float = 0.8
    enable_separation: bool = False
    separation_num_steps: int = 30
    cleanup_audio_cache: bool = True
    timeout_seconds: int = 30
    node_index: int = 0
    num_nodes: int = 1
    scratch_dir: Path | None = None  # local NVMe for temp/audio files; falls back to cache_dir
    separation_workers: int = 2
