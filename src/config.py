from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from duplexchat_pipe.model_options import DIARIZATION_MODELS, SEPARATION_MODELS
except ModuleNotFoundError:
    from model_options import DIARIZATION_MODELS, SEPARATION_MODELS


@dataclass
class Config:
    feeds_db_url: str = "https://public.podcastindex.org/podcastindex_feeds.db.tgz"
    languages: list[str] = field(default_factory=lambda: ["en", "ja"])
    feed_allowlist: Path | None = None
    youtube_allowlist: Path | None = None
    target_hours: float | None = None
    output_dir: Path = Path("./data/wds")
    cache_dir: Path = Path("./data/cache")
    log_root: Path = Path("./logs")
    run_id: str | None = None
    run_dir: Path | None = None
    shard_size_gb: float = 3.0
    rss_workers: int = 64
    download_workers: int = 64
    process_workers: int = 4
    max_duration_minutes: int = 180
    episode_limit_per_feed: int | None = None
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    mp3_bitrate_kbps: int = 128
    min_original_sample_rate: int = 16000
    min_original_bitrate: int = 32_000
    diarization_backend: str = "auto"
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    runtime_device: str = "auto"
    allow_cpu_fallback: bool = True
    multi_gpu_enabled: bool = False
    multi_gpu_device_ids: list[int] = field(default_factory=list)
    multi_gpu_strategy: str = "round_robin"
    diarization_device: str = "auto"
    enable_diarization: bool = False
    dialogue_gap_seconds: float = 5.0
    dialogue_min_duration_seconds: float = 10.0
    dialogue_max_duration_seconds: float = 600.0
    dialogue_max_single_speaker_ratio: float = 0.8
    enable_separation: bool = False
    separation_backend: str = "dialoguesidon"
    separation_model: str | None = None
    separation_num_steps: int = 30
    debug_outputs_enabled: bool = True
    debug_outputs_dir: Path = Path("./outputs")
    cleanup_audio_cache: bool = True
    timeout_seconds: int = 30
    node_index: int = 0
    num_nodes: int = 1
    scratch_dir: Path | None = None  # local NVMe for temp/audio files; falls back to cache_dir
    separation_workers: int = 2
    benchmark_enabled: bool = False
    benchmark_metrics: list[str] = field(default_factory=lambda: [
        "dnsmos",
        "squim_mos",
        "sq_stoi",
        "sq_pesq",
        "sq_si_sdr",
        "itc",
        "itd",
    ])
    benchmark_output_dir: Path = Path("./reports")
    benchmark_device: str = "auto"
    benchmark_squim_objective_enabled: bool = True
    benchmark_squim_subjective_enabled: bool = True
    benchmark_squim_subjective_reference_path: Path | None = None
    benchmark_dnsmos_model_path: Path | None = None
    benchmark_speaker_embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb"


FIELD_ALIASES = {
    "source.feeds_db_url": "feeds_db_url",
    "source.languages": "languages",
    "source.feed_allowlist": "feed_allowlist",
    "source.youtube_allowlist": "youtube_allowlist",
    "source.target_hours": "target_hours",
    "source.episode_limit_per_feed": "episode_limit_per_feed",
    "audio.output_dir": "output_dir",
    "audio.cache_dir": "cache_dir",
    "audio.sample_rate": "audio_sample_rate",
    "audio.channels": "audio_channels",
    "audio.mp3_bitrate_kbps": "mp3_bitrate_kbps",
    "audio.min_original_sample_rate": "min_original_sample_rate",
    "audio.min_original_bitrate": "min_original_bitrate",
    "runtime.log_root": "log_root",
    "runtime.run_id": "run_id",
    "runtime.device": "runtime_device",
    "runtime.allow_cpu_fallback": "allow_cpu_fallback",
    "runtime.multi_gpu.enabled": "multi_gpu_enabled",
    "runtime.multi_gpu.device_ids": "multi_gpu_device_ids",
    "runtime.multi_gpu.strategy": "multi_gpu_strategy",
    "runtime.rss_workers": "rss_workers",
    "runtime.download_workers": "download_workers",
    "runtime.process_workers": "process_workers",
    "runtime.separation_workers": "separation_workers",
    "runtime.scratch_dir": "scratch_dir",
    "runtime.debug_outputs.enabled": "debug_outputs_enabled",
    "runtime.debug_outputs.dir": "debug_outputs_dir",
    "runtime.timeout_seconds": "timeout_seconds",
    "runtime.node_index": "node_index",
    "runtime.num_nodes": "num_nodes",
    "diarization.enabled": "enable_diarization",
    "diarization.backend": "diarization_backend",
    "diarization.model": "diarization_model",
    "diarization.device": "diarization_device",
    "diarization.dialogue_gap_seconds": "dialogue_gap_seconds",
    "diarization.dialogue_min_duration_seconds": "dialogue_min_duration_seconds",
    "diarization.dialogue_max_duration_seconds": "dialogue_max_duration_seconds",
    "diarization.dialogue_max_single_speaker_ratio": "dialogue_max_single_speaker_ratio",
    "separation.enabled": "enable_separation",
    "separation.backend": "separation_backend",
    "separation.model": "separation_model",
    "separation.num_steps": "separation_num_steps",
    "benchmark.enabled": "benchmark_enabled",
    "benchmark.metrics": "benchmark_metrics",
    "benchmark.output_dir": "benchmark_output_dir",
    "benchmark.device": "benchmark_device",
    "benchmark.squim_objective.enabled": "benchmark_squim_objective_enabled",
    "benchmark.squim_subjective.enabled": "benchmark_squim_subjective_enabled",
    "benchmark.squim_subjective.reference_path": "benchmark_squim_subjective_reference_path",
    "benchmark.dnsmos_model_path": "benchmark_dnsmos_model_path",
    "benchmark.speaker_embedding_model": "benchmark_speaker_embedding_model",
}

PATH_FIELDS = {
    "feed_allowlist",
    "youtube_allowlist",
    "output_dir",
    "cache_dir",
    "log_root",
    "run_dir",
    "debug_outputs_dir",
    "scratch_dir",
    "benchmark_output_dir",
    "benchmark_squim_subjective_reference_path",
    "benchmark_dnsmos_model_path",
}

IGNORED_PREFIXES: tuple[str, ...] = ()


def _flatten_mapping(node: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in node.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_mapping(value, full_key))
        else:
            flat[full_key] = value
    return flat


def _coerce_value(field_name: str, value: Any) -> Any:
    if value == "":
        value = None
    if field_name in PATH_FIELDS:
        return Path(value) if value is not None else None
    return value


def apply_config_data(cfg: Config, data: dict[str, Any]) -> Config:
    if not isinstance(data, dict):
        raise TypeError("config.json must contain a JSON object")

    for config_key, value in _flatten_mapping(data).items():
        if config_key in FIELD_ALIASES:
            field_name = FIELD_ALIASES[config_key]
            setattr(cfg, field_name, _coerce_value(field_name, value))
            continue
        if config_key.startswith(IGNORED_PREFIXES):
            continue
        raise ValueError(f"unknown config key: {config_key}")
    return cfg


def _parse_override_value(raw_value: str) -> Any:
    if raw_value == "":
        return None
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def _apply_profile_shortcut(cfg: Config, group: str, value: str) -> bool:
    profile = value.strip().lower()
    if group == "diarization":
        aliases = {
            "pyannote_community": ("pyannote", "pyannote/speaker-diarization-community-1"),
            "pyannote-community": ("pyannote", "pyannote/speaker-diarization-community-1"),
            "pyannote_3_1": ("pyannote", DIARIZATION_MODELS["pyannote-3.1"]),
            "pyannote-3.1": ("pyannote", DIARIZATION_MODELS["pyannote-3.1"]),
            "sortformer": ("sortformer", DIARIZATION_MODELS["sortformer"]),
            "diarizen": ("diarizen", DIARIZATION_MODELS["diarizen"]),
        }
        if profile not in aliases:
            raise ValueError(f"unknown diarization profile: {value}")
        cfg.enable_diarization = True
        cfg.diarization_backend, cfg.diarization_model = aliases[profile]
        return True

    if group == "separation":
        if profile not in SEPARATION_MODELS:
            raise ValueError(f"unknown separation profile: {value}")
        cfg.enable_separation = True
        cfg.separation_backend = profile
        cfg.separation_model = SEPARATION_MODELS[profile]
        cfg.separation_num_steps = 30 if profile == "dialoguesidon" else 0
        return True

    return False


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    data: dict[str, Any] = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must use key=value syntax: {override}")
        key, raw_value = override.split("=", 1)
        if not key:
            raise ValueError(f"override key cannot be empty: {override}")
        if "." not in key and _apply_profile_shortcut(cfg, key, raw_value):
            continue
        data[key] = _parse_override_value(raw_value)
    return apply_config_data(cfg, data)


def load_config(path: Path = Path("config.json")) -> Config:
    with Path(path).open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return apply_config_data(Config(), data)
