from __future__ import annotations

import sys
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from duplexchat_pipe.benchmark import run_benchmark
from duplexchat_pipe.config import Config
from duplexchat_pipe.logging_utils import append_stats_table, setup_run_logging, write_artifacts, write_resolved_config
from duplexchat_pipe.pipeline import _iter_feed_urls, crawl_and_build_dataset


def _to_plain(value: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(value, resolve=True)
    except Exception:
        return value


def _flatten_cfg(node: dict, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in node.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_cfg(value, full_key))
        else:
            flat[full_key] = value
    return flat


FIELD_ALIASES = {
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
}


def config_from_hydra(node: Any) -> Config:
    data = _to_plain(node)
    if not isinstance(data, dict):
        raise TypeError("Hydra config must compose to a mapping")
    flat = _flatten_cfg(data)
    valid_fields = {f.name: f for f in fields(Config)}
    cfg = Config()
    for hydra_key, field_name in FIELD_ALIASES.items():
        if hydra_key not in flat or field_name not in valid_fields:
            continue
        value = flat[hydra_key]
        if value is None or value == "":
            value = None
        if field_name.endswith("_dir") or field_name.endswith("allowlist"):
            value = Path(value) if value is not None else None
        setattr(cfg, field_name, value)
    return cfg


def collect_sources(cfg: Config) -> None:
    run_dir = setup_run_logging(cfg.log_root, cfg.run_id)
    cfg.run_id = run_dir.name
    cfg.run_dir = run_dir
    write_resolved_config(run_dir, cfg)

    feeds = _iter_feed_urls(cfg)
    out_dir = Path("data/interim/sources") / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    feeds_path = out_dir / "feeds.jsonl"
    with feeds_path.open("w", encoding="utf-8") as fp:
        for rss_url, language in feeds:
            fp.write(json.dumps({"rss_url": rss_url, "language": language}, ensure_ascii=False) + "\n")

    by_language: dict[str, int] = {}
    for _, language in feeds:
        by_language[language] = by_language.get(language, 0) + 1
    stats = {"feeds": len(feeds), "languages": by_language, "feeds_jsonl": str(feeds_path)}
    stats_path = out_dir / "source_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    append_stats_table(run_dir, "collect_sources", {"feeds": len(feeds)})
    write_artifacts(run_dir, {"feeds": str(feeds_path), "source_stats": str(stats_path)})


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    phase = "end2end"
    overrides = []
    for arg in args:
        if arg.startswith("phase="):
            phase = arg.split("=", 1)[1]
        else:
            overrides.append(arg)

    try:
        from hydra import compose, initialize_config_dir
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Hydra runner requires `hydra-core`. Install with `uv sync`.") from exc

    conf_dir = Path(__file__).resolve().parents[2] / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        hydra_cfg = compose(config_name="config", overrides=overrides)

    cfg = config_from_hydra(hydra_cfg)
    if phase in {"end2end", "run", "separate"}:
        crawl_and_build_dataset(cfg)
        return
    if phase == "download_clean":
        cfg.enable_diarization = False
        cfg.enable_separation = False
        crawl_and_build_dataset(cfg)
        return
    if phase == "diarize_segment":
        cfg.enable_diarization = True
        cfg.enable_separation = False
        crawl_and_build_dataset(cfg)
        return
    if phase == "collect_sources":
        collect_sources(cfg)
        return
    if phase == "benchmark":
        output_dir = Path("reports") / (cfg.run_id or "vi_poc")
        run_benchmark(cfg.output_dir, output_dir, cfg.log_root, cfg.run_id)
        return
    raise ValueError(f"Unsupported phase '{phase}'")
