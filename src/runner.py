from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from duplexchat_pipe.benchmark import run_benchmark
from duplexchat_pipe.config import Config
from duplexchat_pipe.logging_utils import append_stats_table, setup_run_logging, write_artifacts, write_resolved_config
from duplexchat_pipe.pipeline import _iter_feed_urls, crawl_and_build_dataset


def _benchmark_output_dir(cfg: Config) -> Path:
    output_dir = cfg.benchmark_output_dir
    if output_dir == Path("./reports") or output_dir == Path("reports"):
        return output_dir / (cfg.run_id or "vi_poc")
    return output_dir


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


def _run_progress_steps(description: str, steps: list[tuple[str, Callable[[], None]]]) -> None:
    with tqdm(total=len(steps), desc=description, unit="phase") as pbar:
        for label, callback in steps:
            pbar.set_postfix_str(label, refresh=True)
            callback()
            pbar.update(1)


def run_phase(cfg: Config, phase: str) -> None:
    if phase in {"end2end", "run", "separate"}:
        steps = [("crawl/download/diarize/separate", lambda: crawl_and_build_dataset(cfg))]
        if phase in {"end2end", "run"} and cfg.benchmark_enabled:
            steps.append(("benchmark", lambda: run_benchmark(cfg.output_dir, _benchmark_output_dir(cfg), cfg)))
        _run_progress_steps(phase, steps)
        return
    if phase == "download_clean":
        def run_download_clean() -> None:
            cfg.enable_diarization = False
            cfg.enable_separation = False
            crawl_and_build_dataset(cfg)

        _run_progress_steps(phase, [("download/clean", run_download_clean)])
        return
    if phase == "diarize_segment":
        def run_diarize_segment() -> None:
            cfg.enable_diarization = True
            cfg.enable_separation = False
            crawl_and_build_dataset(cfg)

        _run_progress_steps(phase, [("download/diarize/segment", run_diarize_segment)])
        return
    if phase == "collect_sources":
        _run_progress_steps(phase, [("collect sources", lambda: collect_sources(cfg))])
        return
    if phase == "benchmark":
        _run_progress_steps(phase, [("benchmark", lambda: run_benchmark(cfg.output_dir, _benchmark_output_dir(cfg), cfg))])
        return
    raise ValueError(f"Unsupported phase '{phase}'")
