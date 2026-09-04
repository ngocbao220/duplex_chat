from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from duplexchat_pipe.config import apply_overrides, load_config
from duplexchat_pipe.model_options import DIARIZATION_MODELS, SEPARATION_MODELS, resolve_model_alias
from duplexchat_pipe.runtime_warnings import suppress_pyannote_tf32_warning
from duplexchat_pipe.runner import run_phase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="duplexchat-pipe")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run crawl and build WebDataset")
    run_parser.add_argument("--config", type=Path, default=Path("configs/config.json"), help="JSON config path (default: configs/config.json).")
    run_parser.add_argument(
        "--phase",
        choices=["collect_sources", "download_clean", "diarize_segment", "separate", "benchmark", "end2end", "run"],
        default="end2end",
        help="Pipeline phase to run (default: end2end).",
    )
    run_parser.add_argument("--output", type=Path, default=argparse.SUPPRESS)
    run_parser.add_argument("--cache-dir", type=Path, default=argparse.SUPPRESS, help="Cache directory for feeds DB and processed.sqlite.")
    run_parser.add_argument("--log-root", type=Path, default=argparse.SUPPRESS)
    run_parser.add_argument("--run-id", default=argparse.SUPPRESS)
    run_parser.add_argument("--languages", nargs="+", default=argparse.SUPPRESS)
    run_parser.add_argument("--feed-allowlist", type=Path, default=argparse.SUPPRESS)
    run_parser.add_argument("--youtube-allowlist", type=Path, default=argparse.SUPPRESS)
    run_parser.add_argument("--target-hours", type=float, default=argparse.SUPPRESS)
    run_parser.add_argument("--episode-limit", type=int, default=argparse.SUPPRESS)
    run_parser.add_argument("--max-duration-min", type=int, default=argparse.SUPPRESS)
    run_parser.add_argument("--min-original-sample-rate", type=int, default=argparse.SUPPRESS)
    run_parser.add_argument("--min-original-bitrate", type=int, default=argparse.SUPPRESS)
    run_parser.add_argument("--shard-size-gb", type=float, default=argparse.SUPPRESS, help="Max shard size in GB.")
    run_parser.add_argument("--rss-workers", type=int, default=argparse.SUPPRESS)
    run_parser.add_argument("--download-workers", type=int, default=argparse.SUPPRESS)
    run_parser.add_argument("--process-workers", type=int, default=argparse.SUPPRESS, help="Number of concurrent diarization threads.")
    run_parser.add_argument("--separation-workers", type=int, default=argparse.SUPPRESS, help="Number of concurrent separation threads.")
    run_parser.add_argument("--scratch-dir", type=Path, default=argparse.SUPPRESS, help="Local NVMe scratch dir for temp files.")
    run_parser.add_argument("--debug-outputs-dir", type=Path, default=argparse.SUPPRESS, help="Directory for phase debug outputs.")
    run_parser.add_argument("--no-debug-outputs", action="store_true", default=argparse.SUPPRESS, help="Disable phase debug outputs.")
    run_parser.add_argument("--mp3-bitrate-kbps", type=int, default=argparse.SUPPRESS)
    run_parser.add_argument("--refresh-db", action="store_true", default=argparse.SUPPRESS)
    run_parser.add_argument(
        "--enable-diarization",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable speaker diarization (requires Hugging Face auth).",
    )
    run_parser.add_argument(
        "--diarization-device",
        default=argparse.SUPPRESS,
        help="Device for diarization model (default: auto = cuda then cpu).",
    )
    run_parser.add_argument(
        "--diarization-backend",
        choices=["auto", "pyannote", "sortformer", "diarizen"],
        default=argparse.SUPPRESS,
        help="Diarization backend (default: infer from model).",
    )
    run_parser.add_argument(
        "--diarization-model",
        default=argparse.SUPPRESS,
        help=(
            "Diarization model id or alias "
            f"({', '.join(sorted(DIARIZATION_MODELS))})."
        ),
    )
    run_parser.add_argument("--runtime-device", default=argparse.SUPPRESS, help="Global device policy (auto, cpu, cuda, cuda:N).")
    run_parser.add_argument("--no-cpu-fallback", action="store_true", default=argparse.SUPPRESS, help="Fail instead of falling back to CPU when CUDA is requested but unavailable.")
    run_parser.add_argument("--multi-gpu", action="store_true", default=argparse.SUPPRESS, help="Enable round-robin CUDA task assignment.")
    run_parser.add_argument("--multi-gpu-device-ids", nargs="*", type=int, default=argparse.SUPPRESS)
    run_parser.add_argument(
        "--enable-separation",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Run DialogueSidon speaker separation on each dialogue (requires --enable-diarization).",
    )
    run_parser.add_argument(
        "--separation-backend",
        choices=["dialoguesidon", "sepformer", "mossformer2"],
        default=argparse.SUPPRESS,
        help="Speech separation backend (default: dialoguesidon).",
    )
    run_parser.add_argument(
        "--separation-model",
        default=argparse.SUPPRESS,
        help="Override separation model id/checkpoint repository.",
    )
    run_parser.add_argument(
        "--separation-num-steps",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of diffusion steps for separation (default: 30, more = better quality).",
    )
    run_parser.add_argument(
        "--dialogue-gap-seconds",
        type=float,
        default=argparse.SUPPRESS,
        help="Minimum silence gap (seconds) between turns to split a new dialogue (default: 5.0).",
    )
    run_parser.add_argument(
        "--dialogue-max-single-speaker-ratio",
        type=float,
        default=argparse.SUPPRESS,
        help="Discard dialogues where one speaker exceeds this fraction of total turn time (default: 0.8).",
    )
    run_parser.add_argument(
        "--node-index",
        type=int,
        default=argparse.SUPPRESS,
        help="Index of this node (0-based) when distributing feeds across multiple nodes.",
    )
    run_parser.add_argument(
        "--num-nodes",
        type=int,
        default=argparse.SUPPRESS,
        help="Total number of nodes sharing the feed list.",
    )
    run_parser.add_argument("overrides", nargs="*", help="Config overrides such as source.target_hours=2.5.")

    return parser


def _apply_run_args(cfg, args: argparse.Namespace) -> None:
    if hasattr(args, "output"):
        cfg.output_dir = args.output
    if hasattr(args, "log_root"):
        cfg.log_root = args.log_root
    if hasattr(args, "run_id"):
        cfg.run_id = args.run_id
    if hasattr(args, "cache_dir"):
        cfg.cache_dir = args.cache_dir
    if hasattr(args, "feed_allowlist"):
        cfg.feed_allowlist = args.feed_allowlist
    if hasattr(args, "youtube_allowlist"):
        cfg.youtube_allowlist = args.youtube_allowlist
    if hasattr(args, "target_hours"):
        cfg.target_hours = args.target_hours
    if hasattr(args, "max_duration_min"):
        cfg.max_duration_minutes = args.max_duration_min
    if hasattr(args, "min_original_sample_rate"):
        cfg.min_original_sample_rate = args.min_original_sample_rate
    if hasattr(args, "min_original_bitrate"):
        cfg.min_original_bitrate = args.min_original_bitrate
    if hasattr(args, "shard_size_gb"):
        cfg.shard_size_gb = args.shard_size_gb
    if hasattr(args, "rss_workers"):
        cfg.rss_workers = args.rss_workers
    if hasattr(args, "download_workers"):
        cfg.download_workers = args.download_workers
    if hasattr(args, "process_workers"):
        cfg.process_workers = args.process_workers
    if hasattr(args, "separation_workers"):
        cfg.separation_workers = args.separation_workers
    if hasattr(args, "scratch_dir"):
        cfg.scratch_dir = args.scratch_dir
    if hasattr(args, "debug_outputs_dir"):
        cfg.debug_outputs_dir = args.debug_outputs_dir
    if hasattr(args, "no_debug_outputs"):
        cfg.debug_outputs_enabled = False
    if hasattr(args, "mp3_bitrate_kbps"):
        cfg.mp3_bitrate_kbps = args.mp3_bitrate_kbps
    if hasattr(args, "enable_diarization"):
        cfg.enable_diarization = bool(args.enable_diarization)
    if hasattr(args, "diarization_backend"):
        cfg.diarization_backend = args.diarization_backend
    if hasattr(args, "diarization_model"):
        cfg.diarization_model = resolve_model_alias(args.diarization_model, DIARIZATION_MODELS)
    if hasattr(args, "runtime_device"):
        cfg.runtime_device = args.runtime_device
    if hasattr(args, "no_cpu_fallback"):
        cfg.allow_cpu_fallback = False
    if hasattr(args, "multi_gpu"):
        cfg.multi_gpu_enabled = bool(args.multi_gpu)
    if hasattr(args, "multi_gpu_device_ids"):
        cfg.multi_gpu_device_ids = args.multi_gpu_device_ids
    if hasattr(args, "diarization_device"):
        cfg.diarization_device = args.diarization_device
    if hasattr(args, "enable_separation"):
        cfg.enable_separation = bool(args.enable_separation)
    if hasattr(args, "separation_backend"):
        cfg.separation_backend = args.separation_backend
    if hasattr(args, "separation_model"):
        cfg.separation_model = resolve_model_alias(args.separation_model, SEPARATION_MODELS)
    if hasattr(args, "separation_num_steps"):
        cfg.separation_num_steps = args.separation_num_steps
    if hasattr(args, "dialogue_gap_seconds"):
        cfg.dialogue_gap_seconds = args.dialogue_gap_seconds
    if hasattr(args, "dialogue_max_single_speaker_ratio"):
        cfg.dialogue_max_single_speaker_ratio = args.dialogue_max_single_speaker_ratio
    if hasattr(args, "node_index"):
        cfg.node_index = args.node_index
    if hasattr(args, "num_nodes"):
        cfg.num_nodes = args.num_nodes
    if hasattr(args, "languages"):
        cfg.languages = args.languages
    if hasattr(args, "episode_limit"):
        cfg.episode_limit_per_feed = None if args.episode_limit == 0 else args.episode_limit


def main() -> None:
    suppress_pyannote_tf32_warning()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        cfg = load_config(args.config)
        apply_overrides(cfg, args.overrides)
        _apply_run_args(cfg, args)

        if hasattr(args, "refresh_db"):
            tgz_path = cfg.cache_dir / "podcastindex_feeds.db.tgz"
            if tgz_path.exists():
                tgz_path.unlink()
            db_dir = cfg.cache_dir / "db"
            if db_dir.exists():
                shutil.rmtree(db_dir)

        run_phase(cfg, args.phase)


if __name__ == "__main__":
    main()
