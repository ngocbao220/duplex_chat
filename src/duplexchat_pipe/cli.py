from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from duplexchat_pipe.config import Config
from duplexchat_pipe.pipeline import crawl_and_build_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="duplexchat-pipe")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run crawl and build WebDataset")
    run_parser.add_argument("--output", type=Path, default=Path("./data/wds"))
    run_parser.add_argument("--cache-dir", type=Path, default=None, help="Cache directory for feeds DB and processed.sqlite (default: ./data/cache).")
    run_parser.add_argument("--languages", nargs="+", default=None)
    run_parser.add_argument("--episode-limit", type=int, default=0)
    run_parser.add_argument("--max-duration-min", type=int, default=180)
    run_parser.add_argument("--shard-size-gb", type=float, default=3.0, help="Max shard size in GB (default: 3.0)")
    run_parser.add_argument("--rss-workers", type=int, default=16)
    run_parser.add_argument("--download-workers", type=int, default=4)
    run_parser.add_argument("--process-workers", type=int, default=4, help="Number of concurrent diarization threads (default: 4).")
    run_parser.add_argument("--separation-workers", type=int, default=2, help="Number of concurrent separation threads (default: 2).")
    run_parser.add_argument("--scratch-dir", type=Path, default=None, help="Local NVMe scratch dir for temp files (e.g. $LOCALDIR on Miyabi).")
    run_parser.add_argument("--mp3-bitrate-kbps", type=int, default=128)
    run_parser.add_argument("--refresh-db", action="store_true")
    run_parser.add_argument(
        "--enable-diarization",
        action="store_true",
        help="Enable speaker diarization (requires Hugging Face auth).",
    )
    run_parser.add_argument(
        "--diarization-device",
        default="cuda",
        help="Device for diarization model (default: cuda, falls back to cpu if unavailable).",
    )
    run_parser.add_argument(
        "--enable-separation",
        action="store_true",
        help="Run DialogueSidon speaker separation on each dialogue (requires --enable-diarization).",
    )
    run_parser.add_argument(
        "--separation-num-steps",
        type=int,
        default=30,
        help="Number of diffusion steps for separation (default: 30, more = better quality).",
    )
    run_parser.add_argument(
        "--dialogue-gap-seconds",
        type=float,
        default=5.0,
        help="Minimum silence gap (seconds) between turns to split a new dialogue (default: 5.0).",
    )
    run_parser.add_argument(
        "--dialogue-max-single-speaker-ratio",
        type=float,
        default=0.8,
        help="Discard dialogues where one speaker exceeds this fraction of total turn time (default: 0.8).",
    )
    run_parser.add_argument(
        "--node-index",
        type=int,
        default=0,
        help="Index of this node (0-based) when distributing feeds across multiple nodes.",
    )
    run_parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="Total number of nodes sharing the feed list.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        cfg = Config()
        cfg.output_dir = args.output
        if args.cache_dir:
            cfg.cache_dir = args.cache_dir
        cfg.max_duration_minutes = args.max_duration_min
        cfg.shard_size_gb = args.shard_size_gb
        cfg.rss_workers = args.rss_workers
        cfg.download_workers = args.download_workers
        cfg.process_workers = args.process_workers
        cfg.separation_workers = args.separation_workers
        cfg.scratch_dir = args.scratch_dir
        cfg.mp3_bitrate_kbps = args.mp3_bitrate_kbps
        cfg.enable_diarization = bool(args.enable_diarization)
        cfg.diarization_device = args.diarization_device
        cfg.enable_separation = bool(args.enable_separation)
        cfg.separation_num_steps = args.separation_num_steps
        cfg.dialogue_gap_seconds = args.dialogue_gap_seconds
        cfg.dialogue_max_single_speaker_ratio = args.dialogue_max_single_speaker_ratio
        cfg.node_index = args.node_index
        cfg.num_nodes = args.num_nodes

        if args.languages:
            cfg.languages = args.languages
        cfg.episode_limit_per_feed = None if args.episode_limit == 0 else args.episode_limit

        if args.refresh_db:
            tgz_path = cfg.cache_dir / "podcastindex_feeds.db.tgz"
            if tgz_path.exists():
                tgz_path.unlink()
            db_dir = cfg.cache_dir / "db"
            if db_dir.exists():
                shutil.rmtree(db_dir)

        crawl_and_build_dataset(cfg)


if __name__ == "__main__":
    main()
