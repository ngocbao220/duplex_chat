from __future__ import annotations

import argparse
from pathlib import Path

from duplexchat_pipe.config import load_config
from duplexchat_pipe.runtime_warnings import suppress_pyannote_tf32_warning
from duplexchat_pipe.runner import run_phase


def main() -> None:
    suppress_pyannote_tf32_warning()
    parser = argparse.ArgumentParser(description="Run DuplexChat end-to-end pipeline.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.json"))
    parser.add_argument("--target_hours", type=float, default=None)
    parser.add_argument("--youtube-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.target_hours is not None:
        cfg.target_hours = args.target_hours
    if args.youtube_only:
        cfg.youtube_only = True
    run_phase(cfg, "end2end")


if __name__ == "__main__":
    main()
