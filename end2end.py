from __future__ import annotations

import argparse
from pathlib import Path

from duplexchat_pipe.config import load_config
from duplexchat_pipe.runner import run_phase


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DuplexChat end-to-end pipeline.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.json"))
    parser.add_argument("--target_hours", type=float, default=10.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.target_hours = args.target_hours
    run_phase(cfg, "end2end")


if __name__ == "__main__":
    main()
