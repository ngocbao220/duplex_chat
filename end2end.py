from __future__ import annotations

import argparse
import math
from pathlib import Path

from duplexchat_pipe import benchmark
from duplexchat_pipe.config import load_config
from duplexchat_pipe.outputs import save_wav
from duplexchat_pipe.runtime_warnings import suppress_pyannote_tf32_warning
from duplexchat_pipe.runner import run_phase


def _phase(title: str) -> None:
    print(f"========= Phase {title} =========", flush=True)


def _run_otospeech(cfg, args: argparse.Namespace) -> int:
    from comparison.runner import run_otospeech
    return run_otospeech(cfg, args, benchmark, save_wav)


def main() -> None:
    suppress_pyannote_tf32_warning()
    parser = argparse.ArgumentParser(description="Run DuplexChat end-to-end pipeline.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.json"))
    parser.add_argument("--target_hours", type=float, default=None)
    parser.add_argument("--youtube-only", action="store_true")
    parser.add_argument("--otospeech-root", type=Path, default=None, help="Use a local dataset without downloading.")
    parser.add_argument("--max-samples", type=int, default=None, help="Bound the number of samples for a smoke run.")
    parser.add_argument("--max-seconds", type=float, default=None, help="Bound each mixture for a real smoke run; originals remain intact.")
    parser.add_argument("--data", choices=["crawl", "oto-speech"], default="crawl")
    parser.add_argument("--max_gb", type=float, default=10.0, help="Maximum OtoSpeech download size in GB.")
    parser.add_argument("--debug", action="store_true", help="Persist intermediate per-sample artifacts.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--pred-root", type=Path, default=None)
    parser.add_argument("--mixture-root", type=Path, default=None)
    parser.add_argument("--benchmark-output", type=Path, default=None)
    parser.add_argument("--otospeech-repo", default=benchmark.OTOSPEECH_REPO_ID)
    parser.add_argument("--otospeech-local-dir", type=Path, default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--vad-threshold-db", type=float, default=-40.0)
    parser.add_argument("--crosstalk-threshold-db", type=float, default=-20.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.pipeline = "duplexchat"
    for name in ("max_gb", "sample_rate", "max_samples", "max_seconds"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    cfg = load_config(args.config)
    if args.target_hours is not None:
        cfg.target_hours = args.target_hours
    if args.youtube_only:
        cfg.youtube_only = True
    if args.data == "oto-speech":
        raise SystemExit(_run_otospeech(cfg, args))
    run_phase(cfg, "end2end")


if __name__ == "__main__":
    main()
