from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from duplexchat_pipe import benchmark
from duplexchat_pipe.cholimex import run_cholimex_file
from duplexchat_pipe.config import load_config
from duplexchat_pipe.outputs import save_wav
from duplexchat_pipe.runtime_warnings import suppress_pyannote_tf32_warning
from duplexchat_pipe.runner import run_phase


def _phase(title: str) -> None:
    print(f"=====================Phase {title}", flush=True)


def _run_otospeech(cfg, args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    pred_root = Path(args.pred_root) if args.pred_root is not None else output_root / "otospeech"
    mixture_root = Path(args.mixture_root) if args.mixture_root is not None else output_root / "otospeech_mixtures"
    report = Path(args.benchmark_output) if args.benchmark_output is not None else cfg.benchmark_output_dir / "summary.json"

    _phase("1: Downloading OtoSpeech")
    local_dir = benchmark.download_otospeech_dataset(
        repo_id=args.otospeech_repo,
        local_dir=args.otospeech_local_dir,
        max_download_gb=args.size_gb,
    )
    samples = benchmark.discover_otospeech_samples(local_dir)
    print(f"OtoSpeech local_dir={local_dir}", flush=True)
    print(f"OtoSpeech samples={len(samples)} size_gb<={args.size_gb}", flush=True)

    _phase("2: Mixing speaker streams")
    prepared = []
    for idx, sample in enumerate(samples, 1):
        key = sample["key"]
        mixture, _, sample_rate = benchmark.mix_ground_truth_pair(
            Path(sample["gt_speaker_1"]),
            Path(sample["gt_speaker_2"]),
            args.sample_rate,
        )
        mixture_path = mixture_root / key / "mixture.wav"
        save_wav(mixture_path, mixture, sample_rate)
        for field in ("gt_speaker_1", "gt_speaker_2"):
            source = Path(sample[field])
            destination = mixture_path.parent / f"{field}.wav"
            if not destination.exists() or not source.samefile(destination):
                shutil.copy2(source, destination)
        prepared.append((sample, mixture_path))
        print(f"[{idx}/{len(samples)}] mixed {key} -> {mixture_path} (originals: gt_speaker_1.wav, gt_speaker_2.wav)", flush=True)

    _phase("3: Running pipeline")
    for idx, (sample, mixture_path) in enumerate(prepared, 1):
        key = sample["key"]
        out_dir = pred_root / key
        if (
            not args.force
            and (
                ((out_dir / "speaker_0.wav").is_file() and (out_dir / "speaker_1.wav").is_file())
                or ((out_dir / "speaker_A.wav").is_file() and (out_dir / "speaker_B.wav").is_file())
            )
        ):
            print(f"[{idx}/{len(prepared)}] skip existing {key}", flush=True)
            continue
        print(f"[{idx}/{len(prepared)}] predict {key}", flush=True)
        run_cholimex_file(mixture_path, out_dir, cfg)

    _phase("4: Running benchmark")
    benchmark.run_reference_benchmark(
        samples=samples,
        pred_root=pred_root,
        output=report,
        target_sample_rate=args.sample_rate,
        vad_threshold_db=args.vad_threshold_db,
        crosstalk_threshold_db=args.crosstalk_threshold_db,
    )
    print(f"Saved benchmark: {report}", flush=True)
    print(f"Saved sample metrics: {report.parent / 'sample_metrics.jsonl'}", flush=True)
    print(f"Saved summary table: {report.parent / 'summary.md'}", flush=True)


def main() -> None:
    suppress_pyannote_tf32_warning()
    parser = argparse.ArgumentParser(description="Run DuplexChat end-to-end pipeline.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.json"))
    parser.add_argument("--target_hours", type=float, default=None)
    parser.add_argument("--youtube-only", action="store_true")
    parser.add_argument("--data", choices=["crawl", "oto-speech"], default="crawl")
    parser.add_argument("--size_gb", type=float, default=10.0)
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

    cfg = load_config(args.config)
    if args.target_hours is not None:
        cfg.target_hours = args.target_hours
    if args.youtube_only:
        cfg.youtube_only = True
    if args.data == "oto-speech":
        _run_otospeech(cfg, args)
        return
    run_phase(cfg, "end2end")


if __name__ == "__main__":
    main()
