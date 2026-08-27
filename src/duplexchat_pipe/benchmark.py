from __future__ import annotations

import gzip
import json
import logging
import tarfile
from pathlib import Path
from typing import Iterator

from duplexchat_pipe.logging_utils import append_stats_table, setup_run_logging, write_artifacts


LOGGER = logging.getLogger(__name__)
METRIC_NAMES = ("dnsmos", "sq_stoi", "sq_pesq", "itc", "itd")


def _iter_samples(input_dir: Path) -> Iterator[tuple[str, dict]]:
    for shard_path in sorted(input_dir.rglob("*.tar.gz")):
        with gzip.open(shard_path, "rb") as gz:
            with tarfile.open(fileobj=gz, mode="r|") as tf:
                current_key = ""
                current: dict[str, bytes] = {}
                for member in tf:
                    if not member.isfile():
                        continue
                    dot = member.name.find(".")
                    if dot < 0:
                        continue
                    key = member.name[:dot]
                    ext = member.name[dot + 1:]
                    if current_key and key != current_key:
                        meta_raw = current.get("meta.json")
                        if meta_raw:
                            yield current_key, json.loads(meta_raw.decode("utf-8"))
                        current = {}
                    current_key = key
                    f = tf.extractfile(member)
                    if f is not None:
                        current[ext] = f.read()
                meta_raw = current.get("meta.json")
                if current_key and meta_raw:
                    yield current_key, json.loads(meta_raw.decode("utf-8"))


def run_benchmark(input_dir: Path, output_dir: Path, log_root: Path, run_id: str | None) -> None:
    """Write the benchmark report contract with explicit unavailable metrics."""
    run_dir = setup_run_logging(log_root, run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    review_path = output_dir / "review_manifest.jsonl"
    total = 0
    total_duration = 0.0
    backends: dict[str, int] = {}

    with metrics_path.open("w", encoding="utf-8") as metrics_fp, review_path.open(
        "w", encoding="utf-8"
    ) as review_fp:
        for key, meta in _iter_samples(input_dir):
            total += 1
            duration = float(meta.get("dialogue_duration_sec") or meta.get("duration_sec") or 0.0)
            total_duration += duration
            backend = str(meta.get("separation_backend") or "none")
            backends[backend] = backends.get(backend, 0) + 1
            row = {
                "key": key,
                "duration_sec": duration,
                "separation_backend": backend,
                **{name: None for name in METRIC_NAMES},
                "metric_status": "unavailable",
                "metric_error": "Install/enable DNSMOS, SQUIM, and speaker embedding evaluators.",
            }
            metrics_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            if total <= 100:
                review_fp.write(
                    json.dumps({"key": key, "meta": meta, "metrics": row}, ensure_ascii=False)
                    + "\n"
                )

    summary = {
        "samples": total,
        "hours": total_duration / 3600,
        "metric_status": "unavailable",
        "backends": backends,
        "metrics_jsonl": str(metrics_path),
        "review_manifest_jsonl": str(review_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    append_stats_table(run_dir, "benchmark", summary)
    write_artifacts(run_dir, {"summary": str(summary_path), "metrics": str(metrics_path)})
    LOGGER.info("benchmark report written to %s", summary_path)
