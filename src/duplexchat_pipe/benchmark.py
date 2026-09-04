from __future__ import annotations

import gzip
import json
import logging
import math
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as F_audio

from duplexchat_pipe.config import Config
from duplexchat_pipe.logging_utils import append_stats_table, setup_run_logging, write_artifacts


LOGGER = logging.getLogger(__name__)
OBJECTIVE_METRICS = ("sq_stoi", "sq_pesq", "sq_si_sdr")
SUBJECTIVE_METRICS = ("squim_mos",)
EMBEDDING_METRICS = ("itc", "itd")
METRIC_NAMES = (*SUBJECTIVE_METRICS, *OBJECTIVE_METRICS, *EMBEDDING_METRICS)


@dataclass
class BenchmarkSample:
    key: str
    meta: dict[str, Any]
    audio_bytes: bytes | None
    diarization: dict[str, Any] | None


def _iter_samples(input_dir: Path) -> Iterator[BenchmarkSample]:
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
                        sample = _sample_from_members(current_key, current)
                        if sample is not None:
                            yield sample
                        current = {}
                    current_key = key
                    f = tf.extractfile(member)
                    if f is not None:
                        current[ext] = f.read()
                sample = _sample_from_members(current_key, current)
                if sample is not None:
                    yield sample


def _sample_from_members(key: str, members: dict[str, bytes]) -> BenchmarkSample | None:
    meta_raw = members.get("meta.json")
    if not key or not meta_raw:
        return None
    diarization = None
    if members.get("diarization.json"):
        diarization = json.loads(members["diarization.json"].decode("utf-8"))
    return BenchmarkSample(
        key=key,
        meta=json.loads(meta_raw.decode("utf-8")),
        audio_bytes=members.get("audio.mp3"),
        diarization=diarization,
    )


def decode_audio_bytes(audio_bytes: bytes) -> tuple[torch.Tensor, int]:
    with tempfile.NamedTemporaryFile(suffix=".mp3") as fp:
        fp.write(audio_bytes)
        fp.flush()
        waveform, sample_rate = torchaudio.load(fp.name)
    return _ensure_stereo(waveform), int(sample_rate)


def _ensure_stereo(waveform: torch.Tensor) -> torch.Tensor:
    waveform = waveform.detach().cpu().float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    if waveform.shape[0] > 2:
        waveform = waveform[:2]
    return waveform


def _mean(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return sum(finite) / len(finite)


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "gpu":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _resample(waveform: torch.Tensor, sample_rate: int, target_sample_rate: int) -> torch.Tensor:
    if sample_rate == target_sample_rate:
        return waveform
    return F_audio.resample(waveform, sample_rate, target_sample_rate)


def _add_channel_metric(row: dict[str, Any], name: str, values: list[float | None]) -> None:
    left = values[0] if values else None
    right = values[1] if len(values) > 1 else None
    row[f"{name}_A"] = left
    row[f"{name}_B"] = right
    row[f"{name}_mean"] = _mean([left, right])


class SquimObjectiveScorer:
    def __init__(self, cfg: Config):
        self.unavailable_reason: str | None = None
        self.device = torch.device(_resolve_device(cfg.benchmark_device))
        if not cfg.benchmark_squim_objective_enabled:
            self.unavailable_reason = "SQUIM objective disabled"
            self.model = None
            self.sample_rate = 16000
            return
        try:
            from torchaudio.pipelines import SQUIM_OBJECTIVE

            self.sample_rate = int(SQUIM_OBJECTIVE.sample_rate)
            self.model = SQUIM_OBJECTIVE.get_model().to(self.device).eval()
        except Exception as exc:  # noqa: BLE001
            self.model = None
            self.sample_rate = 16000
            self.unavailable_reason = f"SQUIM objective unavailable: {exc}"

    def score(self, waveform: torch.Tensor, sample_rate: int) -> dict[str, list[float | None]]:
        if self.model is None:
            return {}
        prepared = _resample(waveform, sample_rate, self.sample_rate).to(self.device)
        with torch.inference_mode():
            stoi, pesq, si_sdr = self.model(prepared)
        return {
            "sq_stoi": [_float_or_none(value) for value in stoi.detach().cpu().tolist()],
            "sq_pesq": [_float_or_none(value) for value in pesq.detach().cpu().tolist()],
            "sq_si_sdr": [_float_or_none(value) for value in si_sdr.detach().cpu().tolist()],
        }


class SquimSubjectiveScorer:
    def __init__(self, cfg: Config):
        self.unavailable_reason: str | None = None
        self.device = torch.device(_resolve_device(cfg.benchmark_device))
        if not cfg.benchmark_squim_subjective_enabled:
            self.unavailable_reason = "SQUIM subjective disabled"
            self.model = None
            self.reference = None
            self.sample_rate = 16000
            return
        if cfg.benchmark_squim_subjective_reference_path is None:
            self.unavailable_reason = "SQUIM subjective requires benchmark.squim_subjective.reference_path"
            self.model = None
            self.reference = None
            self.sample_rate = 16000
            return
        try:
            from torchaudio.pipelines import SQUIM_SUBJECTIVE

            self.sample_rate = int(SQUIM_SUBJECTIVE.sample_rate)
            self.model = SQUIM_SUBJECTIVE.get_model().to(self.device).eval()
            reference, reference_sr = torchaudio.load(str(cfg.benchmark_squim_subjective_reference_path))
            reference = reference.mean(dim=0, keepdim=True).float()
            self.reference = _resample(reference, int(reference_sr), self.sample_rate).to(self.device)
        except Exception as exc:  # noqa: BLE001
            self.model = None
            self.reference = None
            self.sample_rate = 16000
            self.unavailable_reason = f"SQUIM subjective unavailable: {exc}"

    def score(self, waveform: torch.Tensor, sample_rate: int) -> dict[str, list[float | None]]:
        if self.model is None or self.reference is None:
            return {}
        prepared = _resample(waveform, sample_rate, self.sample_rate).to(self.device)
        reference = self.reference.repeat(prepared.shape[0], 1)
        with torch.inference_mode():
            mos = self.model(prepared, reference)
        return {"squim_mos": [_float_or_none(value) for value in mos.detach().cpu().tolist()]}


class EmbeddingScorer:
    def __init__(self, cfg: Config):
        self.unavailable_reason: str | None = None
        self.device = _resolve_device(cfg.benchmark_device)
        self.sample_rate = 16000
        try:
            from speechbrain.inference.classifiers import EncoderClassifier

            self.model = EncoderClassifier.from_hparams(
                source=cfg.benchmark_speaker_embedding_model,
                run_opts={"device": self.device},
            )
        except Exception as exc:  # noqa: BLE001
            self.model = None
            self.unavailable_reason = f"speaker embedding unavailable: {exc}"

    def score(self, waveform: torch.Tensor, sample_rate: int) -> dict[str, float | None]:
        if self.model is None:
            return {}
        prepared = _resample(waveform, sample_rate, self.sample_rate)
        left = prepared[0]
        right = prepared[1] if prepared.shape[0] > 1 else prepared[0]
        return {
            "itc": _mean([self._track_consistency(left), self._track_consistency(right)]),
            "itd": self._track_distinctiveness(left, right),
        }

    def _embedding(self, wav: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            emb = self.model.encode_batch(wav.unsqueeze(0).to(self.device))
        return emb.detach().cpu().reshape(-1).float()

    def _track_consistency(self, wav: torch.Tensor, window_seconds: float = 3.0) -> float | None:
        window = int(self.sample_rate * window_seconds)
        min_window = self.sample_rate
        chunks = [
            wav[start : start + window]
            for start in range(0, max(0, wav.numel() - min_window + 1), window)
            if wav[start : start + window].numel() >= min_window
        ]
        if len(chunks) < 2:
            return 1.0 if wav.numel() >= min_window else None
        embeddings = [self._embedding(chunk) for chunk in chunks]
        scores = []
        for idx, emb_a in enumerate(embeddings):
            for emb_b in embeddings[idx + 1 :]:
                scores.append(_float_or_none(F.cosine_similarity(emb_a, emb_b, dim=0)))
        return _mean(scores)

    def _track_distinctiveness(self, left: torch.Tensor, right: torch.Tensor) -> float | None:
        if left.numel() < self.sample_rate or right.numel() < self.sample_rate:
            return None
        similarity = _float_or_none(F.cosine_similarity(self._embedding(left), self._embedding(right), dim=0))
        if similarity is None:
            return None
        return 1.0 - similarity


def run_benchmark(input_dir: Path, output_dir: Path, cfg: Config) -> None:
    """Compute inspectable benchmark metrics for WebDataset audio samples."""
    run_dir = setup_run_logging(cfg.log_root, cfg.run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    review_path = output_dir / "review_manifest.jsonl"
    total = 0
    total_duration = 0.0
    backends: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    selected_metrics = tuple(name for name in METRIC_NAMES if name in set(cfg.benchmark_metrics))

    objective_scorer = SquimObjectiveScorer(cfg) if set(selected_metrics) & set(OBJECTIVE_METRICS) else None
    subjective_scorer = SquimSubjectiveScorer(cfg) if set(selected_metrics) & set(SUBJECTIVE_METRICS) else None
    embedding_scorer = EmbeddingScorer(cfg) if set(selected_metrics) & set(EMBEDDING_METRICS) else None

    with metrics_path.open("w", encoding="utf-8") as metrics_fp, review_path.open(
        "w", encoding="utf-8"
    ) as review_fp:
        for sample in _iter_samples(input_dir):
            total += 1
            meta = sample.meta
            duration = float(meta.get("dialogue_duration_sec") or meta.get("duration_sec") or 0.0)
            total_duration += duration
            backend = str(meta.get("separation_backend") or "none")
            backends[backend] = backends.get(backend, 0) + 1
            row = _score_sample(
                sample,
                duration,
                backend,
                objective_scorer,
                subjective_scorer,
                embedding_scorer,
                selected_metrics,
            )
            status_counts[row["metric_status"]] = status_counts.get(row["metric_status"], 0) + 1
            metrics_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            if total <= 100:
                review_fp.write(
                    json.dumps({"key": sample.key, "meta": meta, "metrics": row}, ensure_ascii=False)
                    + "\n"
                )

    summary = {
        "samples": total,
        "hours": total_duration / 3600,
        "metric_status": "ok" if status_counts.get("ok") == total and total > 0 else "partial",
        "metric_status_counts": status_counts,
        "backends": backends,
        "metrics": list(selected_metrics),
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


def _score_sample(
    sample: BenchmarkSample,
    duration: float,
    backend: str,
    objective_scorer: SquimObjectiveScorer | None,
    subjective_scorer: SquimSubjectiveScorer | None,
    embedding_scorer: EmbeddingScorer | None,
    selected_metrics: tuple[str, ...],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "key": sample.key,
        "duration_sec": duration,
        "separation_backend": backend,
    }
    for metric in selected_metrics:
        if metric in OBJECTIVE_METRICS or metric in SUBJECTIVE_METRICS:
            _add_channel_metric(row, metric, [None, None])
        else:
            row[metric] = None

    errors: dict[str, str] = {}
    if not sample.audio_bytes:
        errors["audio"] = "audio.mp3 missing"
        row["metric_status"] = "unavailable"
        row["metric_errors"] = errors
        return row

    try:
        waveform, sample_rate = decode_audio_bytes(sample.audio_bytes)
    except Exception as exc:  # noqa: BLE001
        errors["audio"] = f"audio decode failed: {exc}"
        row["metric_status"] = "unavailable"
        row["metric_errors"] = errors
        return row

    if objective_scorer is not None:
        unavailable_reason = getattr(objective_scorer, "unavailable_reason", None)
        if unavailable_reason:
            errors["squim_objective"] = unavailable_reason
        else:
            for name, values in objective_scorer.score(waveform, sample_rate).items():
                if name in selected_metrics:
                    _add_channel_metric(row, name, values)

    if subjective_scorer is not None:
        unavailable_reason = getattr(subjective_scorer, "unavailable_reason", None)
        if unavailable_reason:
            errors["squim_subjective"] = unavailable_reason
        else:
            for name, values in subjective_scorer.score(waveform, sample_rate).items():
                if name in selected_metrics:
                    _add_channel_metric(row, name, values)

    if embedding_scorer is not None:
        unavailable_reason = getattr(embedding_scorer, "unavailable_reason", None)
        if unavailable_reason:
            errors["speaker_embedding"] = unavailable_reason
        else:
            row.update({
                name: value
                for name, value in embedding_scorer.score(waveform, sample_rate).items()
                if name in selected_metrics
            })

    present = any(
        row.get(name) is not None
        for name in [
            *(f"{name}_mean" for name in (*SUBJECTIVE_METRICS, *OBJECTIVE_METRICS) if name in selected_metrics),
            *(name for name in EMBEDDING_METRICS if name in selected_metrics),
        ]
    )
    row["metric_status"] = "ok" if present and not errors else "partial" if present else "unavailable"
    if errors:
        row["metric_errors"] = errors
    return row
