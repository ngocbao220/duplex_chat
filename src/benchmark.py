from __future__ import annotations

import gzip
import argparse
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

try:
    from duplexchat_pipe.config import Config
    from duplexchat_pipe.logging_utils import append_stats_table, setup_run_logging, write_artifacts
except ModuleNotFoundError:
    from config import Config
    from logging_utils import append_stats_table, setup_run_logging, write_artifacts


LOGGER = logging.getLogger(__name__)
DNSMOS_METRICS = ("dnsmos",)
OBJECTIVE_METRICS = ("sq_stoi", "sq_pesq", "sq_si_sdr")
SUBJECTIVE_METRICS = ("squim_mos",)
EMBEDDING_METRICS = ("itc", "itd")
METRIC_NAMES = (*DNSMOS_METRICS, *SUBJECTIVE_METRICS, *OBJECTIVE_METRICS, *EMBEDDING_METRICS)
DEFAULT_METRIC_NAMES = (*SUBJECTIVE_METRICS, *OBJECTIVE_METRICS, *EMBEDDING_METRICS)
QUALITY_TABLE_COLUMNS = [
    "key",
    "duration_sec",
    "metric_status",
    "dnsmos_mean",
    "sq_stoi_mean",
    "sq_pesq_mean",
    "sq_si_sdr_mean",
    "squim_mos_mean",
    "itc",
    "itd",
]
TURN_TAKING_TABLE_COLUMNS = [
    "key",
    "turn_exchanges_per_min",
    "mean_turn_duration_sec",
    "backchannels_per_min",
    "simultaneous_speech_pct",
    "overlapping_transitions_pct",
]


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


def load_speaker_pair(speaker_a: Path, speaker_b: Path) -> tuple[torch.Tensor, int]:
    wav_a, sr_a = torchaudio.load(str(speaker_a))
    wav_b, sr_b = torchaudio.load(str(speaker_b))
    wav_a = wav_a.mean(dim=0, keepdim=True).float()
    wav_b = wav_b.mean(dim=0, keepdim=True).float()
    if sr_b != sr_a:
        wav_b = F_audio.resample(wav_b, sr_b, sr_a)
    length = min(wav_a.shape[1], wav_b.shape[1])
    if length <= 0:
        raise ValueError("speaker audio is empty")
    return torch.cat([wav_a[:, :length], wav_b[:, :length]], dim=0), int(sr_a)


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


def _format_table_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown_table(path: Path, rows: list[dict[str, Any]], columns: list[str], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_format_table_value(row.get(column)) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metrics_table_path(output: Path) -> Path:
    return output.with_suffix(".metrics.md")


def _turn_taking_table_path(output: Path) -> Path:
    return output.with_suffix(".turn_taking.md")


def _segments_intersection_duration(a: list[dict[str, float]], b: list[dict[str, float]]) -> float:
    total = 0.0
    i = 0
    j = 0
    while i < len(a) and j < len(b):
        start = max(a[i]["start"], b[j]["start"])
        end = min(a[i]["end"], b[j]["end"])
        if end > start:
            total += end - start
        if a[i]["end"] <= b[j]["end"]:
            i += 1
        else:
            j += 1
    return total


def _activity_segments(track: torch.Tensor, sample_rate: int) -> list[dict[str, float]]:
    frame = max(1, int(sample_rate * 0.02))
    hop = frame
    if track.numel() < frame:
        return []
    frames = track[: (track.numel() // hop) * hop].reshape(-1, hop)
    rms = torch.sqrt(torch.mean(frames.float() ** 2, dim=1) + 1e-12)
    if torch.max(rms).item() <= 1e-6:
        return []
    threshold = max(1e-4, torch.quantile(rms, 0.75).item() * 0.5, torch.max(rms).item() * 0.05)
    active = rms > threshold
    raw_segments = []
    start_idx = None
    for idx, is_active in enumerate(active.tolist()):
        if is_active and start_idx is None:
            start_idx = idx
        elif not is_active and start_idx is not None:
            raw_segments.append({"start": start_idx * hop / sample_rate, "end": idx * hop / sample_rate})
            start_idx = None
    if start_idx is not None:
        raw_segments.append({"start": start_idx * hop / sample_rate, "end": len(active) * hop / sample_rate})

    merged: list[dict[str, float]] = []
    for segment in raw_segments:
        if segment["end"] - segment["start"] < 0.12:
            continue
        if merged and segment["start"] - merged[-1]["end"] <= 0.2:
            merged[-1]["end"] = segment["end"]
        else:
            merged.append(segment)
    return merged


def compute_turn_taking_stats(waveform: torch.Tensor, sample_rate: int) -> dict[str, float | None]:
    waveform = _ensure_stereo(waveform)
    duration_sec = waveform.shape[-1] / sample_rate if sample_rate > 0 else 0.0
    duration_min = duration_sec / 60.0 if duration_sec > 0 else 0.0
    tracks = [
        [{**segment, "speaker": speaker} for segment in _activity_segments(waveform[idx], sample_rate)]
        for idx, speaker in enumerate(["A", "B"])
    ]
    turns = sorted([segment for track in tracks for segment in track], key=lambda item: (item["start"], item["end"]))
    if not turns or duration_min <= 0:
        return {
            "turn_exchanges_per_min": None,
            "mean_turn_duration_sec": None,
            "backchannels_per_min": None,
            "simultaneous_speech_pct": None,
            "overlapping_transitions_pct": None,
        }

    exchanges = 0
    overlapping_exchanges = 0
    previous = turns[0]
    for turn in turns[1:]:
        if turn["speaker"] == previous["speaker"]:
            if turn["end"] > previous["end"]:
                previous = turn
            continue
        exchanges += 1
        if turn["start"] < previous["end"]:
            overlapping_exchanges += 1
        previous = turn

    backchannels = 0
    for idx, track in enumerate(tracks):
        other = tracks[1 - idx]
        for turn in track:
            turn_duration = turn["end"] - turn["start"]
            if 0.2 <= turn_duration <= 1.0 and _segments_intersection_duration([turn], other) > 0:
                backchannels += 1

    simultaneous = _segments_intersection_duration(tracks[0], tracks[1])
    turn_durations = [turn["end"] - turn["start"] for turn in turns]
    return {
        "turn_exchanges_per_min": exchanges / duration_min,
        "mean_turn_duration_sec": _mean(turn_durations),
        "backchannels_per_min": backchannels / duration_min,
        "simultaneous_speech_pct": (simultaneous / duration_sec) * 100.0 if duration_sec else None,
        "overlapping_transitions_pct": (overlapping_exchanges / exchanges) * 100.0 if exchanges else 0.0,
    }


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


class DNSMOSScorer:
    def __init__(self, cfg: Config):
        self.unavailable_reason: str | None = None
        self.sample_rate = 16000
        self.window_samples = self.sample_rate * 9
        if cfg.benchmark_dnsmos_model_path is None:
            self.session = None
            self.input_name = None
            self.unavailable_reason = "DNSMOS requires benchmark.dnsmos_model_path"
            return
        try:
            import onnxruntime as ort

            self.session = ort.InferenceSession(str(cfg.benchmark_dnsmos_model_path))
            self.input_name = self.session.get_inputs()[0].name
        except Exception as exc:  # noqa: BLE001
            self.session = None
            self.input_name = None
            self.unavailable_reason = f"DNSMOS unavailable: {exc}"

    def score(self, waveform: torch.Tensor, sample_rate: int) -> dict[str, list[float | None]]:
        if self.session is None or self.input_name is None:
            return {}
        prepared = _resample(waveform, sample_rate, self.sample_rate).detach().cpu()
        return {
            "dnsmos": [
                self._score_track(prepared[idx])
                for idx in range(prepared.shape[0])
            ]
        }

    def _score_track(self, track: torch.Tensor) -> float | None:
        import numpy as np

        audio = track.float().numpy()
        if audio.size == 0:
            return None
        if audio.size < self.window_samples:
            audio = np.pad(audio, (0, self.window_samples - audio.size))
        values = []
        for start in range(0, audio.size, self.window_samples):
            chunk = audio[start : start + self.window_samples]
            if chunk.size < self.sample_rate:
                continue
            if chunk.size < self.window_samples:
                chunk = np.pad(chunk, (0, self.window_samples - chunk.size))
            inputs = {self.input_name: chunk[np.newaxis, :].astype(np.float32)}
            outputs = self.session.run(None, inputs)
            if not outputs:
                continue
            ovr_score = outputs[-1]
            values.append(_float_or_none(np.asarray(ovr_score).mean()))
        return _mean(values)


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
            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except ModuleNotFoundError:
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

    def _track_consistency(self, wav: torch.Tensor, window_seconds: float = 2.0) -> float | None:
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
    metric_rows: list[dict[str, Any]] = []
    turn_taking_rows: list[dict[str, Any]] = []

    dnsmos_scorer = DNSMOSScorer(cfg) if set(selected_metrics) & set(DNSMOS_METRICS) else None
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
                dnsmos_scorer,
                objective_scorer,
                subjective_scorer,
                embedding_scorer,
                selected_metrics,
            )
            status_counts[row["metric_status"]] = status_counts.get(row["metric_status"], 0) + 1
            metric_rows.append(row)
            turn_taking_rows.append(row)
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
    metrics_table_path = output_dir / "metrics_table.md"
    turn_taking_table_path = output_dir / "turn_taking_table.md"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown_table(
        metrics_table_path,
        metric_rows,
        QUALITY_TABLE_COLUMNS,
        "DuplexChat Benchmark Metrics",
    )
    write_markdown_table(
        turn_taking_table_path,
        turn_taking_rows,
        TURN_TAKING_TABLE_COLUMNS,
        "Turn-Taking Statistics of DuplexChat",
    )
    append_stats_table(run_dir, "benchmark", summary)
    write_artifacts(
        run_dir,
        {
            "summary": str(summary_path),
            "metrics": str(metrics_path),
            "metrics_table": str(metrics_table_path),
            "turn_taking_table": str(turn_taking_table_path),
        },
    )
    LOGGER.info("benchmark report written to %s", summary_path)


def score_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    cfg: Config,
    key: str = "single",
    duration_sec: float | None = None,
    separation_backend: str = "single",
) -> dict[str, Any]:
    selected_metrics = tuple(name for name in METRIC_NAMES if name in set(cfg.benchmark_metrics))
    dnsmos_scorer = DNSMOSScorer(cfg) if set(selected_metrics) & set(DNSMOS_METRICS) else None
    objective_scorer = SquimObjectiveScorer(cfg) if set(selected_metrics) & set(OBJECTIVE_METRICS) else None
    subjective_scorer = SquimSubjectiveScorer(cfg) if set(selected_metrics) & set(SUBJECTIVE_METRICS) else None
    embedding_scorer = EmbeddingScorer(cfg) if set(selected_metrics) & set(EMBEDDING_METRICS) else None
    sample = BenchmarkSample(
        key=key,
        meta={},
        audio_bytes=b"decoded",
        diarization=None,
    )
    return _score_decoded_waveform(
        sample,
        _ensure_stereo(waveform),
        sample_rate,
        duration_sec if duration_sec is not None else waveform.shape[-1] / sample_rate,
        separation_backend,
        dnsmos_scorer,
        objective_scorer,
        subjective_scorer,
        embedding_scorer,
        selected_metrics,
    )


def _score_sample(
    sample: BenchmarkSample,
    duration: float,
    backend: str,
    dnsmos_scorer: DNSMOSScorer | None,
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
        if metric in DNSMOS_METRICS or metric in OBJECTIVE_METRICS or metric in SUBJECTIVE_METRICS:
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

    return _score_decoded_waveform(
        sample,
        waveform,
        sample_rate,
        duration,
        backend,
        dnsmos_scorer,
        objective_scorer,
        subjective_scorer,
        embedding_scorer,
        selected_metrics,
        errors,
    )


def _score_decoded_waveform(
    sample: BenchmarkSample,
    waveform: torch.Tensor,
    sample_rate: int,
    duration: float,
    backend: str,
    dnsmos_scorer: DNSMOSScorer | None,
    objective_scorer: SquimObjectiveScorer | None,
    subjective_scorer: SquimSubjectiveScorer | None,
    embedding_scorer: EmbeddingScorer | None,
    selected_metrics: tuple[str, ...],
    errors: dict[str, str] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "key": sample.key,
        "duration_sec": duration,
        "separation_backend": backend,
    }
    for metric in selected_metrics:
        if metric in DNSMOS_METRICS or metric in OBJECTIVE_METRICS or metric in SUBJECTIVE_METRICS:
            _add_channel_metric(row, metric, [None, None])
        else:
            row[metric] = None

    errors = errors or {}

    if dnsmos_scorer is not None:
        unavailable_reason = getattr(dnsmos_scorer, "unavailable_reason", None)
        if unavailable_reason:
            errors["dnsmos"] = unavailable_reason
        else:
            for name, values in dnsmos_scorer.score(waveform, sample_rate).items():
                if name in selected_metrics:
                    _add_channel_metric(row, name, values)

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

    row.update(compute_turn_taking_stats(waveform, sample_rate))

    present = any(
        row.get(name) is not None
        for name in [
            *(f"{name}_mean" for name in (*DNSMOS_METRICS, *SUBJECTIVE_METRICS, *OBJECTIVE_METRICS) if name in selected_metrics),
            *(name for name in EMBEDDING_METRICS if name in selected_metrics),
        ]
    )
    row["metric_status"] = "ok" if present and not errors else "partial" if present else "unavailable"
    if errors:
        row["metric_errors"] = errors
    return row


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark DuplexChat outputs.")
    parser.add_argument("--single", action="store_true", help="Benchmark two separated speaker audio files.")
    parser.add_argument("--speakerA", type=Path, help="Path to speaker A WAV/audio file.")
    parser.add_argument("--speakerB", type=Path, help="Path to speaker B WAV/audio file.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path. Defaults to speakerA parent / benchmark.json.")
    parser.add_argument("--device", default="auto", help="Benchmark device: auto, cpu, cuda, cuda:N.")
    parser.add_argument("--reference", type=Path, default=None, help="Non-matching clean reference WAV for SQUIM subjective MOS.")
    parser.add_argument("--dnsmos-model", type=Path, default=None, help="Path to Microsoft DNSMOS sig_bak_ovr.onnx.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRIC_NAMES),
        help="Metrics to compute. DNSMOS is supported but disabled by default.",
    )
    return parser


def _run_single(args: argparse.Namespace) -> None:
    if args.speakerA is None or args.speakerB is None:
        raise SystemExit("--single requires --speakerA and --speakerB")
    waveform, sample_rate = load_speaker_pair(args.speakerA, args.speakerB)
    cfg = Config(
        benchmark_device=args.device,
        benchmark_metrics=args.metrics,
        benchmark_squim_subjective_reference_path=args.reference,
        benchmark_dnsmos_model_path=args.dnsmos_model,
    )
    row = score_waveform(
        waveform,
        sample_rate,
        cfg,
        key=args.speakerA.stem,
        separation_backend="single",
    )
    output = args.output or (args.speakerA.parent / "benchmark.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics_table_path = _metrics_table_path(output)
    turn_taking_table_path = _turn_taking_table_path(output)
    output.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown_table(
        metrics_table_path,
        [row],
        QUALITY_TABLE_COLUMNS,
        "DuplexChat Benchmark Metrics",
    )
    write_markdown_table(
        turn_taking_table_path,
        [row],
        TURN_TAKING_TABLE_COLUMNS,
        "Turn-Taking Statistics of DuplexChat",
    )
    print(json.dumps(row, ensure_ascii=False, indent=2))
    print(f"Saved benchmark: {output}")
    print(f"Saved metrics table: {metrics_table_path}")
    print(f"Saved turn-taking table: {turn_taking_table_path}")


def main() -> None:
    args = _build_parser().parse_args()
    if args.single:
        _run_single(args)
        return
    raise SystemExit("Only --single is supported by this script. Use duplexchat-pipe run --phase benchmark for WebDataset.")


if __name__ == "__main__":
    main()
