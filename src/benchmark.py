from __future__ import annotations

import gzip
import argparse
import json
import logging
import math
import statistics
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as F_audio
from huggingface_hub import HfApi, snapshot_download

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
OTOSPEECH_REPO_ID = "otoearth/otoSpeech-full-duplex-turn-104h"
OTOSPEECH_ALLOW_PATTERNS = [
    "*/metadata.json",
    "*/speaker_1_annotation_a.srt",
    "*/speaker_2_annotation_a.srt",
    "*/speaker_1_audio.wav",
    "*/speaker_2_audio.wav",
]
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
REFERENCE_CONDITIONS = ("all", "non_overlap", "overlap")
REFERENCE_METRIC_KEYS = (
    "pit_si_sdr",
    "delta_si_sdr",
    "sir",
    "sar",
    "stoi",
    "pesq",
    "crosstalk_rate",
    "leakage_p50_db",
    "leakage_p95_db",
    "vad_f1",
    "onset_mae",
    "offset_mae",
    "overlap_f1",
    "overlap_iou",
)
REFERENCE_SUMMARY_ROWS = [
    ("all", "Chất lượng audio", "PIT-SI-SDR", "Prediction giống ground truth đến mức nào", "pit_si_sdr"),
    ("all", "Artifact/rè/nhiễu", "SAR", "Mức méo và artifact do pipeline tạo ra", "sar"),
    ("overlap", "Separation trong overlap", "SIR", "Mức speaker còn lại bị lọt vào kênh", "sir"),
    ("all", "Độ rõ speech", "ESTOI/STOI", "Khả năng giữ lại nội dung lời nói", "stoi"),
    ("all", "Chất lượng nghe", "PESQ/POLQA", "Mức tương đồng về perceptual quality", "pesq"),
    ("overlap", "Crosstalk", "Crosstalk rate", "Tỷ lệ frame bị lẫn speaker thứ hai", "crosstalk_rate"),
    ("all", "Timing", "VAD F1, onset/offset error", "Prediction có giữ đúng thời điểm nói không", "vad_f1"),
    ("overlap", "Overlap timing", "Overlap F1/IoU", "Prediction có phát hiện đúng vùng overlap không", "overlap_f1"),
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


def load_mono_audio(path: Path, target_sample_rate: int | None = None) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(str(path))
    waveform = waveform.detach().cpu().float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.mean(dim=0, keepdim=True)
    if target_sample_rate is not None and sample_rate != target_sample_rate:
        waveform = F_audio.resample(waveform, int(sample_rate), int(target_sample_rate))
        sample_rate = int(target_sample_rate)
    return waveform, int(sample_rate)


def align_pair_lengths(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    length = min(a.shape[-1], b.shape[-1])
    if length <= 0:
        raise ValueError("audio pair is empty")
    return a[..., :length], b[..., :length]


def mix_ground_truth_pair(
    gt_agent: Path,
    gt_ctm: Path,
    target_sample_rate: int = 16000,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    agent, sample_rate = load_mono_audio(Path(gt_agent), target_sample_rate)
    ctm, _ = load_mono_audio(Path(gt_ctm), target_sample_rate)
    agent, ctm = align_pair_lengths(agent, ctm)
    gt = torch.cat([agent, ctm], dim=0)
    mixture = gt.sum(dim=0, keepdim=True)
    return mixture, gt, sample_rate


def discover_ippc_pairs(root: Path) -> list[dict[str, str]]:
    root = Path(root)
    pairs_dir = root / "pairs" if (root / "pairs").is_dir() else root
    rows: list[dict[str, str]] = []
    for pair_dir in sorted(pairs_dir.glob("pair_*")):
        if not pair_dir.is_dir():
            continue
        wavs = sorted(pair_dir.glob("*.wav"))
        agent = [path for path in wavs if "AGENT" in path.name.upper()]
        ctm = [path for path in wavs if "CTM" in path.name.upper()]
        if len(agent) != 1 or len(ctm) != 1:
            continue
        rows.append({"key": pair_dir.name, "gt_agent": str(agent[0]), "gt_ctm": str(ctm[0])})
    return rows


def discover_otospeech_samples(root: Path, sample_keys: set[str] | None = None) -> list[dict[str, str]]:
    root = Path(root)
    rows: list[dict[str, str]] = []
    for metadata in sorted(root.rglob("metadata.json")):
        sample_dir = metadata.parent
        key = sample_dir.relative_to(root).as_posix()
        if sample_keys is not None and key not in sample_keys:
            continue
        speaker_1 = sample_dir / "speaker_1_audio.wav"
        speaker_2 = sample_dir / "speaker_2_audio.wav"
        speaker_1_srt = sample_dir / "speaker_1_annotation_a.srt"
        speaker_2_srt = sample_dir / "speaker_2_annotation_a.srt"
        if not speaker_1.is_file() or not speaker_2.is_file():
            continue
        rows.append(
            {
                "key": key,
                "metadata": str(metadata),
                "gt_speaker_1": str(speaker_1),
                "gt_speaker_2": str(speaker_2),
                "speaker_1_srt": str(speaker_1_srt) if speaker_1_srt.is_file() else "",
                "speaker_2_srt": str(speaker_2_srt) if speaker_2_srt.is_file() else "",
            }
        )
    return rows


def otospeech_limited_allow_patterns(repo_id: str, max_download_gb: float) -> list[str]:
    budget_bytes = int(max_download_gb * 1024 ** 3)
    api = HfApi()
    files_by_sample: dict[str, dict[str, int]] = {}
    for item in api.list_repo_tree(repo_id=repo_id, repo_type="dataset", recursive=True):
        path = getattr(item, "path", "")
        size = getattr(item, "size", None)
        if not path or "/" not in path:
            continue
        filename = path.rsplit("/", 1)[-1]
        if filename not in {
            "metadata.json",
            "speaker_1_annotation_a.srt",
            "speaker_2_annotation_a.srt",
            "speaker_1_audio.wav",
            "speaker_2_audio.wav",
        }:
            continue
        if size is None:
            continue
        sample = path.rsplit("/", 1)[0]
        files_by_sample.setdefault(sample, {})[filename] = int(size)

    selected: list[str] = []
    used = 0
    for sample, files in sorted(files_by_sample.items()):
        if "speaker_1_audio.wav" not in files or "speaker_2_audio.wav" not in files:
            continue
        sample_paths = [
            f"{sample}/metadata.json",
            f"{sample}/speaker_1_annotation_a.srt",
            f"{sample}/speaker_2_annotation_a.srt",
            f"{sample}/speaker_1_audio.wav",
            f"{sample}/speaker_2_audio.wav",
        ]
        sample_size = sum(files.get(path.rsplit("/", 1)[-1], 0) for path in sample_paths)
        if sample_size <= 0:
            continue
        if used + sample_size > budget_bytes:
            continue
        selected.extend(sample_paths)
        used += sample_size
    if not selected:
        raise RuntimeError(f"no OtoSpeech samples fit within max_download_gb={max_download_gb}")
    return selected


def otospeech_sample_keys_from_patterns(allow_patterns: list[str]) -> set[str]:
    return {
        pattern.rsplit("/", 1)[0]
        for pattern in allow_patterns
        if "/" in pattern and pattern.endswith("speaker_1_audio.wav")
    }


def download_otospeech_dataset(
    repo_id: str = OTOSPEECH_REPO_ID,
    local_dir: Path | None = None,
    max_download_gb: float | None = 10.0,
    allow_patterns: list[str] | None = None,
) -> Path:
    if allow_patterns is None:
        allow_patterns = (
            otospeech_limited_allow_patterns(repo_id, max_download_gb)
            if max_download_gb is not None
            else OTOSPEECH_ALLOW_PATTERNS
        )
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "allow_patterns": allow_patterns,
    }
    if local_dir is not None:
        kwargs["local_dir"] = str(local_dir)
    return Path(snapshot_download(**kwargs))


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


def _null_condition_metrics() -> dict[str, float | None]:
    return {name: None for name in REFERENCE_METRIC_KEYS}


def _null_reference_row(key: str, status: str, errors: dict[str, str] | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "key": key,
        "status": status,
        "all": _null_condition_metrics(),
        "non_overlap": _null_condition_metrics(),
        "overlap": _null_condition_metrics(),
    }
    if errors:
        row["metric_errors"] = errors
    return row


def _safe_db(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-8) -> float:
    value = 10.0 * torch.log10((numerator.float().sum() + eps) / (denominator.float().sum() + eps))
    return float(value.detach().cpu().item())


def _si_sdr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float | None:
    estimate = estimate.reshape(-1).float()
    target = target.reshape(-1).float()
    if estimate.numel() == 0 or target.numel() == 0 or torch.sum(target ** 2).item() <= eps:
        return None
    estimate = estimate - estimate.mean()
    target = target - target.mean()
    scale = torch.dot(estimate, target) / (torch.dot(target, target) + eps)
    target_projection = scale * target
    noise = estimate - target_projection
    return _safe_db(target_projection ** 2, noise ** 2, eps)


def _sir_sar(
    estimate: torch.Tensor,
    target: torch.Tensor,
    interference: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[float | None, float | None]:
    estimate = estimate.reshape(-1).float()
    target = target.reshape(-1).float()
    interference = interference.reshape(-1).float()
    if estimate.numel() == 0 or torch.sum(target ** 2).item() <= eps:
        return None, None
    target_projection = (torch.dot(estimate, target) / (torch.dot(target, target) + eps)) * target
    if torch.sum(interference ** 2).item() <= eps:
        interference_projection = torch.zeros_like(target_projection)
    else:
        interference_projection = (
            torch.dot(estimate, interference) / (torch.dot(interference, interference) + eps)
        ) * interference
    artifact = estimate - target_projection - interference_projection
    sir = _safe_db(target_projection ** 2, interference_projection ** 2, eps)
    sar = _safe_db((target_projection + interference_projection) ** 2, artifact ** 2, eps)
    return sir, sar


def _best_permutation(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, list[int], float | None]:
    scores = []
    for perm in ([0, 1], [1, 0]):
        values = [_si_sdr(pred[perm[idx]], gt[idx]) for idx in range(2)]
        total = None if any(value is None for value in values) else float(sum(value for value in values if value is not None))
        scores.append((total, perm, values))
    best_total, best_perm, best_values = max(scores, key=lambda item: -math.inf if item[0] is None else item[0])
    aligned = torch.stack([pred[best_perm[0]], pred[best_perm[1]]], dim=0)
    mean_score = _mean(best_values)
    return aligned, list(best_perm), mean_score if mean_score is not None else best_total


def _activity_mask(track: torch.Tensor, sample_rate: int, threshold_db: float = -40.0, frame_ms: float = 20.0) -> torch.Tensor:
    track = track.reshape(-1).float()
    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    frames = torch.nn.functional.pad(track, (0, (-track.numel()) % frame)).reshape(-1, frame)
    rms = torch.sqrt(torch.mean(frames ** 2, dim=1) + 1e-12)
    threshold = max(10 ** (threshold_db / 20.0), float(torch.quantile(rms, 0.75).item()) * 0.1)
    active_frames = rms > threshold
    return active_frames.repeat_interleave(frame)[: track.numel()]


def _segments_from_mask(mask: torch.Tensor, sample_rate: int, frame_ms: float = 20.0) -> list[dict[str, float]]:
    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    frame_mask = mask[: (mask.numel() // frame) * frame]
    if frame_mask.numel() == 0:
        return []
    active = frame_mask.reshape(-1, frame).any(dim=1).tolist()
    segments: list[dict[str, float]] = []
    start = None
    for idx, is_active in enumerate(active):
        if is_active and start is None:
            start = idx
        elif not is_active and start is not None:
            segments.append({"start": start * frame / sample_rate, "end": idx * frame / sample_rate})
            start = None
    if start is not None:
        segments.append({"start": start * frame / sample_rate, "end": len(active) * frame / sample_rate})
    return segments


def _f1(pred: torch.Tensor, target: torch.Tensor) -> tuple[float | None, float | None, float | None]:
    pred = pred.bool()
    target = target.bool()
    tp = torch.logical_and(pred, target).sum().item()
    fp = torch.logical_and(pred, ~target).sum().item()
    fn = torch.logical_and(~pred, target).sum().item()
    if tp + fp + fn == 0:
        return None, None, None
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _boundary_mae(pred_mask: torch.Tensor, gt_mask: torch.Tensor, sample_rate: int) -> tuple[float | None, float | None]:
    pred_segments = _segments_from_mask(pred_mask, sample_rate)
    gt_segments = _segments_from_mask(gt_mask, sample_rate)
    if not pred_segments or not gt_segments:
        return None, None
    onset_errors = []
    offset_errors = []
    for gt_segment in gt_segments:
        pred_segment = min(
            pred_segments,
            key=lambda segment: abs(segment["start"] - gt_segment["start"]),
        )
        onset_errors.append(abs(pred_segment["start"] - gt_segment["start"]))
        offset_errors.append(abs(pred_segment["end"] - gt_segment["end"]))
    return _mean(onset_errors), _mean(offset_errors)


def _maybe_stoi(pred: torch.Tensor, gt: torch.Tensor, sample_rate: int) -> float | None:
    try:
        from pystoi import stoi
    except Exception:  # noqa: BLE001
        return None
    values = []
    for idx in range(2):
        try:
            values.append(float(stoi(gt[idx].numpy(), pred[idx].numpy(), sample_rate, extended=True)))
        except Exception:  # noqa: BLE001
            values.append(None)
    return _mean(values)


def _maybe_pesq(pred: torch.Tensor, gt: torch.Tensor, sample_rate: int) -> float | None:
    try:
        from pesq import pesq
    except Exception:  # noqa: BLE001
        return None
    if sample_rate not in {8000, 16000}:
        return None
    mode = "wb" if sample_rate == 16000 else "nb"
    values = []
    for idx in range(2):
        try:
            values.append(float(pesq(sample_rate, gt[idx].numpy(), pred[idx].numpy(), mode)))
        except Exception:  # noqa: BLE001
            values.append(None)
    return _mean(values)


def _metric_for_condition(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mixture: torch.Tensor,
    sample_rate: int,
    mask: torch.Tensor,
) -> dict[str, float | None]:
    result = _null_condition_metrics()
    if mask.sum().item() < sample_rate * 0.05:
        return result
    pred_m = pred[:, mask]
    gt_m = gt[:, mask]
    mix_m = mixture[:, mask].repeat(2, 1)
    result["pit_si_sdr"] = _mean([_si_sdr(pred_m[idx], gt_m[idx]) for idx in range(2)])
    mixture_si_sdr = _mean([_si_sdr(mix_m[idx], gt_m[idx]) for idx in range(2)])
    if result["pit_si_sdr"] is not None and mixture_si_sdr is not None:
        result["delta_si_sdr"] = result["pit_si_sdr"] - mixture_si_sdr
    sir_sar = [_sir_sar(pred_m[idx], gt_m[idx], gt_m[1 - idx]) for idx in range(2)]
    result["sir"] = _mean([item[0] for item in sir_sar])
    result["sar"] = _mean([item[1] for item in sir_sar])
    result["stoi"] = _maybe_stoi(pred_m, gt_m, sample_rate)
    result["pesq"] = _maybe_pesq(pred_m, gt_m, sample_rate)
    return result


def _crosstalk_metrics(
    pred_masks: list[torch.Tensor],
    gt_masks: list[torch.Tensor],
    threshold_db: float,
) -> dict[str, float | None]:
    rates = []
    leakages = []
    for idx in range(2):
        target = gt_masks[idx]
        denom = target.sum().item()
        if denom == 0:
            continue
        unwanted = torch.logical_and(pred_masks[1 - idx], target)
        target_count = pred_masks[idx].float().sum() + 1e-8
        leakage_count = unwanted.float().sum() + 1e-8
        leakage_db = 10.0 * math.log10(float(leakage_count / target_count))
        rates.append(unwanted.sum().item() / denom if leakage_db >= threshold_db else 0.0)
        leakages.append(leakage_db)
    return {
        "crosstalk_rate": _mean(rates),
        "leakage_p50_db": statistics.median(leakages) if leakages else None,
        "leakage_p95_db": max(leakages) if leakages else None,
    }


def _resolve_prediction_paths(key: str, pred_root: Path) -> tuple[Path, Path] | None:
    sample_dir = Path(pred_root) / key
    candidates = [
        (sample_dir / "speaker_A.wav", sample_dir / "speaker_B.wav"),
        (sample_dir / "speaker_0.wav", sample_dir / "speaker_1.wav"),
    ]
    for left, right in candidates:
        if left.is_file() and right.is_file():
            return left, right
    return None


def _resolve_gt_paths(sample: dict[str, Any]) -> tuple[Path, Path]:
    first = sample.get("gt_speaker_1") or sample.get("gt_agent")
    second = sample.get("gt_speaker_2") or sample.get("gt_ctm")
    if first is None or second is None:
        raise KeyError("sample must include gt_speaker_1/gt_speaker_2 or gt_agent/gt_ctm")
    return Path(str(first)), Path(str(second))


def score_reference_sample(
    sample: dict[str, Any],
    pred_root: Path,
    target_sample_rate: int = 16000,
    vad_threshold_db: float = -40.0,
    crosstalk_threshold_db: float = -20.0,
) -> dict[str, Any]:
    key = str(sample.get("key") or Path(str(sample.get("gt_speaker_1") or sample.get("gt_agent", "sample"))).parent.name)
    try:
        gt_first, gt_second = _resolve_gt_paths(sample)
        mixture, gt, sample_rate = mix_ground_truth_pair(
            gt_first,
            gt_second,
            target_sample_rate,
        )
    except Exception as exc:  # noqa: BLE001
        return _null_reference_row(key, "missing_ground_truth", {"ground_truth": str(exc)})

    pred_paths = _resolve_prediction_paths(key, Path(pred_root))
    if pred_paths is None:
        row = _null_reference_row(key, "missing_prediction", {"prediction": "speaker prediction files missing"})
        row["duration_sec"] = gt.shape[-1] / sample_rate
        return row

    try:
        pred_a, _ = load_mono_audio(pred_paths[0], sample_rate)
        pred_b, _ = load_mono_audio(pred_paths[1], sample_rate)
    except Exception as exc:  # noqa: BLE001
        return _null_reference_row(key, "missing_prediction", {"prediction": str(exc)})

    pred_a, pred_b = align_pair_lengths(pred_a, pred_b)
    length = min(pred_a.shape[-1], gt.shape[-1], mixture.shape[-1])
    pred = torch.cat([pred_a[:, :length], pred_b[:, :length]], dim=0)
    gt = gt[:, :length]
    mixture = mixture[:, :length]
    pred, permutation, pit_score = _best_permutation(pred, gt)

    gt_masks = [_activity_mask(gt[idx], sample_rate, vad_threshold_db) for idx in range(2)]
    pred_masks = [_activity_mask(pred[idx], sample_rate, vad_threshold_db) for idx in range(2)]
    all_mask = torch.ones(length, dtype=torch.bool)
    overlap_mask = torch.logical_and(gt_masks[0], gt_masks[1])
    non_overlap_mask = torch.logical_xor(gt_masks[0], gt_masks[1])

    row: dict[str, Any] = {
        "key": key,
        "status": "ok",
        "duration_sec": length / sample_rate,
        "permutation": permutation,
        "pit_si_sdr_for_permutation": pit_score,
        "prediction": [str(pred_paths[0]), str(pred_paths[1])],
        "ground_truth": [str(gt_first), str(gt_second)],
    }
    for condition, mask in [
        ("all", all_mask),
        ("non_overlap", non_overlap_mask),
        ("overlap", overlap_mask),
    ]:
        row[condition] = _metric_for_condition(pred, gt, mixture, sample_rate, mask)

    vad_scores = [_f1(pred_masks[idx], gt_masks[idx]) for idx in range(2)]
    onset_offsets = [_boundary_mae(pred_masks[idx], gt_masks[idx], sample_rate) for idx in range(2)]
    row["all"]["vad_f1"] = _mean([score[2] for score in vad_scores])
    row["all"]["onset_mae"] = _mean([item[0] for item in onset_offsets])
    row["all"]["offset_mae"] = _mean([item[1] for item in onset_offsets])
    pred_overlap = torch.logical_and(pred_masks[0], pred_masks[1])
    _, _, overlap_f1 = _f1(pred_overlap, overlap_mask)
    union = torch.logical_or(pred_overlap, overlap_mask).sum().item()
    row["overlap"]["overlap_f1"] = overlap_f1
    row["overlap"]["overlap_iou"] = (
        torch.logical_and(pred_overlap, overlap_mask).sum().item() / union if union else None
    )
    row["overlap"].update(_crosstalk_metrics(pred_masks, gt_masks, crosstalk_threshold_db))
    return row


def _read_reference_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil((percentile / 100.0) * len(ordered)) - 1))
    return ordered[index]


def write_reference_reports(rows: list[dict[str, Any]], output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample_metrics_path = output.parent / "sample_metrics.jsonl"
    with sample_metrics_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_rows = []
    for condition, target, metric_name, meaning, metric_key in REFERENCE_SUMMARY_ROWS:
        values = [
            float(row.get(condition, {}).get(metric_key))
            for row in rows
            if row.get(condition, {}).get(metric_key) is not None
        ]
        summary_rows.append(
            {
                "condition": condition,
                "target": target,
                "metric": metric_name,
                "meaning": meaning,
                "mean": _mean(values),
                "median": statistics.median(values) if values else None,
                "p95": _percentile(values, 95),
                "available": len(values),
            }
        )
    summary = {
        "samples": len(rows),
        "status_counts": {
            status: sum(1 for row in rows if row.get("status") == status)
            for status in sorted({str(row.get("status")) for row in rows})
        },
        "sample_metrics_jsonl": str(sample_metrics_path),
        "summary": summary_rows,
    }
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md_path = output.parent / "summary.md"
    lines = [
        "# IPPC Reference Benchmark",
        "",
        "| Condition | Mục tiêu | Metric chính | Ý nghĩa | Mean | Median | P95 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["condition"]),
                    str(row["target"]),
                    str(row["metric"]),
                    str(row["meaning"]),
                    _format_table_value(row["mean"]),
                    _format_table_value(row["median"]),
                    _format_table_value(row["p95"]),
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_reference_benchmark(
    samples: list[dict[str, Any]],
    pred_root: Path,
    output: Path,
    target_sample_rate: int,
    vad_threshold_db: float,
    crosstalk_threshold_db: float,
) -> None:
    rows = [
        score_reference_sample(
            sample,
            pred_root=pred_root,
            target_sample_rate=target_sample_rate,
            vad_threshold_db=vad_threshold_db,
            crosstalk_threshold_db=crosstalk_threshold_db,
        )
        for sample in samples
    ]
    write_reference_reports(rows, output)


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
    parser.add_argument("--reference-manifest", type=Path, default=None, help="JSONL manifest with gt_agent/gt_ctm rows.")
    parser.add_argument("--ippc-root", type=Path, default=None, help="IPPC root containing pairs/pair_* ground-truth folders.")
    parser.add_argument("--otospeech-root", type=Path, default=None, help="Local OtoSpeech snapshot root with per-sample metadata and speaker WAVs.")
    parser.add_argument("--download-otospeech", action="store_true", help="Download a capped OtoSpeech subset before benchmarking.")
    parser.add_argument("--otospeech-repo", default=OTOSPEECH_REPO_ID, help="Hugging Face dataset repo for OtoSpeech.")
    parser.add_argument("--otospeech-local-dir", type=Path, default=None, help="Local download/cache directory for OtoSpeech.")
    parser.add_argument("--max-download-gb", type=float, default=10.0, help="Maximum OtoSpeech download size in GB. Use <=10 for the requested cap.")
    parser.add_argument("--pred-root", type=Path, default=Path("outputs/otospeech"), help="Prediction root keyed by sample id.")
    parser.add_argument("--speakerA", type=Path, help="Path to speaker A WAV/audio file.")
    parser.add_argument("--speakerB", type=Path, help="Path to speaker B WAV/audio file.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path. Defaults to speakerA parent / benchmark.json.")
    parser.add_argument("--device", default="auto", help="Benchmark device: auto, cpu, cuda, cuda:N.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Reference benchmark sample rate.")
    parser.add_argument("--vad-threshold-db", type=float, default=-40.0, help="Energy VAD threshold in dBFS.")
    parser.add_argument("--crosstalk-threshold-db", type=float, default=-20.0, help="Leakage threshold in dB.")
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
    if (
        args.reference_manifest is not None
        or args.ippc_root is not None
        or args.otospeech_root is not None
        or args.download_otospeech
    ):
        if args.reference_manifest is not None:
            samples = _read_reference_manifest(args.reference_manifest)
        elif args.download_otospeech:
            local_dir = download_otospeech_dataset(
                repo_id=args.otospeech_repo,
                local_dir=args.otospeech_local_dir,
                max_download_gb=args.max_download_gb,
            )
            samples = discover_otospeech_samples(local_dir)
        elif args.otospeech_root is not None:
            samples = discover_otospeech_samples(args.otospeech_root)
        elif args.ippc_root is not None:
            samples = discover_ippc_pairs(args.ippc_root)
        else:
            samples = []
        output = args.output or Path("reports/otospeech_benchmark/summary.json")
        run_reference_benchmark(
            samples,
            pred_root=args.pred_root,
            output=output,
            target_sample_rate=args.sample_rate,
            vad_threshold_db=args.vad_threshold_db,
            crosstalk_threshold_db=args.crosstalk_threshold_db,
        )
        print(f"Saved reference benchmark: {output}")
        print(f"Saved sample metrics: {output.parent / 'sample_metrics.jsonl'}")
        print(f"Saved summary table: {output.parent / 'summary.md'}")
        return
    if args.single:
        _run_single(args)
        return
    raise SystemExit("Use --single, --reference-manifest, --otospeech-root, --download-otospeech, or --ippc-root. Use duplexchat-pipe run --phase benchmark for WebDataset.")


if __name__ == "__main__":
    main()
