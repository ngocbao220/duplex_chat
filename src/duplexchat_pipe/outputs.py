from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import torch
import torchaudio


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return cleaned.strip("._") or "item"


def normalize_speaker_segments(segments: Iterable[dict]) -> tuple[list[dict], dict[str, str]]:
    mapping: dict[str, str] = {}
    normalized: list[dict] = []
    for segment in segments:
        original = str(segment["speaker"])
        if original not in mapping:
            mapping[original] = f"SPEAKER_{len(mapping):02d}"
        normalized.append(
            {
                **segment,
                "speaker": mapping[original],
                "original_speaker": original,
            }
        )
    return normalized, mapping


def format_label_time(value: float) -> str:
    return f"{float(value):.3f}"


def write_label_file(path: Path, rows: Iterable[tuple[float, float, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{format_label_time(start)}\t{format_label_time(end)}\t{label}"
        for start, end, label in rows
        if float(end) > float(start)
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def vad_segments_from_diarization(
    segments: Iterable[dict],
    duration_sec: float | None = None,
) -> list[dict]:
    spans = sorted(
        (max(0.0, float(s["start"])), max(0.0, float(s["end"])))
        for s in segments
        if float(s["end"]) > float(s["start"])
    )
    if not spans and duration_sec and duration_sec > 0:
        return [{"start": 0.0, "end": float(duration_sec), "label": "speech"}]
    merged: list[list[float]] = []
    for start, end in spans:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [{"start": start, "end": end, "label": "speech"} for start, end in merged]


def write_diarization_labels(
    labels_dir: Path,
    segments: Iterable[dict],
    duration_sec: float | None = None,
    vad_segments: Iterable[dict] | None = None,
) -> dict[str, Any]:
    labels_dir.mkdir(parents=True, exist_ok=True)
    normalized, speaker_mapping = normalize_speaker_segments(list(segments))
    by_speaker: dict[str, list[dict]] = {}
    for segment in normalized:
        by_speaker.setdefault(segment["speaker"], []).append(segment)

    write_label_file(
        labels_dir / "speakers.txt",
        ((s["start"], s["end"], s["speaker"]) for s in normalized),
    )
    for speaker, speaker_segments in by_speaker.items():
        write_label_file(
            labels_dir / f"{speaker}.txt",
            ((s["start"], s["end"], speaker) for s in speaker_segments),
        )

    vad = list(vad_segments) if vad_segments is not None else vad_segments_from_diarization(
        normalized, duration_sec
    )
    write_label_file(
        labels_dir / "vad.txt",
        ((s["start"], s["end"], str(s.get("label", "speech"))) for s in vad),
    )
    return {
        "labels_dir": str(labels_dir),
        "speakers": sorted(by_speaker),
        "speaker_mapping": speaker_mapping,
        "vad_segments": len(vad),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def copy_file(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    return dest


def save_wav(path: Path, wav: torch.Tensor, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), wav.cpu(), sample_rate)
    return path


def write_diarization_phase(
    run_dir: Path,
    segments: Iterable[dict],
    duration_sec: float | None = None,
    model: str | None = None,
    backend: str | None = None,
) -> Path:
    phase_dir = run_dir / "phase_02_diarization"
    segments_list = list(segments)
    label_meta = write_diarization_labels(phase_dir / "labels", segments_list, duration_sec)
    write_json(
        phase_dir / "diarization.json",
        {
            "model": model,
            "backend": backend,
            "duration_sec": duration_sec,
            "segments": segments_list,
            "labels": label_meta,
        },
    )
    return phase_dir


def write_dialogues_phase(run_dir: Path, dialogues: Iterable[Any]) -> Path:
    phase_dir = run_dir / "phase_03_dialogues"
    rows = []
    for idx, dialogue in enumerate(dialogues):
        rows.append(
            {
                "dialogue_idx": idx,
                "start": dialogue.start,
                "end": dialogue.end,
                "duration": dialogue.duration,
                "speakers": dialogue.speakers,
                "segments": dialogue.segments,
            }
        )
    write_json(phase_dir / "dialogues.json", rows)
    return phase_dir
