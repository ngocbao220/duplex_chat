from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from duplexchat_pipe import outputs

from .models import ActivitySegment


def merge_activity_segments(
    segments: Iterable[ActivitySegment],
    merge_gap: float = 0.0,
) -> list[ActivitySegment]:
    ordered = sorted(segments, key=lambda item: (item.speaker, item.start, item.end))
    merged: list[ActivitySegment] = []
    for segment in ordered:
        if segment.end <= segment.start:
            continue
        if (
            merged
            and merged[-1].speaker == segment.speaker
            and segment.start - merged[-1].end <= merge_gap
        ):
            previous = merged.pop()
            merged.append(
                ActivitySegment(
                    start=previous.start,
                    end=max(previous.end, segment.end),
                    speaker=previous.speaker,
                    label=previous.label,
                )
            )
        else:
            merged.append(segment)
    return merged


def filter_min_duration(
    segments: Iterable[ActivitySegment],
    min_duration: float,
) -> list[ActivitySegment]:
    return [segment for segment in segments if segment.duration >= min_duration]


def write_vad_artifacts(
    json_path: Path,
    label_path: Path,
    segments: Iterable[ActivitySegment],
) -> None:
    rows = list(segments)
    outputs.write_json(json_path, [segment.to_dict() for segment in rows])
    outputs.write_label_file(
        label_path,
        ((segment.start, segment.end, "1") for segment in rows),
    )


def run_silero_vad(
    wav: torch.Tensor,
    sample_rate: int,
    speaker: int,
    threshold: float | None = None,
    offset: float | None = None,
    min_duration: float = 0.0,
    merge_gap: float = 0.0,
) -> list[ActivitySegment]:
    import torch

    vad_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
        verbose=False,
    )
    get_speech_timestamps = utils[0]
    audio = wav.detach().cpu().float()
    if audio.ndim > 1:
        audio = audio.mean(dim=0)

    kwargs = {"sampling_rate": sample_rate}
    if threshold is not None:
        kwargs["threshold"] = float(threshold)
    if offset is not None:
        kwargs["neg_threshold"] = float(offset)
    if min_duration > 0:
        kwargs["min_speech_duration_ms"] = int(min_duration * 1000)

    try:
        timestamps = get_speech_timestamps(audio, vad_model, **kwargs)
    except TypeError:
        kwargs.pop("neg_threshold", None)
        timestamps = get_speech_timestamps(audio, vad_model, **kwargs)
    segments = [
        ActivitySegment(
            start=float(ts["start"]) / float(sample_rate),
            end=float(ts["end"]) / float(sample_rate),
            speaker=speaker,
        )
        for ts in timestamps
    ]
    return merge_activity_segments(filter_min_duration(segments, min_duration), merge_gap)
