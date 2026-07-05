from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from huggingface_hub import get_token

from duplexchat_pipe.audio import load_wav_tensor

if TYPE_CHECKING:
    from pyannote.audio import Pipeline


def load_diarization_pipeline(model: str, device: str = "cuda") -> "Pipeline":
    token = get_token()
    if token is None:
        raise RuntimeError(
            "Hugging Face token not found. Please login via huggingface-cli or set a token."
        )

    try:
        from pyannote.audio import Pipeline
    except Exception as exc:
        raise RuntimeError("pyannote.audio is required for diarization") from exc

    try:
        pipeline = Pipeline.from_pretrained(model, use_auth_token=token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(model, token=token)

    resolved_device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
    pipeline.to(torch.device(resolved_device))
    return pipeline


def run_diarization(pipeline: "Pipeline", wav_path: Path) -> list[dict]:
    waveform, sample_rate = load_wav_tensor(wav_path)
    output = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    segments: list[dict] = []

    # pyannote/speaker-diarization-community-1 returns a DiarizeOutput with
    # a .speaker_diarization iterable of (turn, speaker) pairs.
    # Older pipelines return an Annotation with .itertracks(yield_label=True).
    if hasattr(output, "speaker_diarization"):
        for turn, speaker in output.speaker_diarization:
            segments.append({
                "speaker": str(speaker),
                "start": float(turn.start),
                "end": float(turn.end),
            })
    else:
        for turn, _, speaker in output.itertracks(yield_label=True):
            segments.append({
                "speaker": str(speaker),
                "start": float(turn.start),
                "end": float(turn.end),
            })

    segments.sort(key=lambda s: s["start"])
    return segments
