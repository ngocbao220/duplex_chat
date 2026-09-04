from __future__ import annotations

from collections.abc import Callable

import torch

from .models import Region
from .speaker_assignment import EmbeddingExtractor, assign_candidates


SeparatorFn = Callable[[torch.Tensor, int], tuple[torch.Tensor, torch.Tensor, int]]


def reconstruct_tracks(
    original: torch.Tensor,
    sample_rate: int,
    regions: list[Region],
    separator: SeparatorFn | None,
    references: dict[int, torch.Tensor],
    extractor: EmbeddingExtractor | None,
    cosine_threshold: float,
    overlap_padding: float,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    total = original.shape[-1]
    final = torch.zeros(2, total, dtype=original.dtype)
    overlap_records: list[dict] = []

    for region in regions:
        start = _sample(region.start, sample_rate, total)
        end = _sample(region.end, sample_rate, total)
        if end <= start or region.type == "silence":
            continue
        if region.type == "single_speaker":
            if region.speaker is not None:
                final[region.speaker, start:end] = original[0, start:end]
            continue
        if separator is None:
            raise RuntimeError("overlap regions require a separator")

        padded_start_sec = max(0.0, region.start - overlap_padding)
        padded_end_sec = min(total / sample_rate, region.end + overlap_padding)
        padded_start = _sample(padded_start_sec, sample_rate, total)
        padded_end = _sample(padded_end_sec, sample_rate, total)
        separated_0, separated_1, sep_sr = separator(original[:, padded_start:padded_end], sample_rate)
        if sep_sr != sample_rate:
            import torchaudio.functional as F_audio

            separated_0 = F_audio.resample(separated_0, sep_sr, sample_rate)
            separated_1 = F_audio.resample(separated_1, sep_sr, sample_rate)

        expected = padded_end - padded_start
        separated_0 = _fit_length(separated_0, expected)
        separated_1 = _fit_length(separated_1, expected)
        offset = start - padded_start
        length = end - start
        cand_0 = separated_0[:, offset : offset + length]
        cand_1 = separated_1[:, offset : offset + length]
        assigned, assignment = assign_candidates(
            cand_0,
            cand_1,
            sample_rate,
            references,
            extractor,
            cosine_threshold,
        )
        final[0, start:end] = assigned[0].reshape(-1)[:length]
        final[1, start:end] = assigned[1].reshape(-1)[:length]
        overlap_records.append(
            {
                **region.to_dict(),
                "padded_start": padded_start_sec,
                "padded_end": padded_end_sec,
                "assignment": assignment,
            }
        )
    return final[0:1], final[1:2], overlap_records


def _sample(value: float, sample_rate: int, total: int) -> int:
    return max(0, min(total, int(round(value * sample_rate))))


def _fit_length(wav: torch.Tensor, length: int) -> torch.Tensor:
    wav = wav.detach().cpu().float()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[-1] > length:
        return wav[:, :length]
    if wav.shape[-1] < length:
        return torch.nn.functional.pad(wav, (0, length - wav.shape[-1]))
    return wav
