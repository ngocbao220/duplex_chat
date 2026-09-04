from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F
import torchaudio.functional as F_audio

from .models import Region


class EmbeddingExtractor(Protocol):
    def extract(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        ...


class SpeechBrainEmbeddingExtractor:
    def __init__(self, model_id: str, device: str = "cpu") -> None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ModuleNotFoundError:
            from speechbrain.inference.classifiers import EncoderClassifier

        self.device = device
        self.sample_rate = 16000
        self.model = EncoderClassifier.from_hparams(
            source=model_id,
            run_opts={"device": device},
        )

    def extract(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        prepared = wav.detach().cpu().float()
        if prepared.ndim > 1:
            prepared = prepared.mean(dim=0)
        if sample_rate != self.sample_rate:
            prepared = F_audio.resample(prepared.unsqueeze(0), sample_rate, self.sample_rate).squeeze(0)
        with torch.inference_mode():
            emb = self.model.encode_batch(prepared.unsqueeze(0).to(self.device))
        return emb.detach().cpu().reshape(-1).float()


def build_reference_embeddings(
    original: torch.Tensor,
    sample_rate: int,
    regions: list[Region],
    extractor: EmbeddingExtractor,
    min_reference_duration: float,
) -> dict[int, torch.Tensor]:
    refs: dict[int, list[torch.Tensor]] = {0: [], 1: []}
    for region in regions:
        if region.type != "single_speaker" or region.speaker is None:
            continue
        if region.duration < min_reference_duration:
            continue
        start = int(round(region.start * sample_rate))
        end = int(round(region.end * sample_rate))
        if end <= start:
            continue
        refs[region.speaker].append(extractor.extract(original[:, start:end], sample_rate))
    return {
        speaker: torch.stack(embeddings).mean(dim=0)
        for speaker, embeddings in refs.items()
        if embeddings
    }


def assign_candidates(
    candidate_0: torch.Tensor,
    candidate_1: torch.Tensor,
    sample_rate: int,
    references: dict[int, torch.Tensor],
    extractor: EmbeddingExtractor | None,
    threshold: float,
) -> tuple[dict[int, torch.Tensor], dict]:
    if extractor is None or set(references) != {0, 1}:
        return {0: candidate_0, 1: candidate_1}, {"method": "channel_order", "reason": "missing_reference"}

    emb_0 = extractor.extract(candidate_0, sample_rate)
    emb_1 = extractor.extract(candidate_1, sample_rate)

    direct = _score(emb_0, references[0]) + _score(emb_1, references[1])
    swapped = _score(emb_0, references[1]) + _score(emb_1, references[0])
    best = max(direct, swapped) / 2.0
    if best < threshold:
        return {
            0: candidate_0,
            1: candidate_1,
        }, {"method": "channel_order", "reason": "below_threshold", "score": best}
    if swapped > direct:
        return {0: candidate_1, 1: candidate_0}, {"method": "cosine", "swapped": True, "score": best}
    return {0: candidate_0, 1: candidate_1}, {"method": "cosine", "swapped": False, "score": best}


def _score(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.reshape(-1), right.reshape(-1), dim=0))
