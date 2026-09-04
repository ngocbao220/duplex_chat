from __future__ import annotations

DIARIZATION_MODELS = {
    "sortformer": "nvidia/diar_sortformer_4spk-v1",
    "pyannote-3.1": "pyannote/speaker-diarization-3.1",
    "diarizen": "BUT-FIT/diarizen-wavlm-large-s80-md",
}

SEPARATION_MODELS = {
    "dialoguesidon": "sarulab-speech/DialogueSidon",
    "sepformer": "speechbrain/sepformer-wsj02mix",
    "mossformer2": "alibabasglab/MossFormer2_SS_16K",
}


def resolve_model_alias(model: str | None, aliases: dict[str, str]) -> str | None:
    if model is None:
        return None
    return aliases.get(model.strip().lower(), model)


def infer_diarization_backend(model: str) -> str:
    model_norm = model.strip().lower()
    if model_norm.startswith("nvidia/diar_sortformer"):
        return "sortformer"
    if model_norm.startswith("but-fit/diarizen"):
        return "diarizen"
    return "pyannote"
