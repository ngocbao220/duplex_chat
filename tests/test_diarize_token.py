import pytest

import duplexchat_pipe.diarize as diarize


def test_missing_token(monkeypatch):
    monkeypatch.setattr(diarize, "get_token", lambda: None)
    with pytest.raises(RuntimeError, match="Hugging Face token not found"):
        diarize.load_diarization_pipeline("pyannote/speaker-diarization-3.1")


def test_infers_sortformer_backend(monkeypatch):
    called = {}

    def fake_load(model, device):
        called["model"] = model
        called["device"] = device
        return object()

    monkeypatch.setattr(diarize, "_load_sortformer_pipeline", fake_load)

    diarize.load_diarization_pipeline("sortformer", "cpu")

    assert called == {"model": "nvidia/diar_sortformer_4spk-v1", "device": "cpu"}


def test_infers_diarizen_backend(monkeypatch):
    called = {}

    def fake_load(model, device):
        called["model"] = model
        called["device"] = device
        return object()

    monkeypatch.setattr(diarize, "_load_diarizen_pipeline", fake_load)

    diarize.load_diarization_pipeline("diarizen", "cpu")

    assert called == {"model": "BUT-FIT/diarizen-wavlm-large-s80-md", "device": "cpu"}


def test_parse_sortformer_segments():
    output = [
        "0.10 1.20 speaker_0",
        (1.5, 2.25, "speaker_1"),
        {"start": 3.0, "end": 4.0, "speaker": "speaker_0"},
    ]

    assert diarize._segments_from_sortformer_output(output) == [
        {"speaker": "speaker_0", "start": 0.10, "end": 1.20},
        {"speaker": "speaker_1", "start": 1.5, "end": 2.25},
        {"speaker": "speaker_0", "start": 3.0, "end": 4.0},
    ]
