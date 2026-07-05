import pytest

import duplexchat_pipe.diarize as diarize


def test_missing_token(monkeypatch):
    monkeypatch.setattr(diarize, "get_token", lambda: None)
    with pytest.raises(RuntimeError, match="Hugging Face token not found"):
        diarize.load_diarization_pipeline("pyannote/speaker-diarization-3.1")
