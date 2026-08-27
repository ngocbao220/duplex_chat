import json
from pathlib import Path


def test_kaggle_notebook_has_model_audio_tests():
    notebook = json.loads(Path("notebook/kaggle.ipynb").read_text())
    source = "\n".join(
        cell.get("source", "")
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "MODEL_TESTS" in source
    assert "/kaggle/working/model_audio_tests" in source
    assert "nvidia/diar_sortformer_4spk-v1" in source
    assert "pyannote/speaker-diarization-3.1" in source
    assert "BUT-FIT/diarizen-wavlm-large-s80-md" in source
    assert "speechbrain/sepformer-wsj02mix" in source
    assert "alibabasglab/MossFormer2_SS_16K" in source
    assert "--output-prefix" in source
    assert "speaker_A.wav" in source
    assert "speaker_B.wav" in source
