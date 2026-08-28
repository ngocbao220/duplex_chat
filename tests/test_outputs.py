from pathlib import Path

from duplexchat_pipe import outputs


def test_write_diarization_labels_uses_audacity_tab_format(tmp_path: Path):
    segments = [
        {"speaker": "raw_a", "start": 0.0, "end": 1.25},
        {"speaker": "raw_b", "start": 1.5, "end": 2.0},
        {"speaker": "raw_a", "start": 2.0, "end": 3.0},
    ]

    meta = outputs.write_diarization_labels(tmp_path / "labels", segments)

    assert meta["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
    assert (tmp_path / "labels" / "speakers.txt").read_text() == (
        "0.000\t1.250\tSPEAKER_00\n"
        "1.500\t2.000\tSPEAKER_01\n"
        "2.000\t3.000\tSPEAKER_00\n"
    )
    assert (tmp_path / "labels" / "SPEAKER_00.txt").read_text() == (
        "0.000\t1.250\tSPEAKER_00\n"
        "2.000\t3.000\tSPEAKER_00\n"
    )
    assert (tmp_path / "labels" / "SPEAKER_01.txt").read_text() == (
        "1.500\t2.000\tSPEAKER_01\n"
    )
    assert (tmp_path / "labels" / "vad.txt").read_text() == (
        "0.000\t1.250\tspeech\n"
        "1.500\t3.000\tspeech\n"
    )


def test_write_diarization_phase_writes_json_and_labels(tmp_path: Path):
    phase_dir = outputs.write_diarization_phase(
        tmp_path / "sample",
        [{"speaker": "A", "start": 0, "end": 1}],
        duration_sec=1.0,
        model="model-id",
        backend="pyannote",
    )

    assert phase_dir == tmp_path / "sample" / "phase_02_diarization"
    assert (phase_dir / "diarization.json").exists()
    assert (phase_dir / "labels" / "speakers.txt").read_text() == "0.000\t1.000\tSPEAKER_00\n"
