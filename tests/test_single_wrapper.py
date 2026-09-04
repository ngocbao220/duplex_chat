import importlib.util
import json
from pathlib import Path


spec = importlib.util.spec_from_file_location("single", Path("single.py"))
single = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(single)


def test_single_wrapper_defaults_output_dir_to_input_name(monkeypatch, tmp_path: Path):
    input_path = tmp_path / "kaggle" / "inputs" / "adasdasd" / "demo1.wav"
    input_path.parent.mkdir(parents=True)
    input_path.write_bytes(b"fake")
    calls = []

    def fake_run_single_audio(audio_path, **kwargs):
        calls.append((audio_path, kwargs))
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(single, "run_single_audio", fake_run_single_audio)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "single.py",
            "--input",
            str(input_path),
            "--diarize-chunk",
            "90",
            "--separate-chunk",
            "90",
        ],
    )

    single.main()

    assert calls[0][1]["output_dir"] == str(Path("outputs") / "demo1")
    assert calls[0][1]["diarize_chunk"] == 90
    assert calls[0][1]["separate_chunk"] == 90
    run_manifest = json.loads((tmp_path / "outputs" / "demo1" / "run.json").read_text())
    assert run_manifest["speaker_A"] == str(Path("outputs") / "demo1" / "speaker_A.wav")
    assert run_manifest["speaker_B"] == str(Path("outputs") / "demo1" / "speaker_B.wav")
    assert "speakerA" not in run_manifest
    assert "speakerB" not in run_manifest
