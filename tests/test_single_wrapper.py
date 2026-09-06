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
        (output_dir / "phase_00_input").mkdir(parents=True)
        (output_dir / "phase_00_input" / "input.json").write_text(
            json.dumps({"audio_name": Path(audio_path).name}),
            encoding="utf-8",
        )
        (output_dir / "phase_02_diarization").mkdir(parents=True)
        (output_dir / "phase_02_diarization" / "diarization.json").write_text(
            json.dumps({"segments": [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0}]}),
            encoding="utf-8",
        )
        labels_dir = output_dir / "phase_02_diarization" / "labels"
        labels_dir.mkdir(parents=True)
        (labels_dir / "vad.txt").write_text("0.000\t1.000\tspeech\n", encoding="utf-8")
        (labels_dir / "speakers.txt").write_text("0.000\t1.000\tSPEAKER_00\n", encoding="utf-8")
        (output_dir / "phase_04_separation").mkdir(parents=True)
        (output_dir / "phase_04_separation" / "separation.json").write_text(
            json.dumps({"sample_rate": 16000}),
            encoding="utf-8",
        )

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
            "--separate-chunk", "90", "--debug", "--runtime-device", "cpu",
        ],
    )

    single.main()

    assert Path(calls[0][1]["output_dir"]).name.startswith("duplexchat_single_")
    assert calls[0][1]["output_prefix"] == str(Path("outputs") / "demo1" / "speaker")
    assert calls[0][1]["diarize_chunk"] == 90
    assert calls[0][1]["separate_chunk"] == 90
    run_manifest = json.loads((tmp_path / "outputs" / "demo1" / "run.json").read_text())
    assert run_manifest["speakerA"] == str(Path("outputs") / "demo1" / "speakerA.wav")
    assert run_manifest["speakerB"] == str(Path("outputs") / "demo1" / "speakerB.wav")
    assert run_manifest["debug"] is True
    assert run_manifest["device"]["resolved"] == "cpu"
    assert run_manifest["labels"]["vad"] == str(Path("outputs") / "demo1" / "vad.txt")
    assert run_manifest["labels"]["diarization"] == str(Path("outputs") / "demo1" / "diarization.txt")
    assert run_manifest["labels"]["separation"] == str(Path("outputs") / "demo1" / "separation.txt")
    assert run_manifest["models"]["diarization"]["chunk_seconds"] == 90
    assert run_manifest["models"]["separation"]["chunk_seconds"] == 90
    assert run_manifest["phase_results"]["diarization"] == "Detected 1 diarization segments."
    assert (tmp_path / "outputs" / "demo1" / "vad.txt").read_text() == "0.000\t1.000\tspeech\n"
    assert (tmp_path / "outputs" / "demo1" / "diarization.txt").read_text() == "0.000\t1.000\tSPEAKER_00\n"
    assert (tmp_path / "outputs" / "demo1" / "separation.txt").read_text() == (
        "0.000\t1.000\tspeaker_A\n"
        "0.000\t1.000\tspeaker_B\n"
    )
    assert (tmp_path / "outputs" / "demo1" / "debug" / "phase_00_input").exists()
    assert (tmp_path / "outputs" / "demo1" / "debug" / "phase_02_diarization").exists()
    assert (tmp_path / "outputs" / "demo1" / "debug" / "phase_04_separation").exists()
