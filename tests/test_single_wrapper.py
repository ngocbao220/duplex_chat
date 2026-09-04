import importlib.util
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
    monkeypatch.setattr("sys.argv", ["single.py", "--input", str(input_path)])

    single.main()

    assert calls[0][1]["output_dir"] == str(Path("outputs") / "demo1")
    assert (tmp_path / "outputs" / "demo1" / "run.json").exists()
