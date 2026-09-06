import importlib.util
from pathlib import Path

import torch

from duplexchat_pipe.config import Config
from duplexchat_pipe import pipeline


spec = importlib.util.spec_from_file_location("end2end", Path("end2end.py"))
end2end = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(end2end)


class FakePbar:
    def __init__(self):
        self.updates = 0
        self.postfix = ""

    def update(self, value):
        self.updates += value

    def set_postfix_str(self, value, refresh=True):
        self.postfix = value


class FakeDB:
    def __init__(self):
        self.marked = []

    def mark(self, *args):
        self.marked.append(args)


def test_end2end_does_not_override_config_target_hours(monkeypatch):
    cfg = Config(target_hours=1.0)
    seen = {}

    monkeypatch.setattr(end2end, "load_config", lambda path: cfg)
    monkeypatch.setattr(end2end, "run_phase", lambda cfg_arg, phase: seen.update(cfg=cfg_arg, phase=phase))
    monkeypatch.setattr("sys.argv", ["end2end.py"])

    end2end.main()

    assert seen["phase"] == "end2end"
    assert seen["cfg"].target_hours == 1.0


def test_end2end_overrides_target_hours_when_flag_is_passed(monkeypatch):
    cfg = Config(target_hours=1.0)
    seen = {}

    monkeypatch.setattr(end2end, "load_config", lambda path: cfg)
    monkeypatch.setattr(end2end, "run_phase", lambda cfg_arg, phase: seen.update(cfg=cfg_arg, phase=phase))
    monkeypatch.setattr("sys.argv", ["end2end.py", "--target_hours", "2"])

    end2end.main()

    assert seen["cfg"].target_hours == 2.0


def test_end2end_sets_youtube_only_when_flag_is_passed(monkeypatch):
    cfg = Config(youtube_only=False)
    seen = {}

    monkeypatch.setattr(end2end, "load_config", lambda path: cfg)
    monkeypatch.setattr(end2end, "run_phase", lambda cfg_arg, phase: seen.update({"cfg": cfg_arg, "phase": phase}))
    monkeypatch.setattr("sys.argv", ["end2end.py", "--youtube-only"])

    end2end.main()

    assert seen["cfg"].youtube_only is True
    assert seen["phase"] == "end2end"


def test_end2end_otospeech_download_mix_predict_and_benchmark(monkeypatch, tmp_path, capsys):
    cfg = Config(
        runtime_device="cpu",
        allow_cpu_fallback=True,
        separation_num_steps=2,
        benchmark_output_dir=tmp_path / "reports",
    )
    calls = []
    samples = [
        {
            "key": "train/sample_001",
            "gt_speaker_1": str(tmp_path / "s1.wav"),
            "gt_speaker_2": str(tmp_path / "s2.wav"),
        }
    ]

    monkeypatch.setattr(end2end, "load_config", lambda path: cfg)
    monkeypatch.setattr(
        end2end.benchmark,
        "download_otospeech_dataset",
        lambda repo_id, local_dir, max_download_gb: calls.append(("download", repo_id, local_dir, max_download_gb)) or tmp_path / "hf",
    )
    monkeypatch.setattr(end2end.benchmark, "discover_otospeech_samples", lambda root: calls.append(("discover", root)) or samples)
    monkeypatch.setattr(
        end2end.benchmark,
        "mix_ground_truth_pair",
        lambda s1, s2, sample_rate: calls.append(("mix", s1, s2, sample_rate)) or (torch.zeros(1, 160), torch.zeros(2, 160), sample_rate),
    )
    monkeypatch.setattr(end2end, "save_wav", lambda path, wav, sr: calls.append(("save_wav", path, tuple(wav.shape), sr)) or path)
    monkeypatch.setattr(end2end, "run_cholimex_file", lambda input_path, output_dir, cfg_arg: calls.append(("predict", input_path, output_dir, cfg_arg.runtime_device)) or {})
    monkeypatch.setattr(
        end2end.benchmark,
        "run_reference_benchmark",
        lambda samples, pred_root, output, target_sample_rate, vad_threshold_db, crosstalk_threshold_db: calls.append(
            ("benchmark", len(samples), pred_root, output, target_sample_rate)
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "end2end.py",
            "--data",
            "oto-speech",
            "--size_gb",
            "5",
            "--output-root",
            str(tmp_path / "outputs"),
        ],
    )

    end2end.main()

    assert calls[0][0] == "download"
    assert calls[0][3] == 5.0
    assert ("predict", tmp_path / "outputs" / "otospeech_mixtures" / "train" / "sample_001" / "mixture.wav", tmp_path / "outputs" / "otospeech" / "train" / "sample_001", "cpu") in calls
    assert calls[-1] == ("benchmark", 1, tmp_path / "outputs" / "otospeech", tmp_path / "reports" / "summary.json", 16000)
    output = capsys.readouterr().out
    assert "=====================Phase 1: Downloading OtoSpeech" in output
    assert "=====================Phase 2: Mixing speaker streams" in output
    assert "=====================Phase 3: Running pipeline" in output
    assert "=====================Phase 4: Running benchmark" in output


def test_handle_processed_result_skips_write_after_target_reached(monkeypatch):
    writes = []
    cfg = Config(target_hours=1.0, cleanup_audio_cache=False)
    stats = pipeline.Stats(total_duration_sec=3600.0)
    result = pipeline.ProcessedResult(
        key="sample",
        status="ok",
        meta={"duration_sec": 10.0},
        audio_bytes=b"audio",
        diarization=None,
        diarization_error=None,
    )

    monkeypatch.setattr(pipeline.wds, "write_sample", lambda *args: writes.append(args))
    pipeline._handle_processed_result(result, cfg, object(), FakeDB(), FakePbar(), stats)

    assert writes == []
    assert stats.written == 0
    assert stats.skipped == 1
