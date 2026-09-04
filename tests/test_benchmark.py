import gzip
import json
import tarfile
from io import BytesIO
from pathlib import Path

import torch
import pytest

from duplexchat_pipe import benchmark
from duplexchat_pipe.config import Config, apply_config_data, apply_overrides
from duplexchat_pipe.runner import run_phase


def _write_shard(path: Path, key: str = "sample") -> None:
    meta = {"dialogue_duration_sec": 1.0, "separation_backend": "dialoguesidon"}
    members = {
        f"{key}.meta.json": json.dumps(meta).encode("utf-8"),
        f"{key}.audio.mp3": b"fake audio bytes",
        f"{key}.diarization.json": json.dumps({"segments": []}).encode("utf-8"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as gz:
        with tarfile.open(fileobj=gz, mode="w|") as tf:
            for name, data in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tf.addfile(info, BytesIO(data))


def test_config_maps_benchmark_settings(tmp_path: Path):
    cfg = apply_config_data(
        Config(),
        {
            "benchmark": {
                "enabled": True,
                "metrics": ["sq_stoi", "sq_pesq"],
                "output_dir": "reports/test",
                "device": "cpu",
                "squim_subjective": {
                    "enabled": True,
                    "reference_path": "refs/clean.wav",
                },
                "squim_objective": {"enabled": False},
                "speaker_embedding_model": "embedding-model",
            }
        },
    )

    assert cfg.benchmark_enabled is True
    assert cfg.benchmark_metrics == ["sq_stoi", "sq_pesq"]
    assert cfg.benchmark_output_dir == Path("reports/test")
    assert cfg.benchmark_device == "cpu"
    assert cfg.benchmark_squim_subjective_enabled is True
    assert cfg.benchmark_squim_subjective_reference_path == Path("refs/clean.wav")
    assert cfg.benchmark_squim_objective_enabled is False
    assert cfg.benchmark_speaker_embedding_model == "embedding-model"


def test_benchmark_enabled_override():
    cfg = Config(benchmark_enabled=True)

    apply_overrides(cfg, ["benchmark.enabled=false"])

    assert cfg.benchmark_enabled is False


def test_run_phase_auto_runs_benchmark_after_end2end(monkeypatch, tmp_path: Path):
    calls = []
    cfg = Config(
        output_dir=tmp_path / "wds",
        benchmark_enabled=True,
        benchmark_output_dir=tmp_path / "reports",
        log_root=tmp_path / "logs",
        run_id="run-1",
    )

    monkeypatch.setattr("duplexchat_pipe.runner.crawl_and_build_dataset", lambda cfg: calls.append(("crawl", cfg.output_dir)))
    monkeypatch.setattr(
        "duplexchat_pipe.runner.run_benchmark",
        lambda input_dir, output_dir, cfg: calls.append(("benchmark", input_dir, output_dir, cfg.run_id)),
    )

    run_phase(cfg, "end2end")

    assert calls == [
        ("crawl", tmp_path / "wds"),
        ("benchmark", tmp_path / "wds", tmp_path / "reports", "run-1"),
    ]


def test_run_phase_uses_run_id_report_dir_for_default_benchmark_output(monkeypatch, tmp_path: Path):
    calls = []
    cfg = Config(
        output_dir=tmp_path / "wds",
        benchmark_enabled=True,
        log_root=tmp_path / "logs",
        run_id="run-1",
    )

    monkeypatch.setattr("duplexchat_pipe.runner.crawl_and_build_dataset", lambda cfg: calls.append(("crawl", cfg.output_dir)))
    monkeypatch.setattr(
        "duplexchat_pipe.runner.run_benchmark",
        lambda input_dir, output_dir, cfg: calls.append(("benchmark", input_dir, output_dir, cfg.run_id)),
    )

    run_phase(cfg, "end2end")

    assert calls == [
        ("crawl", tmp_path / "wds"),
        ("benchmark", tmp_path / "wds", Path("reports/run-1"), "run-1"),
    ]


def test_run_phase_skips_auto_benchmark_when_disabled(monkeypatch, tmp_path: Path):
    calls = []
    cfg = Config(output_dir=tmp_path / "wds", benchmark_enabled=False)

    monkeypatch.setattr("duplexchat_pipe.runner.crawl_and_build_dataset", lambda cfg: calls.append("crawl"))
    monkeypatch.setattr("duplexchat_pipe.runner.run_benchmark", lambda *args: calls.append("benchmark"))

    run_phase(cfg, "end2end")

    assert calls == ["crawl"]


def test_benchmark_writes_squim_and_track_metrics(monkeypatch, tmp_path: Path):
    _write_shard(tmp_path / "wds" / "000000.tar.gz")

    monkeypatch.setattr(
        benchmark,
        "decode_audio_bytes",
        lambda audio_bytes: (torch.stack([torch.ones(16000), torch.zeros(16000)]), 16000),
    )

    class FakeObjectiveScorer:
        unavailable_reason = None

        def score(self, waveform, sample_rate):
            return {
                "sq_stoi": [0.91, 0.72],
                "sq_pesq": [3.4, 2.1],
                "sq_si_sdr": [16.0, 8.0],
            }

    class FakeSubjectiveScorer:
        unavailable_reason = None

        def score(self, waveform, sample_rate):
            return {"squim_mos": [4.2, 3.1]}

    class FakeEmbeddingScorer:
        unavailable_reason = None

        def score(self, waveform, sample_rate):
            return {"itc": 0.8, "itd": 0.6}

    monkeypatch.setattr(benchmark, "SquimObjectiveScorer", lambda cfg: FakeObjectiveScorer())
    monkeypatch.setattr(benchmark, "SquimSubjectiveScorer", lambda cfg: FakeSubjectiveScorer())
    monkeypatch.setattr(benchmark, "EmbeddingScorer", lambda cfg: FakeEmbeddingScorer())

    cfg = Config(
        benchmark_squim_subjective_enabled=True,
        benchmark_squim_subjective_reference_path=tmp_path / "reference.wav",
        benchmark_output_dir=tmp_path / "reports",
        log_root=tmp_path / "logs",
        run_id="bench",
    )
    benchmark.run_benchmark(tmp_path / "wds", tmp_path / "reports", cfg)

    row = json.loads((tmp_path / "reports" / "metrics.jsonl").read_text().splitlines()[0])
    assert row["sq_stoi_mean"] == 0.815
    assert row["sq_pesq_A"] == 3.4
    assert row["sq_si_sdr_B"] == 8.0
    assert row["squim_mos_mean"] == pytest.approx(3.65)
    assert row["itc"] == 0.8
    assert row["itd"] == 0.6
    assert row["metric_status"] == "ok"


def test_benchmark_marks_unavailable_metric_without_failing(monkeypatch, tmp_path: Path):
    _write_shard(tmp_path / "wds" / "000000.tar.gz")
    monkeypatch.setattr(
        benchmark,
        "decode_audio_bytes",
        lambda audio_bytes: (torch.zeros(2, 16000), 16000),
    )

    class UnavailableObjectiveScorer:
        unavailable_reason = "SQUIM unavailable"

        def score(self, waveform, sample_rate):
            return {}

    class FakeEmbeddingScorer:
        unavailable_reason = None

        def score(self, waveform, sample_rate):
            return {"itc": 0.9, "itd": 0.7}

    monkeypatch.setattr(benchmark, "SquimObjectiveScorer", lambda cfg: UnavailableObjectiveScorer())
    monkeypatch.setattr(benchmark, "SquimSubjectiveScorer", lambda cfg: None)
    monkeypatch.setattr(benchmark, "EmbeddingScorer", lambda cfg: FakeEmbeddingScorer())

    cfg = Config(log_root=tmp_path / "logs", run_id="bench")
    benchmark.run_benchmark(tmp_path / "wds", tmp_path / "reports", cfg)

    row = json.loads((tmp_path / "reports" / "metrics.jsonl").read_text().splitlines()[0])
    assert row["sq_stoi_mean"] is None
    assert row["itc"] == 0.9
    assert row["metric_status"] == "partial"
    assert row["metric_errors"]["squim_objective"] == "SQUIM unavailable"
