import gzip
import json
import tarfile
from io import BytesIO
from pathlib import Path

import torch
import pytest
import torchaudio

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
                "dnsmos_model_path": "DNSMOS/sig_bak_ovr.onnx",
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
    assert cfg.benchmark_dnsmos_model_path == Path("DNSMOS/sig_bak_ovr.onnx")
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

    class FakeDNSMOSScorer:
        unavailable_reason = None

        def score(self, waveform, sample_rate):
            return {"dnsmos": [3.5, 3.0]}

    class FakeSubjectiveScorer:
        unavailable_reason = None

        def score(self, waveform, sample_rate):
            return {"squim_mos": [4.2, 3.1]}

    class FakeEmbeddingScorer:
        unavailable_reason = None

        def score(self, waveform, sample_rate):
            return {"itc": 0.8, "itd": 0.6}

    monkeypatch.setattr(benchmark, "DNSMOSScorer", lambda cfg: FakeDNSMOSScorer())
    monkeypatch.setattr(benchmark, "SquimObjectiveScorer", lambda cfg: FakeObjectiveScorer())
    monkeypatch.setattr(benchmark, "SquimSubjectiveScorer", lambda cfg: FakeSubjectiveScorer())
    monkeypatch.setattr(benchmark, "EmbeddingScorer", lambda cfg: FakeEmbeddingScorer())

    cfg = Config(
        benchmark_metrics=list(benchmark.METRIC_NAMES),
        benchmark_squim_subjective_enabled=True,
        benchmark_squim_subjective_reference_path=tmp_path / "reference.wav",
        benchmark_output_dir=tmp_path / "reports",
        log_root=tmp_path / "logs",
        run_id="bench",
    )
    benchmark.run_benchmark(tmp_path / "wds", tmp_path / "reports", cfg)

    row = json.loads((tmp_path / "reports" / "metrics.jsonl").read_text().splitlines()[0])
    assert row["dnsmos_mean"] == 3.25
    assert row["sq_stoi_mean"] == 0.815
    assert row["sq_pesq_A"] == 3.4
    assert row["sq_si_sdr_B"] == 8.0
    assert row["squim_mos_mean"] == pytest.approx(3.65)
    assert row["itc"] == 0.8
    assert row["itd"] == 0.6
    assert row["metric_status"] == "ok"
    assert (tmp_path / "reports" / "metrics_table.md").exists()
    assert (tmp_path / "reports" / "turn_taking_table.md").exists()


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


def test_single_speaker_cli_writes_benchmark_json(monkeypatch, tmp_path: Path):
    spk_a = tmp_path / "speaker_A.wav"
    spk_b = tmp_path / "speaker_B.wav"
    out_json = tmp_path / "benchmark.json"
    torchaudio.save(str(spk_a), torch.ones(1, 16000), 16000)
    torchaudio.save(str(spk_b), torch.zeros(1, 16000), 16000)

    monkeypatch.setattr(
        benchmark,
        "score_waveform",
        lambda waveform, sample_rate, cfg, key, separation_backend: {
            "key": key,
            "duration_sec": waveform.shape[-1] / sample_rate,
            "separation_backend": separation_backend,
            "sq_stoi_A": 0.9,
            "sq_stoi_B": 0.8,
            "sq_stoi_mean": 0.85,
            "metric_status": "ok",
        },
    )

    args = benchmark._build_parser().parse_args(
        [
            "--single",
            "--speakerA",
            str(spk_a),
            "--speakerB",
            str(spk_b),
            "--output",
            str(out_json),
            "--metrics",
            "sq_stoi",
        ]
    )
    benchmark._run_single(args)

    row = json.loads(out_json.read_text())
    assert row["key"] == "speaker_A"
    assert row["sq_stoi_mean"] == 0.85
    assert out_json.with_suffix(".metrics.md").exists()
    assert out_json.with_suffix(".turn_taking.md").exists()


def test_single_cli_default_metrics_exclude_dnsmos():
    args = benchmark._build_parser().parse_args(["--single"])

    assert "dnsmos" not in args.metrics
    assert "sq_stoi" in args.metrics


def test_compute_turn_taking_stats_from_stereo_activity():
    sample_rate = 1000
    waveform = torch.zeros(2, sample_rate * 4)
    waveform[0, 0:1000] = 0.5
    waveform[1, 800:1400] = 0.5
    waveform[0, 2000:3000] = 0.5
    waveform[1, 3000:3600] = 0.5

    stats = benchmark.compute_turn_taking_stats(waveform, sample_rate)

    assert stats["turn_exchanges_per_min"] is not None
    assert stats["turn_exchanges_per_min"] > 0
    assert stats["mean_turn_duration_sec"] is not None
    assert stats["simultaneous_speech_pct"] is not None
    assert stats["simultaneous_speech_pct"] > 0
    assert stats["overlapping_transitions_pct"] is not None


def test_mix_ground_truth_pair_sums_agent_and_ctm(tmp_path: Path):
    agent = tmp_path / "sample-AGENT.wav"
    ctm = tmp_path / "sample-CTM.wav"
    torchaudio.save(str(agent), torch.ones(1, 16000) * 0.25, 16000)
    torchaudio.save(str(ctm), torch.ones(1, 16000) * 0.5, 16000)

    mixture, gt, sample_rate = benchmark.mix_ground_truth_pair(agent, ctm, 16000)

    assert sample_rate == 16000
    assert gt.shape == (2, 16000)
    assert mixture.shape == (1, 16000)
    assert torch.allclose(gt[0], torch.full((16000,), 0.25))
    assert torch.allclose(gt[1], torch.full((16000,), 0.5), atol=1e-4)
    assert torch.allclose(mixture[0], torch.full((16000,), 0.75), atol=1e-4)


def test_discover_ippc_pairs_finds_agent_and_ctm(tmp_path: Path):
    pair_dir = tmp_path / "pairs" / "pair_001"
    pair_dir.mkdir(parents=True)
    agent = pair_dir / "call_01-AGENT.wav"
    ctm = pair_dir / "call_02-CTM.wav"
    torchaudio.save(str(agent), torch.zeros(1, 100), 1000)
    torchaudio.save(str(ctm), torch.zeros(1, 100), 1000)

    rows = benchmark.discover_ippc_pairs(tmp_path)

    assert rows == [
        {
            "key": "pair_001",
            "gt_agent": str(agent),
            "gt_ctm": str(ctm),
        }
    ]


def test_discover_otospeech_samples_finds_speaker_streams(tmp_path: Path):
    sample_dir = tmp_path / "split" / "sample_001"
    sample_dir.mkdir(parents=True)
    metadata = sample_dir / "metadata.json"
    speaker_1 = sample_dir / "speaker_1_audio.wav"
    speaker_2 = sample_dir / "speaker_2_audio.wav"
    (sample_dir / "speaker_1_annotation_a.srt").write_text("1\n", encoding="utf-8")
    (sample_dir / "speaker_2_annotation_a.srt").write_text("1\n", encoding="utf-8")
    metadata.write_text("{}", encoding="utf-8")
    torchaudio.save(str(speaker_1), torch.zeros(1, 100), 1000)
    torchaudio.save(str(speaker_2), torch.zeros(1, 100), 1000)

    rows = benchmark.discover_otospeech_samples(tmp_path)

    assert rows == [
        {
            "key": "split/sample_001",
            "metadata": str(metadata),
            "gt_speaker_1": str(speaker_1),
            "gt_speaker_2": str(speaker_2),
            "speaker_1_srt": str(sample_dir / "speaker_1_annotation_a.srt"),
            "speaker_2_srt": str(sample_dir / "speaker_2_annotation_a.srt"),
        }
    ]


def test_download_otospeech_dataset_respects_size_cap(monkeypatch):
    calls = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return "/tmp/otospeech"

    class FakeFile:
        def __init__(self, path: str, size: int):
            self.path = path
            self.size = size

    class FakeApi:
        def list_repo_tree(self, **_kwargs):
            return [
                FakeFile("sample_a/metadata.json", 100),
                FakeFile("sample_a/speaker_1_annotation_a.srt", 100),
                FakeFile("sample_a/speaker_2_annotation_a.srt", 100),
                FakeFile("sample_a/speaker_1_audio.wav", 4),
                FakeFile("sample_a/speaker_2_audio.wav", 4),
                FakeFile("sample_b/metadata.json", 100),
                FakeFile("sample_b/speaker_1_annotation_a.srt", 100),
                FakeFile("sample_b/speaker_2_annotation_a.srt", 100),
                FakeFile("sample_b/speaker_1_audio.wav", 7),
                FakeFile("sample_b/speaker_2_audio.wav", 7),
            ]

    monkeypatch.setattr(benchmark, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(benchmark, "HfApi", lambda: FakeApi())

    local_dir = benchmark.download_otospeech_dataset(max_download_gb=0.0000004)

    assert local_dir == Path("/tmp/otospeech")
    assert calls["repo_id"] == "otoearth/otoSpeech-full-duplex-turn-104h"
    assert calls["repo_type"] == "dataset"
    assert "sample_a/speaker_1_audio.wav" in calls["allow_patterns"]
    assert "sample_a/speaker_2_audio.wav" in calls["allow_patterns"]
    assert "sample_b/speaker_1_audio.wav" not in calls["allow_patterns"]


def test_reference_row_missing_prediction_keeps_metrics_null(tmp_path: Path):
    pair_dir = tmp_path / "pairs" / "pair_001"
    pair_dir.mkdir(parents=True)
    agent = pair_dir / "call-AGENT.wav"
    ctm = pair_dir / "call-CTM.wav"
    torchaudio.save(str(agent), torch.ones(1, 1600) * 0.2, 16000)
    torchaudio.save(str(ctm), torch.zeros(1, 1600), 16000)

    row = benchmark.score_reference_sample(
        {"key": "pair_001", "gt_speaker_1": str(agent), "gt_speaker_2": str(ctm)},
        pred_root=tmp_path / "pred",
        target_sample_rate=16000,
    )

    assert row["status"] == "missing_prediction"
    assert row["all"]["pit_si_sdr"] is None
    assert row["overlap"]["sir"] is None
    assert row["metric_errors"]["prediction"] == "speaker prediction files missing"


def test_write_reference_summary_markdown_includes_metric_meaning(tmp_path: Path):
    row = {
        "key": "pair_001",
        "status": "ok",
        "all": {
            "pit_si_sdr": 10.0,
            "sar": 20.0,
            "stoi": None,
            "pesq": None,
            "vad_f1": 0.9,
            "onset_mae": 0.05,
        },
        "overlap": {
            "sir": 12.0,
            "crosstalk_rate": 0.1,
            "overlap_f1": 0.8,
            "overlap_iou": 0.7,
        },
    }

    benchmark.write_reference_reports([row], tmp_path / "summary.json")

    text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "| Condition | Mục tiêu | Metric chính | Ý nghĩa | Mean | Median | P95 |" in text
    assert "Prediction giống ground truth đến mức nào" in text
    assert "Tỷ lệ frame bị lẫn speaker thứ hai" in text
